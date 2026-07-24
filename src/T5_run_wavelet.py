#!/usr/bin/env python
# coding=utf-8

import logging
import os
import sys
import json
import time
from dataclasses import dataclass, field
from typing import Optional
from waveletLoRAAdapter import (
    LinearWaveletFilter,
    materialize_wavelet_lora_factors,
    wrap_peft_lora_factors,
)

import datasets
import nltk
import numpy as np
from datasets import load_dataset

import transformers
from filelock import FileLock
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForSeq2SeqLM,
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    Seq2SeqTrainingArguments,
    set_seed,
)
from transformers.file_utils import is_offline_mode
from transformers.trainer_utils import get_last_checkpoint
from peft import get_peft_config, get_peft_model, LoraConfig, TaskType, PeftModel, PeftConfig

from uie_collator import DataCollatorForUIE
from uie_dataset_lora import gen_cache_path
from uie_trainer_lora import UIETrainer, DenserEvalCallback, skip_instructions
from compute_metrics import compute_metrics, compute_grouped_metrics
from model.llama import LlamaForCausalLM_with_lossmask

import torch
import torch.nn as nn
import warnings

# Disable WANDB and warnings
os.environ['WANDB_DISABLED'] = "True"
warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)
CURRENT_DIR = os.path.dirname(__file__)
EXP_MODE = True

try:
    nltk.data.find("tokenizers/punkt")
except (LookupError, OSError):
    if is_offline_mode():
        raise LookupError("Offline mode: run this script without TRANSFORMERS_OFFLINE first to download nltk data files")
    with FileLock(".lock"):
        nltk.download("punkt", quiet=True)


@dataclass
class ModelArguments:
    model_name_or_path: str = field(metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"})
    config_name: Optional[str] = field(default=None)
    tokenizer_name: Optional[str] = field(default=None)
    cache_dir: Optional[str] = field(default=None)
    use_fast_tokenizer: bool = field(default=True)
    model_revision: str = field(default="main")
    use_auth_token: bool = field(default=False)
    resize_position_embeddings: Optional[bool] = field(default=None)
    lora_dim: Optional[int] = field(default=16)


@dataclass
class DataTrainingArguments:
    lang: str = field(default=None)
    data_dir: str = field(default=None)
    task_config_dir: str = field(default=None)
    instruction_file: str = field(default=None)
    instruction_strategy: Optional[str] = field(default='single')
    overwrite_cache: bool = field(default=False)
    input_record_file: str = field(default=None)
    preprocessing_num_workers: Optional[int] = field(default=None)
    max_source_length: Optional[int] = field(default=512)
    max_target_length: Optional[int] = field(default=50)
    repetition_penalty: Optional[float] = field(default=1.0)
    num_beams: Optional[int] = field(default=1)
    max_num_instances_per_task: int = field(default=10000)
    max_num_instances_per_eval_task: int = field(default=200)
    max_train_samples: Optional[int] = field(default=None)
    max_eval_samples: Optional[int] = field(default=None)
    max_predict_samples: Optional[int] = field(default=None)
    num_examples: Optional[int] = field(default=0)
    ignore_pad_token_for_loss: bool = field(default=True)
    add_task_name: Optional[bool] = field(default=False)
    add_dataset_name: Optional[bool] = field(default=False)


@dataclass
class UIETrainingArguments(Seq2SeqTrainingArguments):
    gradient_checkpointing: Optional[bool] = field(default=False)
    denser_evaluation: Optional[bool] = field(default=False)
    do_demo: bool = field(default=False)
    lambda1: float = field(default=0.01, metadata={"help": "Weight for wavelet theta regularization term"})
    lambda2: float = field(default=0.001, metadata={"help": "Weight for MLP regularization term"})
    theta_norm_p: float = field(default=5.0, metadata={"help": "Norm degree for theta (wavelet filter)"})
    mlp_norm_p: float = field(default=2.0, metadata={"help": "Norm degree for MLP parameters"})
    adapter_update_strategy: str = field(
        default="AB",
        metadata={"help": "Which LoRA factors to replace with wavelet filters: A, B, or AB"},
    )
    wavelet_kernel: str = field(
        default="heat",
        metadata={"help": "Kernel used in the DEAL wavelet filters; paper default is heat"},
    )

def extract_lora_params(peft_model):
    """
    Returns dictionaries containing references to the LoRA factor modules named 'lora_A' or 'lora_B'.
    """
    lora_A, lora_B = {}, {}
    for name, module in peft_model.named_modules():
        # In PEFT, the actual rank-weight is stored in something like module.lora_A.default.weight
        # The parent module is a LoRALayer, e.g., 'xxx.lora_A'
        if "lora_A.default" in name:
            lora_A[name] = module
        elif "lora_B.default" in name:
            lora_B[name] = module
    return lora_A, lora_B


def replace_lora_params(peft_model, new_lora_A, new_lora_B):
    for name, module in peft_model.named_modules():
        if 'lora_A.default' in name:
            parent_name, _, child_name = name.rpartition('.')
            parent_module = dict(peft_model.named_modules())[parent_name]
            parent_module._modules[child_name] = new_lora_A[name]
        elif 'lora_B.default' in name:
            parent_name, _, child_name = name.rpartition('.')
            parent_module = dict(peft_model.named_modules())[parent_name]
            parent_module._modules[child_name] = new_lora_B[name]
    return peft_model

def main():
    
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, UIETrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    training_args.local_rank = -1
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}, "
        f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # Checkpoint detection
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(f"Checkpoint detected, resuming training at {last_checkpoint}.")

    set_seed(training_args.seed)
    data_cache_dir = gen_cache_path(training_args.output_dir, data_args)

    raw_datasets = load_dataset(
        os.path.join(CURRENT_DIR, "uie_dataset_lora.py"),
        data_dir=data_args.data_dir,
        task_config_dir=data_args.task_config_dir,
        instruction_file=data_args.instruction_file,
        instruction_strategy=data_args.instruction_strategy,
        cache_dir=data_cache_dir,
        max_num_instances_per_task=data_args.max_num_instances_per_task,
        max_num_instances_per_eval_task=data_args.max_num_instances_per_eval_task,
        num_examples=data_args.num_examples
    )
    raw_datasets.cleanup_cache_files()

    # Load model & tokenizer
    if 'adapter' in model_args.model_name_or_path:
        config = PeftConfig.from_pretrained(model_args.model_name_or_path)
        tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path, use_fast=True)
    else:
        config = AutoConfig.from_pretrained(
            model_args.config_name or model_args.model_name_or_path,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.tokenizer_name or model_args.model_name_or_path,
            cache_dir=model_args.cache_dir,
            use_fast=model_args.use_fast_tokenizer,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
        )

    model_class = AutoModelForSeq2SeqLM

    if 'adapter' in model_args.model_name_or_path:
        model = model_class.from_pretrained(config.base_model_name_or_path)
        #model.resize_token_embeddings(len(tokenizer))
        model = PeftModel.from_pretrained(model, model_args.model_name_or_path)
        
    else:
        model = model_class.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
        )
        peft_config = LoraConfig(
            task_type=TaskType.SEQ_2_SEQ_LM,
            inference_mode=False,
            r=model_args.lora_dim,
            target_modules=["q", "v"],
            lora_alpha=32,
            lora_dropout=0.0,
        )
        model = get_peft_model(model, peft_config)

    # wavelet-based logic if adapter checkpoint
    if 'adapter' in model_args.model_name_or_path.lower():
        logger.info(
            "** wavelet-based wrapping: strategy=%s, kernel=%s **",
            training_args.adapter_update_strategy,
            training_args.wavelet_kernel,
        )
        num_wavelet_modules = wrap_peft_lora_factors(
            model,
            strategy=training_args.adapter_update_strategy,
            wavelet=training_args.wavelet_kernel,
            hidden_dim=128,
            device=training_args.device,
        )

        if num_wavelet_modules == 0:
            raise RuntimeError(
                "No LinearWaveletFilter modules were inserted. "
                "The wavelet regularization would be disabled."
            )
        
        logger.info(
            "Inserted %d LinearWaveletFilter modules.",
            num_wavelet_modules,
        )
     
    if (
            hasattr(model.config, "max_position_embeddings")
            and model.config.max_position_embeddings < data_args.max_source_length
    ):
        if model_args.resize_position_embeddings is None:
            logger.warning(
                f"Increasing the model's number of position embedding vectors from {model.config.max_position_embeddings} "
                f"to {data_args.max_source_length}."
            )
            model.resize_position_embeddings(data_args.max_source_length)
        elif model_args.resize_position_embeddings:
            model.resize_position_embeddings(data_args.max_source_length)
        else:
            raise ValueError(
                f"`--max_source_length` is set to {data_args.max_source_length}, but the model only has {model.config.max_position_embeddings}"
                f" position encodings. Consider either reducing `--max_source_length` to {model.config.max_position_embeddings} or to automatically "
                "resize the model's position encodings by passing `--resize_position_embeddings`."
            )

    # Prepare datasets
    if training_args.do_train:
        train_dataset = raw_datasets["train"]
        if data_args.max_train_samples:
            train_dataset = train_dataset.select(range(data_args.max_train_samples))
    if training_args.do_eval:
        eval_dataset = raw_datasets["validation"]
        if data_args.max_eval_samples:
            eval_dataset = eval_dataset.select(range(data_args.max_eval_samples))
    if training_args.do_predict:
        predict_dataset = raw_datasets["test"]
        if data_args.max_predict_samples:
            predict_dataset = predict_dataset.select(range(data_args.max_predict_samples))

    # Data collator
    label_pad_token_id = -100 if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id
    data_collator = DataCollatorForUIE(
        tokenizer,
        model=model,
        padding="longest",
        max_source_length=data_args.max_source_length,
        max_target_length=data_args.max_target_length,
        label_pad_token_id=label_pad_token_id,
        pad_to_multiple_of=8 if training_args.fp16 else None,
        add_task_name=data_args.add_task_name,
        add_dataset_name=data_args.add_dataset_name,
        num_examples=data_args.num_examples,
        input_record_file=data_args.input_record_file
    )
    training_args.remove_unused_columns = False

    # Metrics
    def compute_rouge_metrics(dataset, preds, save_prefix=None):
        decoded_preds = skip_instructions(model, preds, tokenizer)
        references = [e["Instance"]["label"] for e in dataset]
        result = compute_metrics(predictions=decoded_preds, references=references)
        result.update(compute_grouped_metrics(predictions=decoded_preds, references=references, groups=dataset["Task"]))
        result.update(compute_grouped_metrics(predictions=decoded_preds, references=references, groups=dataset["Dataset"]))
        prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
        result["gen_len"] = np.mean(prediction_lens)
        result = {k: round(v, 4) for k, v in result.items()}
        if save_prefix:
            with open(os.path.join(training_args.output_dir, f"{save_prefix}_eval_predictions.jsonl"), "w") as fout:
                for example, pred in zip(dataset, decoded_preds):
                    fout.write(json.dumps({
                        "Task": example["Task"],
                        "Dataset": example["Dataset"],
                        "Instance": example["Instance"],
                        "Prediction": pred
                    }) + "\n")
        return result

    logger.info(f"Gradient checkpointing? {training_args.gradient_checkpointing}")
    if training_args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    print(f"-----Gradient checkpointing: {training_args.gradient_checkpointing} -----")

    trainer = UIETrainer(
        model=model,
        args=training_args,
        path=model_args.model_name_or_path,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_rouge_metrics,
        callbacks=(
            [DenserEvalCallback]
            if training_args.denser_evaluation
            else None
        ),
        lambda1=training_args.lambda1,
        lambda2=training_args.lambda2,
        theta_norm_p=training_args.theta_norm_p,
        mlp_norm_p=training_args.mlp_norm_p,
    )

    all_metrics = {"run_name": training_args.run_name}

    if training_args.do_train:
        checkpoint = training_args.resume_from_checkpoint or last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        peft_model_id = os.path.join(training_args.output_dir, "adapter")
        num_materialized = materialize_wavelet_lora_factors(trainer.model)
        if num_materialized:
            logger.info(
                "Materialized %d LinearWaveletFilter modules before saving.",
                num_materialized,
            )
        trainer.model.save_pretrained(peft_model_id)
        tokenizer.save_pretrained(peft_model_id)
        metrics = train_result.metrics
        metrics["train_samples"] = min(data_args.max_train_samples or len(train_dataset), len(train_dataset))
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()
        all_metrics.update(metrics)

    results = {}
    
    if training_args.do_predict:
        logger.info("*** Prediction ***")
        predict_results = trainer.predict(
            predict_dataset,
            metric_key_prefix="predict",
            max_new_tokens=training_args.generation_max_length or data_args.max_target_length,
            num_beams=data_args.num_beams or training_args.generation_num_beams,
            repetition_penalty=data_args.repetition_penalty,
            pad_token_id=tokenizer.pad_token_id
        )
        metrics = predict_results.metrics
        metrics["predict_samples"] = min(data_args.max_predict_samples or len(predict_dataset), len(predict_dataset))
        trainer.log(metrics)
        trainer.log_metrics("predict", metrics)
        trainer.save_metrics("predict", metrics)
        all_metrics.update(metrics)

    return results


if __name__ == "__main__":
    main()
