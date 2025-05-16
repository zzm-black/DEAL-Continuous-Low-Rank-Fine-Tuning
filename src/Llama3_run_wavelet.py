#!/usr/bin/env python
# coding=utf-8

import logging
import os
import sys
import json
import time
from dataclasses import dataclass, field
from typing import Optional

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
    BitsAndBytesConfig,
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
from pytorch_wavelets import DWTForward, DWTInverse
import torch.nn.functional as F
import warnings
import torch.nn.init as init

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
    lora_dim: Optional[int] = field(default=32)


@dataclass
class DataTrainingArguments:
    lang: str = field(default=None)
    data_dir: str = field(default=None)
    task_config_dir: str = field(default=None)
    instruction_file: str = field(default=None)
    instruction_strategy: Optional[str] = field(default='single')
    overwrite_cache: bool = field(default=False)
    input_record_file: str = field(default='/home/kaili37/clkft/O-LoRA/logs_and_outputs_llama/order_1/logs/records.txt')
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
    theta_norm_p: float = field(default=10.0, metadata={"help": "Norm degree for theta (wavelet filter)"})
    mlp_norm_p: float = field(default=2.0, metadata={"help": "Norm degree for MLP parameters"})
    
    
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, init='ones'):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()
        
        if init == 'ones':
            nn.init.ones_(self.fc1.weight)
            nn.init.ones_(self.fc1.bias)
            nn.init.ones_(self.fc2.weight)
            nn.init.ones_(self.fc2.bias)
        elif init == 'kaiming':
            nn.init.kaiming_uniform_(self.fc1.weight, nonlinearity='relu')
            nn.init.zeros_(self.fc1.bias)
            nn.init.kaiming_uniform_(self.fc2.weight, nonlinearity='relu')
            nn.init.zeros_(self.fc2.bias)
        else:
            raise ValueError(f"Unknown init method: {init}")

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

class LinearWaveletFilter(nn.Module):
    def __init__(
        self, 
        original_weight, 
        original_bias=None, 
        wavelet='haar', 
        input_dim=8, 
        hidden_dim=64, 
        output_dim=8, 
        mlp_init='kaiming',
        device=None
    ):
        super(LinearWaveletFilter, self).__init__()

        self.register_buffer('original_weight', original_weight)
        if original_bias is not None:
            self.register_buffer('original_bias', original_bias)
        else:
            self.original_bias = None

        self.wavelet = wavelet
        self.weight = original_weight
        self.dwt = DWTForward(J=1, wave=self.wavelet, mode='zero').to(device)
        self.idwt = DWTInverse(wave=self.wavelet, mode='zero').to(device)
        weight_4d = self.original_weight.unsqueeze(0).unsqueeze(0).to(device)
        cA, _ = self.dwt(weight_4d)
        cA_shape = cA.shape
        self.theta = nn.Parameter(torch.ones(cA_shape, device=device))
        self.mlp = MLP(
            input_dim=input_dim, 
            hidden_dim=hidden_dim, 
            output_dim=output_dim, 
            init=mlp_init
        ).to(device)

        self.device = device

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input: shape [batch_size, seq_len, input_dim]
        Return:
            shape [batch_size, seq_len, output_dim]
        """
        
        input = input.to(self.device)
        weight_4d = self.original_weight.unsqueeze(0).unsqueeze(0).to(self.device)
        cA, cD = self.dwt(weight_4d)
        cA_modified = cA * self.theta
        filtered_weight_4d = self.idwt((cA_modified, cD))
        filtered_weight = filtered_weight_4d.squeeze(0).squeeze(0)
        self.weight = nn.Parameter(filtered_weight)
        linear_output = F.linear(input, filtered_weight, self.original_bias)
        # shape: [batch_size, seq_len, output_dim(这里是 out_features)]
        batch_size, seq_len, dim = linear_output.shape
        mlp_in = linear_output.view(-1, dim).to(self.device)
        mlp_out = self.mlp(mlp_in)  # [batch_size * seq_len, output_dim]

        mlp_out = mlp_out.view(batch_size, seq_len, -1)
        return mlp_out

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

    # logging setup
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

    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu} "
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
        num_examples=data_args.num_examples,
    )
    raw_datasets.cleanup_cache_files()
    
    # Load model & tokenizer
    if 'adapter' in model_args.model_name_or_path.lower():
        logger.info("Detected 'adapter' => LoRA adapter scenario.")
        config = PeftConfig.from_pretrained(model_args.model_name_or_path)
        tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token

    elif 'llama' in model_args.model_name_or_path.lower():
        logger.info("Detected 'llama' => base Llama scenario.")
        config = AutoConfig.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None
        )
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=model_args.cache_dir,
            use_fast=model_args.use_fast_tokenizer,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
        )
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        config.pad_token_id = tokenizer.pad_token_id
        config.eos_token_id = tokenizer.eos_token_id
        config.bos_token_id = tokenizer.bos_token_id

    # if 'llama' in model_args.model_name_or_path.lower():
    #     model_class = LlamaForCausalLM_with_lossmask
    #     tokenizer.padding_side = 'left'
    
    tokenizer.padding_side = 'left'

    if 'adapter' in model_args.model_name_or_path.lower():
        base_model = config.base_model_name_or_path
        model = AutoModelForCausalLM.from_pretrained(base_model)
        #model = model_class.from_pretrained(base_model)
        model = PeftModel.from_pretrained(model, model_args.model_name_or_path)

    elif 'llama' in model_args.model_name_or_path.lower():
        model = AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
        )
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=model_args.lora_dim,
            lora_alpha=32,
            lora_dropout=0.1,
            target_modules=["q_proj", "v_proj"]
        )
        model = get_peft_model(model, peft_config)

    model.generation_config.pad_token_id = tokenizer.pad_token_id
    model.generation_config.eos_token_id = tokenizer.eos_token_id
    model.config.pad_token_id            = tokenizer.pad_token_id
    model.config.eos_token_id            = tokenizer.eos_token_id

    if 'adapter' in model_args.model_name_or_path.lower():
        logger.info("** wavelet-based wrapping: freeze lora_B, unfreeze lora_A. **")
        for name, param in model.named_parameters():
            if 'lora_A' in name:
                param.requires_grad = True
            elif 'lora_B' in name:
                param.requires_grad = False

        lora_A_dict, lora_B_dict = extract_lora_params(model)
        device = str(training_args.device)

        lora_A_prime = {}
        for full_name, lora_A_module in lora_A_dict.items():
            # LoRA-A shape ~ [r, hidden_dim]
            original_weight = lora_A_module.weight
            original_bias = getattr(lora_A_module, 'bias', None)

            # input_dim = hidden_dim，output_dim = r
            in_dim = original_weight.shape[1]
            out_dim = original_weight.shape[0]

            new_lora_A = LinearWaveletFilter(
                original_weight=original_weight,
                original_bias=original_bias,
                wavelet='haar',
                input_dim=out_dim,
                hidden_dim=max(128, out_dim // 2),  
                output_dim=out_dim,
                mlp_init='kaiming',
                device=training_args.device
            )
            lora_A_prime[full_name] = new_lora_A

        lora_B_prime = {}
        for full_name, lora_B_module in lora_B_dict.items():
            lora_B_prime[full_name] = lora_B_module

        model = replace_lora_params(model, lora_A_prime, lora_B_prime)
        
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

    
    # prepare dataset splits
    if training_args.do_train:
        if "train" not in raw_datasets:
            raise ValueError("--do_train requires a 'train' dataset.")
        train_dataset = raw_datasets["train"]
        if data_args.max_train_samples is not None:
            train_dataset = train_dataset.select(range(data_args.max_train_samples))

    if training_args.do_eval:
        if "validation" not in raw_datasets:
            raise ValueError("--do_eval requires a 'validation' dataset.")
        eval_dataset = raw_datasets["validation"]
        if data_args.max_eval_samples is not None:
            eval_dataset = eval_dataset.select(range(data_args.max_eval_samples))

    if training_args.do_predict:
        if "test" not in raw_datasets:
            raise ValueError("--do_predict requires a 'test' dataset.")
        predict_dataset = raw_datasets["test"]
        if data_args.max_predict_samples is not None:
            predict_dataset = predict_dataset.select(range(data_args.max_predict_samples))

    # data_collator
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
        input_record_file=data_args.input_record_file,
    )
    training_args.remove_unused_columns = False

    # metrics
    def compute_rouge_metrics(dataset, preds, save_prefix=None):
        decoded_preds = skip_instructions(model, preds, tokenizer)
        references = [ex["Instance"]["label"] for ex in dataset]

        # overall
        result = compute_metrics(decoded_preds, references)
        # by task
        result_per_task = compute_grouped_metrics(decoded_preds, references, dataset["Task"])
        result.update(result_per_task)
        # by dataset
        cats = dataset["Dataset"]
        result_per_cat = compute_grouped_metrics(decoded_preds, references, cats)
        result.update(result_per_cat)

        prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
        result["gen_len"] = np.mean(prediction_lens)
        result = {k: round(v, 4) for k, v in result.items()}

        if save_prefix is not None:
            save_path = os.path.join(training_args.output_dir, f"{save_prefix}_eval_predictions.jsonl")
            with open(save_path, "w", encoding="utf-8") as fout:
                for example, decoded_pred in zip(dataset, decoded_preds):
                    fout.write(json.dumps({
                        "Task": example["Task"],
                        "Dataset": example["Dataset"],
                        "Instance": example["Instance"],
                        "Prediction": decoded_pred
                    }, ensure_ascii=False) + "\n")
        return result

    logger.info(f"Gradient checkpointing? {training_args.gradient_checkpointing}")
    if training_args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    # custom trainer
    trainer = UIETrainer(
        model=model,
        args=training_args,
        path=model_args.model_name_or_path,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_rouge_metrics,
        callbacks=[DenserEvalCallback] if training_args.denser_evaluation else None
    )

    all_metrics = {"run_name": training_args.run_name}

    # ---- training ----
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint

        train_result = trainer.train(resume_from_checkpoint=checkpoint)

        # save adapter
        peft_model_id = os.path.join(training_args.output_dir, "adapter")
        trainer.model.save_pretrained(peft_model_id)
        tokenizer.save_pretrained(peft_model_id)

        metrics = train_result.metrics
        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

        logger.info(f"Training metrics: {metrics}")
        all_metrics.update(metrics)

    # ---- evaluation / prediction ----
    results = {}
    max_new_tokens = (
        training_args.generation_max_length
        if training_args.generation_max_length is not None
        else data_args.max_target_length
    )
    num_beams = data_args.num_beams or training_args.generation_num_beams
    repetition_penalty = data_args.repetition_penalty

    if training_args.do_predict:
        logger.info("*** do_predict ***")
        if data_args.max_predict_samples is not None:
            predict_dataset = predict_dataset.select(range(data_args.max_predict_samples))
        
        predict_results = trainer.predict(
            predict_dataset,
            metric_key_prefix="predict",
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            repetition_penalty=repetition_penalty,
            pad_token_id=tokenizer.pad_token_id,
        )
        metrics = predict_results.metrics
        max_predict_samples = (
            data_args.max_predict_samples if data_args.max_predict_samples is not None else len(predict_dataset)
        )
        metrics["predict_samples"] = min(max_predict_samples, len(predict_dataset))

        trainer.log_metrics("predict", metrics)
        trainer.save_metrics("predict", metrics)
        all_metrics.update(metrics)

    return results


if __name__ == "__main__":
    main()


