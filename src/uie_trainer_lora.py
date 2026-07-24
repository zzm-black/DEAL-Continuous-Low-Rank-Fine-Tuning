import torch
from transformers import GenerationConfig
from transformers.trainer_seq2seq import Seq2SeqTrainer
from transformers.trainer import *
from transformers.trainer_callback import TrainerCallback

from uie_collator import SUPPORTED_DECODER_MODELS, check_model
from uie_dataset_lora import ANSWER_PREFIX
import torch.nn.functional as F
from accelerate.utils import pad_across_processes
from waveletLoRAAdapter import LinearWaveletFilter, MLP

def lp_power(parameters, p: float, device: torch.device) -> torch.Tensor:
    """Return sum_i |theta_i|^p over a parameter collection."""
    penalty = torch.zeros((), device=device, dtype=torch.float32)

    for parameter in parameters:
        if parameter.requires_grad:
            penalty = penalty + parameter.float().abs().pow(p).sum()

    return penalty

def skip_instructions(model, predictions_ids, tokenizer, ignore_idx=-100):
    predictions_ids = np.where(predictions_ids == ignore_idx, tokenizer.pad_token_id, predictions_ids)

    predictions = tokenizer.batch_decode(
        predictions_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
    )

    final_predictions = []
    if check_model(model.config._name_or_path, SUPPORTED_DECODER_MODELS):
        for pred in predictions:

            if ANSWER_PREFIX in pred:
                splits = pred.split(ANSWER_PREFIX)
                final_predictions.append(splits[-1].strip())
            else:
                final_predictions.append('')
    else:
        final_predictions = predictions

    return final_predictions


class DenserEvalCallback(TrainerCallback):

    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):

        log_eval_steps = [1, 50, 100, 200]

        # Log
        if args.logging_strategy == IntervalStrategy.STEPS and state.global_step in log_eval_steps:
            control.should_log = True

        # Evaluate
        if args.evaluation_strategy == IntervalStrategy.STEPS and state.global_step in log_eval_steps:
            control.should_evaluate = True

        # Save
        # if args.save_strategy

        return control

class UIETrainer(Seq2SeqTrainer):
    def __init__(
        self,
        *args,
        path: str,
        lambda1: float = 0.01,
        lambda2: float = 0.001,
        theta_norm_p: float = 5.0,
        mlp_norm_p: float = 2.0,
        debug: bool = False,
        **kwargs,
    ):
        kwargs["args"].disable_tqdm = True
        super().__init__(*args, **kwargs)
        self.path = path
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.theta_norm_p = theta_norm_p
        self.mlp_norm_p = mlp_norm_p
        self.debug = debug
        
    def training_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        *args,
        **kwargs,
    ) -> torch.Tensor:
        model.train()
    
        inputs = self._prepare_inputs(inputs)
    
        with self.compute_loss_context_manager():
            loss = self.compute_loss(model, inputs)
    
        reg_loss = torch.zeros(
            (),
            device=loss.device,
            dtype=loss.dtype,
        )
        
        retention_reg = torch.zeros(
            (),
            device=loss.device,
            dtype=torch.float32,
        )
        
        adaptation_reg = torch.zeros(
            (),
            device=loss.device,
            dtype=torch.float32,
        )
        
        num_wavelet_modules = 0
        
        for module in model.modules():
            if not isinstance(module, LinearWaveletFilter):
                continue
        
            num_wavelet_modules += 1
        
            retention_reg = retention_reg + lp_power(
                module.retention_parameters(),
                self.theta_norm_p,
                loss.device,
            )
        
            adaptation_reg = adaptation_reg + lp_power(
                module.adaptation_parameters(),
                self.mlp_norm_p,
                loss.device,
            )
        
        reg_loss = (
            self.lambda1 * retention_reg
            + self.lambda2 * adaptation_reg
        ).to(loss.dtype)
        
        total_loss = loss + reg_loss
        
        if self.args.gradient_accumulation_steps > 1:
            total_loss = total_loss / self.args.gradient_accumulation_steps
        
        if (
            self.args.logging_steps > 0
            and self.state.global_step % self.args.logging_steps == 0
        ):
            self.log({
                "base_loss": float(loss.detach().cpu()),
                "reg_loss": float(reg_loss.detach().cpu()),
                "retention_reg": float(retention_reg.detach().cpu()),
                "adaptation_reg": float(adaptation_reg.detach().cpu()),
                "wavelet_modules": num_wavelet_modules,
            })
        
        self.accelerator.backward(total_loss)
        
        return total_loss.detach()
    def evaluation_loop(
        self,
        dataloader: DataLoader,
        description: str,
        prediction_loss_only: Optional[bool] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> EvalLoopOutput:
        """
        Prediction/evaluation loop, shared by `Trainer.evaluate()` and `Trainer.predict()`.

        Works both with or without labels.
        """
        args = self.args

        prediction_loss_only = prediction_loss_only if prediction_loss_only is not None else args.prediction_loss_only

        # Simplify model setup (removed deepspeed logic)
        model = self._wrap_model(self.model, training=False)

        # If full fp16 or bf16 eval is needed, adjust the model dtype
        if not self.is_in_train:
            if args.fp16_full_eval:
                model = model.to(dtype=torch.float16, device=args.device)
            elif args.bf16_full_eval:
                model = model.to(dtype=torch.bfloat16, device=args.device)

        batch_size = dataloader.batch_size

        logger.info(f"***** Running {description} *****")
        if has_length(dataloader.dataset):
            logger.info(f"  Num examples = {self.num_examples(dataloader)}")
        else:
            logger.info("  Num examples: Unknown")
        logger.info(f"  Batch size = {batch_size}")

        model.eval()

        self.callback_handler.eval_dataloader = dataloader
        eval_dataset = dataloader.dataset

        if args.past_index >= 0:
            self._past = None

        # Initialize containers
        losses_host = None
        preds_host = None
        labels_host = None
        all_losses = None
        all_preds = None
        all_labels = None
        observed_num_examples = 0

        # Main evaluation loop
        for step, inputs in enumerate(dataloader):
            # Update the observed num examples
            observed_batch_size = find_batch_size(inputs)
            if observed_batch_size is not None:
                observed_num_examples += observed_batch_size
                if batch_size is None:
                    batch_size = observed_batch_size

            # Prediction step
            loss, logits, labels = self.prediction_step(model, inputs, prediction_loss_only, ignore_keys=ignore_keys)

            # Update containers
            if loss is not None:
                losses = self._nested_gather(loss.repeat(batch_size))
                losses_host = losses if losses_host is None else torch.cat((losses_host, losses), dim=0)
            if labels is not None:
                #labels = self._pad_across_processes(labels)
                labels = self._nested_gather(labels)
                labels_host = labels if labels_host is None else nested_concat(labels_host, labels, padding_index=-100)
            if logits is not None:
                #logits = self._pad_across_processes(logits)
                logits = self._nested_gather(logits)
                if self.preprocess_logits_for_metrics is not None:
                    logits = self.preprocess_logits_for_metrics(logits, labels)
                preds_host = logits if preds_host is None else nested_concat(preds_host, logits, padding_index=-100)
            self.control = self.callback_handler.on_prediction_step(args, self.state, self.control)

            # Gather all tensors and put them back on the CPU if we have done enough accumulation steps.
            if args.eval_accumulation_steps is not None and (step + 1) % args.eval_accumulation_steps == 0:
                if losses_host is not None:
                    losses = nested_numpify(losses_host)
                    all_losses = losses if all_losses is None else np.concatenate((all_losses, losses), axis=0)
                if preds_host is not None:
                    logits = nested_numpify(preds_host)
                    all_preds = logits if all_preds is None else nested_concat(all_preds, logits, padding_index=-100)
                if labels_host is not None:
                    labels = nested_numpify(labels_host)
                    all_labels = (
                        labels if all_labels is None else nested_concat(all_labels, labels, padding_index=-100)
                    )

                # Reset accumulation containers
                losses_host, preds_host, labels_host = None, None, None

        # Clean up state
        if args.past_index and hasattr(self, "_past"):
            delattr(self, "_past")

        # Gather all remaining tensors
        if losses_host is not None:
            losses = nested_numpify(losses_host)
            all_losses = losses if all_losses is None else np.concatenate((all_losses, losses), axis=0)
        if preds_host is not None:
            logits = nested_numpify(preds_host)
            all_preds = logits if all_preds is None else nested_concat(all_preds, logits, padding_index=-100)
        if labels_host is not None:
            labels = nested_numpify(labels_host)
            all_labels = labels if all_labels is None else nested_concat(all_labels, labels, padding_index=-100)

        # Number of samples
        if has_length(eval_dataset):
            num_samples = len(eval_dataset)
        elif isinstance(eval_dataset, IterableDatasetShard) and hasattr(eval_dataset, "num_examples"):
            num_samples = eval_dataset.num_examples
        else:
            num_samples = observed_num_examples

        # Truncate results to match the number of samples
        if all_losses is not None:
            all_losses = all_losses[:num_samples]
        if all_preds is not None:
            all_preds = all_preds[:num_samples]
        if all_labels is not None:
            all_labels = all_labels[:num_samples]

        # Compute metrics
        if self.compute_metrics is not None and all_preds is not None and all_labels is not None:
            metrics = self.compute_metrics(dataset=eval_dataset, preds=all_preds, save_prefix=metric_key_prefix)
        else:
            metrics = {}

        metrics["global_step"] = self.state.global_step

        # Remove numpy types and zero-d tensors for JSON serialization
        metrics = denumpify_detensorize(metrics)

        if all_losses is not None:
            metrics[f"{metric_key_prefix}_loss"] = all_losses.mean().item()

        # Prefix all keys with metric_key_prefix
        for key in list(metrics.keys()):
            if not key.startswith(f"{metric_key_prefix}_"):
                metrics[f"{metric_key_prefix}_{key}"] = metrics.pop(key)

        return EvalLoopOutput(predictions=all_preds, label_ids=all_labels, metrics=metrics, num_samples=num_samples)


    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Perform an evaluation step on `model` using `inputs`.

        Subclass and override to inject custom behavior.

        Args:
            model (`nn.Module`):
                The model to evaluate.
            inputs (`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.
            prediction_loss_only (`bool`):
                Whether or not to return the loss only.

        Return:
            Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]: A tuple with the loss, logits and
            labels (each being optional).
        """
        
        # If we are not predicting with generation, use the default prediction step
        if not self.args.predict_with_generate or prediction_loss_only:
            return super().prediction_step(
                model, inputs, prediction_loss_only=prediction_loss_only, ignore_keys=ignore_keys
            )

        has_labels = "labels" in inputs
        inputs = self._prepare_inputs(inputs)

        # Disable synced_gpus (no distributed or deepspeed)
        gen_kwargs = self._gen_kwargs

        # Add attention mask if it exists
        if "attention_mask" in inputs:
            gen_kwargs["attention_mask"] = inputs["attention_mask"]

        # Prepare generation config (model-specific)
        generation_config = GenerationConfig(**gen_kwargs)
        
        # Handle encoder-decoder models with varying input names
        if hasattr(self.model, "encoder") and self.model.encoder.main_input_name != self.model.main_input_name:
            generation_inputs = inputs[self.model.encoder.main_input_name]
        else:
            generation_inputs = inputs[self.model.main_input_name]

        # Generate tokens from model
        generated_tokens = self.model.generate(
            input_ids=generation_inputs, 
            generation_config=generation_config
        )

        # Determine max length for padding
        bs, source_len = inputs['input_ids'].shape
        if check_model(self.model.config._name_or_path, SUPPORTED_DECODER_MODELS):
            max_length = source_len + gen_kwargs["max_new_tokens"]
        else:
            max_length = gen_kwargs["max_new_tokens"]

        # Pad generated tokens if necessary
        if generated_tokens.shape[-1] < max_length:
            generated_tokens = self._pad_tensors_to_max_len(generated_tokens, max_length)

        # Compute loss if labels are available
        with torch.no_grad():
            if has_labels:
                with self.autocast_smart_context_manager():
                    outputs = model(**inputs)
                if self.label_smoother is not None:
                    loss = self.label_smoother(outputs, inputs["labels"]).mean().detach()
                else:
                    loss = (outputs["loss"] if isinstance(outputs, dict) else outputs[0]).mean().detach()
            else:
                loss = None

        if self.args.prediction_loss_only:
            return (loss, None, None)

        # Pad labels if necessary
        if has_labels:
            labels = inputs["labels"]
            if labels.shape[-1] < gen_kwargs["max_new_tokens"]:
                labels = self._pad_tensors_to_max_len(labels, gen_kwargs["max_new_tokens"])
        else:
            labels = None

        return (loss, generated_tokens, labels)
