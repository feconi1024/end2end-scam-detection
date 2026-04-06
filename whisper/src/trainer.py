from __future__ import annotations

import contextlib
from inspect import signature
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional

import torch
import torch.nn.functional as F
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments, WhisperProcessor
from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled

try:
    from torch.distributed.fsdp import FullyShardedDataParallel
except Exception:  # pragma: no cover - optional runtime dependency
    FullyShardedDataParallel = tuple()  # type: ignore[assignment]

from .data_processor import DataCollatorSpeechSeq2SeqWithPadding


class WeightedLossSeq2SeqTrainer(Seq2SeqTrainer):
    """
    Seq2SeqTrainer variant that supports per-token loss weights.

    This is important for the current WhiSLU JSON target because a few intent
    tokens should not be numerically dominated by dozens of transcript/schema
    tokens in the standard token-level cross-entropy.
    """

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.get("labels")
        loss_weights = inputs.pop("loss_weights", None)

        outputs = model(**inputs)
        logits = outputs.get("logits") if isinstance(outputs, dict) else outputs.logits

        if labels is None:
            loss = outputs.get("loss") if isinstance(outputs, dict) else outputs.loss
            return (loss, outputs) if return_outputs else loss

        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        token_losses = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            reduction="none",
            ignore_index=-100,
        ).view_as(shift_labels)

        valid_mask = shift_labels.ne(-100)
        if loss_weights is not None:
            shift_weights = loss_weights[..., 1:].contiguous().to(token_losses.dtype)
            weighted_mask = shift_weights * valid_mask.to(token_losses.dtype)
            normalizer = weighted_mask.sum().clamp_min(1.0)
            loss = (token_losses * shift_weights).sum() / normalizer
        else:
            normalizer = valid_mask.sum().clamp_min(1)
            loss = (token_losses * valid_mask.to(token_losses.dtype)).sum() / normalizer

        return (loss, outputs) if return_outputs else loss

    def prediction_step(
        self,
        model,
        inputs,
        prediction_loss_only,
        ignore_keys=None,
        **gen_kwargs,
    ):
        if not self.args.predict_with_generate or prediction_loss_only:
            return super().prediction_step(
                model,
                inputs,
                prediction_loss_only=prediction_loss_only,
                ignore_keys=ignore_keys,
            )

        has_labels = "labels" in inputs
        inputs = self._prepare_inputs(inputs)
        forward_inputs = {k: v for k, v in inputs.items() if k != "loss_weights"}

        if len(gen_kwargs) == 0 and hasattr(self, "_gen_kwargs"):
            gen_kwargs = self._gen_kwargs.copy()
        if "num_beams" in gen_kwargs and gen_kwargs["num_beams"] is None:
            gen_kwargs.pop("num_beams")
        if "max_length" in gen_kwargs and gen_kwargs["max_length"] is None:
            gen_kwargs.pop("max_length")

        default_synced_gpus = is_deepspeed_zero3_enabled()
        gen_kwargs["synced_gpus"] = gen_kwargs.get("synced_gpus", default_synced_gpus)

        generation_inputs = forward_inputs.copy()
        if (
            "labels" in generation_inputs
            and "decoder_input_ids" in generation_inputs
            and generation_inputs["labels"].shape == generation_inputs["decoder_input_ids"].shape
        ):
            generation_inputs = {
                k: v
                for k, v in generation_inputs.items()
                if k not in ("decoder_input_ids", "decoder_attention_mask")
            }

        summon_full_params_context = (
            FullyShardedDataParallel.summon_full_params(self.model)
            if isinstance(self.model, FullyShardedDataParallel)
            else contextlib.nullcontext()
        )

        with summon_full_params_context:
            generated_tokens = self.model.generate(**generation_inputs, **gen_kwargs)

        if self.model.generation_config._from_model_config:
            self.model.generation_config._from_model_config = False

        gen_config = self.model.generation_config
        if generated_tokens.shape[-1] < gen_config.max_length:
            generated_tokens = self._pad_tensors_to_max_len(generated_tokens, gen_config.max_length)
        elif gen_config.max_new_tokens is not None and generated_tokens.shape[-1] < gen_config.max_new_tokens + 1:
            generated_tokens = self._pad_tensors_to_max_len(generated_tokens, gen_config.max_new_tokens + 1)

        with torch.no_grad():
            if has_labels:
                with self.compute_loss_context_manager():
                    outputs = model(**forward_inputs)
                loss = (outputs["loss"] if isinstance(outputs, dict) else outputs[0]).detach().mean()
            else:
                loss = None

        if self.args.prediction_loss_only:
            return loss, None, None

        if has_labels:
            labels = inputs["labels"]
            if labels.shape[-1] < gen_config.max_length:
                labels = self._pad_tensors_to_max_len(labels, gen_config.max_length)
            elif gen_config.max_new_tokens is not None and labels.shape[-1] < gen_config.max_new_tokens + 1:
                labels = self._pad_tensors_to_max_len(labels, gen_config.max_new_tokens + 1)
        else:
            labels = None

        return loss, generated_tokens, labels


def create_training_arguments(
    training_cfg: Mapping[str, Any],
    output_dir: Path,
) -> Seq2SeqTrainingArguments:
    """
    Build Seq2SeqTrainingArguments in a way that is robust to
    different transformers versions by only passing supported
    keyword arguments.
    """
    use_fp16 = False
    if training_cfg.get("fp16", False) and torch.cuda.is_available():
        use_fp16 = True

    use_bf16 = bool(training_cfg.get("bf16", False)) and torch.cuda.is_available()

    base_kwargs: Dict[str, Any] = {
        "output_dir": str(output_dir),
        "per_device_train_batch_size": int(training_cfg.get("per_device_train_batch_size", 4)),
        "per_device_eval_batch_size": int(training_cfg.get("per_device_eval_batch_size", 4)),
        "gradient_accumulation_steps": int(training_cfg.get("gradient_accumulation_steps", 1)),
        "num_train_epochs": float(training_cfg.get("num_train_epochs", 3.0)),
        "learning_rate": float(training_cfg.get("learning_rate", 1e-4)),
        "weight_decay": float(training_cfg.get("weight_decay", 0.0)),
        "warmup_steps": int(training_cfg.get("warmup_steps", 0)),
        "logging_steps": int(training_cfg.get("logging_steps", 50)),
        "save_steps": int(training_cfg.get("save_steps", 1000)),
        "eval_steps": int(training_cfg.get("eval_steps", 1000)),
        "save_total_limit": int(training_cfg.get("save_total_limit", 2)),
        "predict_with_generate": bool(training_cfg.get("predict_with_generate", True)),
        "fp16": use_fp16,
        "bf16": use_bf16,
        "generation_max_length": int(training_cfg.get("generation_max_length", 128)),
    }
    max_eval = int(training_cfg.get("max_eval_samples", 0))
    if max_eval > 0:
        base_kwargs["max_eval_samples"] = max_eval

    eval_strategy_value = training_cfg.get("evaluation_strategy", "steps")
    sig = signature(Seq2SeqTrainingArguments.__init__)
    valid_params = set(sig.parameters.keys())

    if "evaluation_strategy" in valid_params:
        base_kwargs["evaluation_strategy"] = eval_strategy_value
    elif "eval_strategy" in valid_params:
        base_kwargs["eval_strategy"] = eval_strategy_value

    if "report_to" in valid_params:
        base_kwargs["report_to"] = training_cfg.get("report_to", "none")

    if "save_safetensors" in valid_params:
        base_kwargs["save_safetensors"] = False

    filtered_kwargs = {k: v for k, v in base_kwargs.items() if k in valid_params}
    return Seq2SeqTrainingArguments(**filtered_kwargs)


def create_trainer(
    model,
    processor: WhisperProcessor,
    train_dataset,
    eval_dataset,
    training_cfg: Mapping[str, Any],
    output_dir: Path,
    compute_metrics: Optional[Callable[[Any], Dict[str, float]]] = None,
) -> Seq2SeqTrainer:
    args = create_training_arguments(training_cfg=training_cfg, output_dir=output_dir)

    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

    trainer = WeightedLossSeq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        tokenizer=processor.tokenizer,
        compute_metrics=compute_metrics,
    )

    return trainer
