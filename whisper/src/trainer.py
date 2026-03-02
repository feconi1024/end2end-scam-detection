from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional

from inspect import signature

from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    WhisperProcessor,
)

from .data_processor import DataCollatorSpeechSeq2SeqWithPadding


def create_training_arguments(
    training_cfg: Mapping[str, Any],
    output_dir: Path,
) -> Seq2SeqTrainingArguments:
    """
    Build Seq2SeqTrainingArguments in a way that is robust to
    different transformers versions by only passing supported
    keyword arguments.
    """
    base_kwargs: Dict[str, Any] = {
        "output_dir": str(output_dir),
        "per_device_train_batch_size": int(training_cfg.get("per_device_train_batch_size", 4)),
        "per_device_eval_batch_size": int(training_cfg.get("per_device_eval_batch_size", 4)),
        "gradient_accumulation_steps": int(training_cfg.get("gradient_accumulation_steps", 1)),
        "num_train_epochs": float(training_cfg.get("num_train_epochs", 3.0)),
        "learning_rate": float(training_cfg.get("learning_rate", 1e-4)),
        "warmup_steps": int(training_cfg.get("warmup_steps", 0)),
        "logging_steps": int(training_cfg.get("logging_steps", 50)),
        "save_steps": int(training_cfg.get("save_steps", 1000)),
        "eval_steps": int(training_cfg.get("eval_steps", 1000)),
        "save_total_limit": int(training_cfg.get("save_total_limit", 2)),
        "predict_with_generate": bool(training_cfg.get("predict_with_generate", True)),
        "fp16": bool(training_cfg.get("fp16", True)),
        "generation_max_length": int(training_cfg.get("generation_max_length", 128)),
    }

    # Handle evaluation strategy naming differences across versions
    eval_strategy_value = training_cfg.get("evaluation_strategy", "steps")
    sig = signature(Seq2SeqTrainingArguments.__init__)
    valid_params = set(sig.parameters.keys())

    if "evaluation_strategy" in valid_params:
        base_kwargs["evaluation_strategy"] = eval_strategy_value
    elif "eval_strategy" in valid_params:
        base_kwargs["eval_strategy"] = eval_strategy_value

    # Disable external reporting integrations (e.g., wandb) by default to avoid
    # interactive login / network issues on clusters.
    if "report_to" in valid_params:
        # Some versions expect a list, others accept "none" as special value.
        base_kwargs["report_to"] = training_cfg.get("report_to", "none")

    # Filter kwargs to only those accepted by the installed transformers version
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

    trainer = Seq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        tokenizer=processor.tokenizer,
        compute_metrics=compute_metrics,
    )

    return trainer

