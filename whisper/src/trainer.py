from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional

import numpy as np
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
    kwargs: Dict[str, Any] = {
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
        "evaluation_strategy": training_cfg.get("evaluation_strategy", "steps"),
        "save_total_limit": int(training_cfg.get("save_total_limit", 2)),
        "predict_with_generate": bool(training_cfg.get("predict_with_generate", True)),
        "fp16": bool(training_cfg.get("fp16", True)),
        "generation_max_length": int(training_cfg.get("generation_max_length", 128)),
    }

    return Seq2SeqTrainingArguments(**kwargs)


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

