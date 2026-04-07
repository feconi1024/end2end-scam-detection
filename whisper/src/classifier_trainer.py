from __future__ import annotations

from inspect import signature
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from transformers import Trainer, TrainingArguments

from .classifier_data import DataCollatorAudioClassificationWithPadding


def build_classification_compute_metrics_fn(id2label: Mapping[int, str]) -> Any:
    del id2label

    def compute_metrics(eval_pred: Any) -> Dict[str, float]:
        logits, labels = eval_pred
        if isinstance(logits, tuple):
            logits = logits[0]
        if isinstance(labels, tuple):
            labels = labels[0]

        predictions = np.argmax(logits, axis=-1)
        precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
            labels,
            predictions,
            average="macro",
            zero_division=0.0,
        )
        accuracy = accuracy_score(labels, predictions)
        f1_binary = f1_score(labels, predictions, average="binary", zero_division=0.0)

        return {
            "accuracy": float(accuracy),
            "precision_macro": float(precision_macro),
            "recall_macro": float(recall_macro),
            "f1_macro": float(f1_macro),
            "f1_binary": float(f1_binary),
        }

    return compute_metrics


def _compute_balanced_class_weights(train_dataset, num_labels: int) -> torch.Tensor:
    labels = np.asarray(train_dataset["class_label"], dtype=np.int64)
    counts = np.bincount(labels, minlength=num_labels).astype(np.float32)
    counts = np.maximum(counts, 1.0)
    weights = counts.sum() / (num_labels * counts)
    return torch.tensor(weights, dtype=torch.float32)


def create_classifier_training_arguments(
    training_cfg: Mapping[str, Any],
    output_dir: Path,
) -> TrainingArguments:
    use_fp16 = bool(training_cfg.get("fp16", False)) and torch.cuda.is_available()
    use_bf16 = bool(training_cfg.get("bf16", False)) and torch.cuda.is_available()

    base_kwargs: Dict[str, Any] = {
        "output_dir": str(output_dir),
        "per_device_train_batch_size": int(training_cfg.get("per_device_train_batch_size", 4)),
        "per_device_eval_batch_size": int(training_cfg.get("per_device_eval_batch_size", 4)),
        "gradient_accumulation_steps": int(training_cfg.get("gradient_accumulation_steps", 1)),
        "num_train_epochs": float(training_cfg.get("num_train_epochs", 3.0)),
        "learning_rate": float(training_cfg.get("learning_rate", 2e-5)),
        "weight_decay": float(training_cfg.get("weight_decay", 0.0)),
        "warmup_steps": int(training_cfg.get("warmup_steps", 0)),
        "logging_steps": int(training_cfg.get("logging_steps", 50)),
        "save_steps": int(training_cfg.get("save_steps", 1000)),
        "eval_steps": int(training_cfg.get("eval_steps", 1000)),
        "save_total_limit": int(training_cfg.get("save_total_limit", 2)),
        "remove_unused_columns": False,
        "fp16": use_fp16,
        "bf16": use_bf16,
    }

    eval_strategy_value = training_cfg.get("evaluation_strategy", "steps")
    sig = signature(TrainingArguments.__init__)
    valid_params = set(sig.parameters.keys())

    if "evaluation_strategy" in valid_params:
        base_kwargs["evaluation_strategy"] = eval_strategy_value
    elif "eval_strategy" in valid_params:
        base_kwargs["eval_strategy"] = eval_strategy_value

    if "report_to" in valid_params:
        base_kwargs["report_to"] = training_cfg.get("report_to", "none")
    if "save_safetensors" in valid_params:
        base_kwargs["save_safetensors"] = False
    if "load_best_model_at_end" in valid_params:
        base_kwargs["load_best_model_at_end"] = bool(
            training_cfg.get("load_best_model_at_end", True)
        )
    if "label_names" in valid_params:
        # Keep family_labels available to the model forward pass, but make
        # Trainer metrics/checkpoint selection operate on the main task label.
        base_kwargs["label_names"] = ["labels"]
    if "metric_for_best_model" in valid_params:
        base_kwargs["metric_for_best_model"] = str(
            training_cfg.get("metric_for_best_model", "f1_macro")
        )
    if "greater_is_better" in valid_params:
        base_kwargs["greater_is_better"] = bool(training_cfg.get("greater_is_better", True))

    filtered_kwargs = {k: v for k, v in base_kwargs.items() if k in valid_params}
    return TrainingArguments(**filtered_kwargs)


def create_classifier_trainer(
    model,
    processor,
    train_dataset,
    eval_dataset,
    training_cfg: Mapping[str, Any],
    output_dir: Path,
    compute_metrics: Optional[Callable[[Any], Dict[str, float]]] = None,
) -> Trainer:
    args = create_classifier_training_arguments(training_cfg=training_cfg, output_dir=output_dir)
    data_collator = DataCollatorAudioClassificationWithPadding(processor=processor)

    if train_dataset is not None and str(training_cfg.get("class_weighting", "balanced")).lower() == "balanced":
        class_weights = _compute_balanced_class_weights(train_dataset, model.num_labels)
        model.set_class_weights(class_weights)

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    return trainer
