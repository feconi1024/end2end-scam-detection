from __future__ import annotations

import argparse
import sys
from pathlib import Path

from transformers import set_seed

_whisper_root = Path(__file__).resolve().parent
if str(_whisper_root / "src") not in sys.path:
    sys.path.insert(0, str(_whisper_root / "src"))

from src.data_processor import (
    DataCollatorSpeechSeq2SeqWithPadding,
    create_processor,
    load_and_prepare_datasets,
    load_config,
)
from src.evaluator import build_compute_metrics_fn
from src.trainer import create_trainer
from src.whislu_model import initialize_whislu_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune Whisper for WhiSLU scam detection with JSON-formatted multitask targets."
    )
    parser.add_argument(
        "--config",
        "-c",
        type=Path,
        default=_whisper_root / "config" / "whislu_config.yaml",
        help="Path to whislu_config.yaml",
    )
    parser.add_argument(
        "--dataset_path",
        "-d",
        type=Path,
        required=True,
        help="Path to TeleAntiFraud-28k dataset prepared as a Hugging Face DatasetDict (load_from_disk).",
    )
    parser.add_argument(
        "--train_split",
        type=str,
        default="train",
        help="Name of the training split in the DatasetDict.",
    )
    parser.add_argument(
        "--eval_split",
        type=str,
        default="validation",
        help="Name of the evaluation split in the DatasetDict.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--num_proc",
        type=int,
        default=None,
        help="Number of processes for dataset preprocessing.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    set_seed(args.seed)

    config = load_config(args.config)
    processor, _ = create_processor(config)

    training_cfg = config.get("training", {})
    model_name = config["model_name"]

    train_dataset, eval_dataset = load_and_prepare_datasets(
        dataset_path=args.dataset_path,
        processor=processor,
        config=config,
        train_split=args.train_split,
        eval_split=args.eval_split,
        num_proc=args.num_proc,
    )

    # Subsample eval set to avoid OOM during generation (eval is memory-heavy),
    # and stratify to get a roughly balanced scam/non_scam mix if labels exist.
    max_eval_samples = int(training_cfg.get("max_eval_samples", 0))
    if max_eval_samples > 0 and len(eval_dataset) > max_eval_samples:
        if "label" in eval_dataset.column_names:
            labels = eval_dataset["label"]
            scam_indices = [i for i, y in enumerate(labels) if y == "scam"]
            non_indices = [i for i, y in enumerate(labels) if y == "non_scam"]
            per_class = max_eval_samples // 2
            selected = scam_indices[:per_class] + non_indices[:per_class]
            # If we don't have enough of one class, fill from the other.
            if len(selected) < max_eval_samples:
                remaining = max_eval_samples - len(selected)
                extra = (
                    scam_indices[per_class : per_class + remaining]
                    + non_indices[per_class : per_class + remaining]
                )
                selected += extra[:remaining]
            if selected:
                eval_dataset = eval_dataset.select(selected)
            else:
                eval_dataset = eval_dataset.select(range(max_eval_samples))
        else:
            eval_dataset = eval_dataset.select(range(max_eval_samples))

    model = initialize_whislu_model(
        model_name=model_name,
        processor=processor,
        config=config,
    )

    compute_metrics = build_compute_metrics_fn(
        processor=processor,
        special_tokens=config.get("special_tokens", []),
        label2token=config.get("label2token", {}),
    )

    output_dir = Path(training_cfg.get("output_dir", "outputs/whislu"))
    output_dir.mkdir(parents=True, exist_ok=True)

    trainer = create_trainer(
        model=model,
        processor=processor,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        training_cfg=training_cfg,
        output_dir=output_dir,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    # Save final full model weights (and tokenizer) for later inference
    trainer.model.save_pretrained(str(output_dir / "model"))
    processor.save_pretrained(str(output_dir / "processor"))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

