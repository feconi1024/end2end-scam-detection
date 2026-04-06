from __future__ import annotations

import argparse
import logging
import random
import sys
from pathlib import Path

from transformers import set_seed

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s: %(message)s",
)

_whisper_root = Path(__file__).resolve().parent
if str(_whisper_root / "src") not in sys.path:
    sys.path.insert(0, str(_whisper_root / "src"))

from src.classifier_data import load_and_prepare_classification_datasets
from src.classifier_model import initialize_classifier_model
from src.classifier_trainer import (
    build_classification_compute_metrics_fn,
    create_classifier_trainer,
)
from src.data_processor import create_processor, load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a Whisper encoder classifier for scam detection."
    )
    parser.add_argument(
        "--config",
        "-c",
        type=Path,
        default=_whisper_root / "config" / "whisper_classifier_config.yaml",
        help="Path to whisper_classifier_config.yaml",
    )
    parser.add_argument(
        "--dataset_path",
        "-d",
        type=Path,
        required=True,
        help="Path to a manifest folder, DatasetDict, or CSV manifest.",
    )
    parser.add_argument(
        "--train_split",
        type=str,
        default="train",
        help="Training split name.",
    )
    parser.add_argument(
        "--eval_split",
        type=str,
        default="val",
        help="Evaluation split name.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )
    parser.add_argument(
        "--num_proc",
        type=int,
        default=None,
        help="Number of dataset preprocessing workers.",
    )
    return parser.parse_args()


def _subsample_eval_dataset(eval_dataset, max_eval_samples: int, seed: int):
    if max_eval_samples <= 0 or len(eval_dataset) <= max_eval_samples:
        return eval_dataset

    labels = eval_dataset["class_label"]
    grouped_indices = {}
    for idx, label in enumerate(labels):
        grouped_indices.setdefault(int(label), []).append(idx)

    rng = random.Random(seed)
    for indices in grouped_indices.values():
        rng.shuffle(indices)

    num_classes = max(1, len(grouped_indices))
    per_class = max(1, max_eval_samples // num_classes)

    selected = []
    for indices in grouped_indices.values():
        selected.extend(indices[:per_class])

    if len(selected) < max_eval_samples:
        remaining_pool = []
        for indices in grouped_indices.values():
            remaining_pool.extend(indices[per_class:])
        rng.shuffle(remaining_pool)
        selected.extend(remaining_pool[: max_eval_samples - len(selected)])

    selected = sorted(set(selected))[:max_eval_samples]
    return eval_dataset.select(selected)


def main() -> int:
    args = parse_args()
    set_seed(args.seed)

    config = load_config(args.config)
    processor, _ = create_processor(config)

    train_dataset, eval_dataset, label2id, id2label = load_and_prepare_classification_datasets(
        dataset_path=args.dataset_path,
        processor=processor,
        config=config,
        train_split=args.train_split,
        eval_split=args.eval_split,
        num_proc=args.num_proc,
    )

    training_cfg = config.get("training", {})
    max_eval_samples = int(training_cfg.get("max_eval_samples", 0))
    eval_dataset = _subsample_eval_dataset(eval_dataset, max_eval_samples, args.seed)

    model = initialize_classifier_model(
        model_name=config["model_name"],
        label2id=label2id,
        id2label=id2label,
        config=config,
    )

    output_dir = Path(training_cfg.get("output_dir", "outputs/whisper_classifier"))
    output_dir.mkdir(parents=True, exist_ok=True)

    trainer = create_classifier_trainer(
        model=model,
        processor=processor,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        training_cfg=training_cfg,
        output_dir=output_dir,
        compute_metrics=build_classification_compute_metrics_fn(id2label),
    )

    trainer.train()
    trainer.model.save_pretrained(str(output_dir / "model"))
    processor.save_pretrained(str(output_dir / "processor"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
