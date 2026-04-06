from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_whisper_root = Path(__file__).resolve().parent
if str(_whisper_root / "src") not in sys.path:
    sys.path.insert(0, str(_whisper_root / "src"))

from src.classifier_data import load_and_prepare_classification_datasets
from src.classifier_model import load_classifier_for_inference
from src.classifier_trainer import (
    build_classification_compute_metrics_fn,
    create_classifier_trainer,
)
from src.data_processor import create_processor, load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a Whisper encoder classifier checkpoint."
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
        "--model_dir",
        "-m",
        type=Path,
        required=True,
        help="Path to a saved classifier model directory.",
    )
    parser.add_argument(
        "--processor_dir",
        type=Path,
        default=None,
        help="Optional saved processor directory. Falls back to base processor config.",
    )
    parser.add_argument(
        "--train_split",
        type=str,
        default="train",
        help="Training split name used to load the manifest family.",
    )
    parser.add_argument(
        "--eval_split",
        type=str,
        default="val",
        help="Evaluation split name.",
    )
    parser.add_argument(
        "--num_proc",
        type=int,
        default=None,
        help="Number of dataset preprocessing workers.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.config)

    if args.processor_dir is not None and args.processor_dir.exists():
        from transformers import WhisperProcessor

        processor = WhisperProcessor.from_pretrained(str(args.processor_dir))
    else:
        processor, _ = create_processor(config)

    _, eval_dataset, _, id2label = load_and_prepare_classification_datasets(
        dataset_path=args.dataset_path,
        processor=processor,
        config=config,
        train_split=args.train_split,
        eval_split=args.eval_split,
        num_proc=args.num_proc,
    )

    model = load_classifier_for_inference(args.model_dir)
    training_cfg = config.get("training", {})
    trainer = create_classifier_trainer(
        model=model,
        processor=processor,
        train_dataset=None,
        eval_dataset=eval_dataset,
        training_cfg=training_cfg,
        output_dir=args.model_dir.parent / "eval_tmp",
        compute_metrics=build_classification_compute_metrics_fn(id2label),
    )

    metrics = trainer.evaluate()
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
