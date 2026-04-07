from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s: %(message)s",
)

_whisper_root = Path(__file__).resolve().parent
if str(_whisper_root / "src") not in sys.path:
    sys.path.insert(0, str(_whisper_root / "src"))

from src.classifier_data import load_and_prepare_classification_datasets
from src.classifier_model import load_classifier_for_inference, resolve_classifier_model_dir
from src.classifier_trainer import (
    build_classification_compute_metrics_fn,
    create_classifier_trainer,
)
from src.data_processor import create_processor, load_config

logger = logging.getLogger(__name__)


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
        default="test",
        help="Evaluation split name. Use 'test' for final reporting; 'val' is mainly for model selection/debugging.",
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
    resolved_model_dir = resolve_classifier_model_dir(args.model_dir)

    if args.eval_split == "val":
        logger.warning(
            "You are evaluating on the validation split. Since training also selects the best checkpoint on val, "
            "this score is optimistic and should not be treated as the final reported result."
        )
    logger.info("Loading classifier checkpoint from %s", resolved_model_dir)

    if args.processor_dir is not None and args.processor_dir.exists():
        from transformers import WhisperProcessor

        processor = WhisperProcessor.from_pretrained(str(args.processor_dir))
    else:
        processor, _ = create_processor(config)

    _, eval_dataset, _, id2label, _, _ = load_and_prepare_classification_datasets(
        dataset_path=args.dataset_path,
        processor=processor,
        config=config,
        train_split=args.train_split,
        eval_split=args.eval_split,
        num_proc=args.num_proc,
    )

    logger.info("Evaluating %d sample(s) from split '%s'", len(eval_dataset), args.eval_split)

    model = load_classifier_for_inference(resolved_model_dir)
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
    metrics["resolved_model_dir"] = str(resolved_model_dir)
    metrics["eval_split"] = str(args.eval_split)
    metrics["eval_num_samples"] = int(len(eval_dataset))
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
