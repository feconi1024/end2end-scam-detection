from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

from transformers import Seq2SeqTrainer

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
from src.trainer import create_training_arguments
from src.whislu_model import load_whislu_for_inference


def _default_dataset_path() -> Path:
    corrected = Path("TeleAntiFraud-28k") / "corrected_manifests"
    if corrected.exists():
        return corrected
    return Path("TeleAntiFraud-28k")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a saved WhiSLU checkpoint on a manifest split."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=_whisper_root / "config" / "whislu_config.yaml",
        help="Path to the WhiSLU YAML config.",
    )
    parser.add_argument(
        "--dataset_path",
        type=Path,
        default=_default_dataset_path(),
        help="Path to a saved DatasetDict, a manifest folder, or a single CSV manifest.",
    )
    parser.add_argument(
        "--eval_split",
        type=str,
        default="val",
        help="Evaluation split name. For manifest folders this usually maps to val_manifest.csv.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used for balanced eval subsampling.",
    )
    parser.add_argument(
        "--model_dir",
        type=Path,
        default=None,
        help="Optional path to a saved fine-tuned model directory.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    config_path = args.config
    config = load_config(config_path)
    processor, _ = create_processor(config)
    training_cfg = config.get("training", {})

    _, eval_ds = load_and_prepare_datasets(
        dataset_path=args.dataset_path,
        processor=processor,
        config=config,
        train_split="train",
        eval_split=args.eval_split,
        num_proc=None,
    )

    max_eval_samples = int(training_cfg.get("max_eval_samples", 0))
    if max_eval_samples > 0 and len(eval_ds) > max_eval_samples:
        if "label" in eval_ds.column_names:
            labels = eval_ds["label"]
            scam_indices = [i for i, y in enumerate(labels) if y == "scam"]
            non_indices = [i for i, y in enumerate(labels) if y == "non_scam"]
            rng = random.Random(args.seed)
            rng.shuffle(scam_indices)
            rng.shuffle(non_indices)
            per_class = max_eval_samples // 2
            selected = scam_indices[:per_class] + non_indices[:per_class]
            if len(selected) < max_eval_samples:
                remaining = max_eval_samples - len(selected)
                extra = (
                    scam_indices[per_class : per_class + remaining]
                    + non_indices[per_class : per_class + remaining]
                )
                selected += extra[:remaining]
            eval_ds = eval_ds.select(selected)
        else:
            eval_ds = eval_ds.select(range(max_eval_samples))

    default_model_dir = Path(training_cfg.get("output_dir", "outputs/whislu")) / "model"
    model_dir = (args.model_dir or default_model_dir).resolve()
    if not model_dir.is_dir():
        raise FileNotFoundError(f"Trained model directory not found: {model_dir}")

    model = load_whislu_for_inference(
        model_dir=str(model_dir),
        processor=processor,
    )

    compute_metrics = build_compute_metrics_fn(
        processor=processor,
        special_tokens=config.get("special_tokens", []),
        label2token=config.get("label2token", {}),
    )

    output_dir = Path(training_cfg.get("output_dir", "outputs/whislu"))
    eval_args = create_training_arguments(training_cfg=training_cfg, output_dir=output_dir)

    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

    trainer = Seq2SeqTrainer(
        model=model,
        args=eval_args,
        train_dataset=None,
        eval_dataset=eval_ds,
        data_collator=data_collator,
        tokenizer=processor.tokenizer,
        compute_metrics=compute_metrics,
    )

    metrics = trainer.evaluate()
    print(metrics)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
