from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from torch.utils.data import DataLoader

_track_root = Path(__file__).resolve().parent
_repo_root = _track_root.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from whisper_qa.src.config import load_config, resolve_repo_relative
from whisper_qa.src.data import (
    TeleAntiFraudManifestDataset,
    WhisperQACollator,
    create_processor,
    load_manifest_records,
)
from whisper_qa.src.model import WhisperQAModel
from whisper_qa.src.questions import QuestionBank
from whisper_qa.src.trainer import WhisperQATrainer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the QA-driven Whisper scam detector.")
    parser.add_argument(
        "--config",
        type=Path,
        default=_track_root / "config" / "whisper_qa_medium.yaml",
        help="Path to the whisper_qa YAML config.",
    )
    parser.add_argument(
        "--dataset_path",
        type=Path,
        default=None,
        help="Optional override for data.dataset_path.",
    )
    parser.add_argument("--train_split", type=str, default=None)
    parser.add_argument("--eval_split", type=str, default=None)
    parser.add_argument("--max_train_rows", type=int, default=None)
    parser.add_argument("--max_eval_rows", type=int, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.config)

    if args.dataset_path is not None:
        config.setdefault("data", {})["dataset_path"] = str(args.dataset_path)
    if args.train_split is not None:
        config.setdefault("data", {})["train_split"] = args.train_split
    if args.eval_split is not None:
        config.setdefault("data", {})["eval_split"] = args.eval_split

    data_cfg = config.get("data", {})
    dataset_path = resolve_repo_relative(data_cfg["dataset_path"], args.config)
    question_bank_path = resolve_repo_relative(config.get("questions", {}).get("bank_path"), args.config)
    output_dir = resolve_repo_relative(config.get("training", {}).get("output_dir", "outputs/whisper_qa"), args.config)

    processor = create_processor(config.get("model", {}))
    question_bank = QuestionBank.from_yaml(question_bank_path)

    train_records = load_manifest_records(
        dataset_path=dataset_path,
        split_name=str(data_cfg.get("train_split", "train")),
        label_column=str(data_cfg.get("label_column", "label")),
        text_column=str(data_cfg.get("text_column", "transcript")),
        max_rows=args.max_train_rows,
    )
    eval_records = load_manifest_records(
        dataset_path=dataset_path,
        split_name=str(data_cfg.get("eval_split", "val")),
        label_column=str(data_cfg.get("label_column", "label")),
        text_column=str(data_cfg.get("text_column", "transcript")),
        max_rows=args.max_eval_rows or int(data_cfg.get("max_eval_samples", 0)) or None,
    )

    collator = WhisperQACollator(processor=processor, config=config)
    train_loader = DataLoader(
        TeleAntiFraudManifestDataset(train_records),
        batch_size=int(config.get("training", {}).get("per_device_train_batch_size", 1)),
        shuffle=True,
        num_workers=int(data_cfg.get("num_workers", 0)),
        collate_fn=collator,
    )
    eval_loader = DataLoader(
        TeleAntiFraudManifestDataset(eval_records),
        batch_size=int(config.get("training", {}).get("per_device_eval_batch_size", 1)),
        shuffle=False,
        num_workers=int(data_cfg.get("num_workers", 0)),
        collate_fn=collator,
    )

    model = WhisperQAModel(processor=processor, config=config)
    trainer = WhisperQATrainer(
        model=model,
        processor=processor,
        question_bank=question_bank,
        train_dataloader=train_loader,
        eval_dataloader=eval_loader,
        config=config,
        output_dir=output_dir,
    )

    history = trainer.train()
    print(json.dumps(history, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
