from __future__ import annotations

import argparse
import platform
from pathlib import Path
from typing import Dict, Any

import torch
from torch.utils.data import DataLoader
from transformers import WhisperProcessor, BertTokenizer

from e2e_cascading.src.dataset import (
    TeleAntiFraudDataset,
    load_config,
    create_collate_fn,
)
from e2e_cascading.src.model import DifferentiableCascadeModel
from e2e_cascading.src.loss import JointCTCSLULoss
from e2e_cascading.src.trainer import Trainer, TrainerConfig


def build_label_mapping(cfg: Dict[str, Any]) -> Dict[str, int]:
    scam_label = cfg["dataset"]["scam_label"]
    non_scam_label = cfg["dataset"]["non_scam_label"]
    # Non-scam = 0, scam = 1
    return {non_scam_label: 0, scam_label: 1}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train/evaluate differentiable cascaded E2E scam detector.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("e2e_cascading/config/default_config.yaml"),
        help="Path to YAML config.",
    )
    parser.add_argument(
        "--eval_only",
        action="store_true",
        help="Run evaluation on the test set only using a provided checkpoint.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Path to model checkpoint (.pt) for evaluation or fine-tuning.",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    config_path = Path(args.config).resolve()
    # Resolve relative manifest paths from repo root (parent of e2e_cascading) so
    # training works from any cwd (e.g. Slurm job directory).
    config_dir = config_path.parent
    repo_root = config_dir.parent.parent
    def _resolve_manifest(p: str) -> str:
        path = Path(p)
        return str(repo_root / p) if not path.is_absolute() else p

    # Build label mapping consistent across dataset and classifier
    label_mapping = build_label_mapping(cfg)

    # Feature processor and tokenizer
    whisper_name = cfg["model"]["whisper_model_name"]
    bert_name = cfg["model"]["bert_model_name"]

    whisper_processor = WhisperProcessor.from_pretrained(whisper_name)
    tokenizer = BertTokenizer.from_pretrained(bert_name)

    # Datasets
    sr = int(cfg["dataset"]["sample_rate"])
    train_manifest = _resolve_manifest(cfg["dataset"].get("train_manifest", ""))
    val_manifest = _resolve_manifest(cfg["dataset"].get("val_manifest", ""))
    test_manifest = _resolve_manifest(cfg["dataset"].get("test_manifest", ""))

    train_ds = TeleAntiFraudDataset(
        manifest_path=train_manifest,
        tokenizer=tokenizer,
        sample_rate=sr,
        split="train",
        label_mapping=label_mapping,
    )
    val_ds = TeleAntiFraudDataset(
        manifest_path=val_manifest,
        tokenizer=tokenizer,
        sample_rate=sr,
        split="val",
        label_mapping=label_mapping,
    )
    test_ds = TeleAntiFraudDataset(
        manifest_path=test_manifest,
        tokenizer=tokenizer,
        sample_rate=sr,
        split="test",
        label_mapping=label_mapping,
    )

    # Collate function
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else cfg["model"]["ctc_blank_token_id"]
    collate_fn = create_collate_fn(
        processor=whisper_processor,
        pad_token_id=pad_token_id,
        sample_rate=sr,
    )

    # DataLoaders: use 0 workers on Windows (spawn can't pickle closures;
    # WhisperCollator is picklable, but Windows multiprocessing is still fragile).
    # On Linux/Slurm use config value for faster loading.
    _requested_workers = int(cfg["training"].get("num_workers", 4))
    num_workers = 0 if platform.system() == "Windows" else _requested_workers
    batch_size = int(cfg["training"]["batch_size"])

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )

    # Model
    num_labels = int(cfg["model"]["num_labels"])
    ctc_vocab_size = tokenizer.vocab_size
    ctc_blank_id = int(cfg["model"]["ctc_blank_token_id"])

    # Build id2label / label2id for classifier
    id2label = {v: k for k, v in label_mapping.items()}

    projector_cfg_overrides = cfg["model"].get("projector", {})

    model = DifferentiableCascadeModel(
        whisper_model_name=whisper_name,
        bert_model_name=bert_name,
        num_labels=num_labels,
        ctc_vocab_size=ctc_vocab_size,
        ctc_blank_id=ctc_blank_id,
        projector_cfg_overrides=projector_cfg_overrides,
        label2id=label_mapping,
        id2label=id2label,
    )

    if args.checkpoint is not None and args.checkpoint.is_file():
        state = torch.load(args.checkpoint, map_location="cpu")
        model.load_state_dict(state)
        print(f"Loaded checkpoint from {args.checkpoint}")

    # Loss and trainer
    loss_fn = JointCTCSLULoss(
        ctc_blank_id=ctc_blank_id,
        ctc_weight=cfg["training"]["ctc_weight"],
        slu_weight=cfg["training"]["slu_weight"],
    )

    output_dir = Path(cfg["training"]["output_dir"])

    # Device: respect CUDA_VISIBLE_DEVICES on Linux/Slurm; fallback to CPU if no GPU.
    device_cfg = str(cfg["training"].get("device", "cuda")).lower()
    if device_cfg == "cuda" and not torch.cuda.is_available():
        device_cfg = "cpu"
        print("CUDA not available; using device='cpu'.")

    trainer_cfg = TrainerConfig(
        num_epochs=int(cfg["training"]["num_epochs"]),
        learning_rate=float(cfg["training"]["learning_rate"]),
        weight_decay=float(cfg["training"]["weight_decay"]),
        lr_after_unfreeze_factor=float(cfg["training"]["lr_after_unfreeze_factor"]),
        freeze_epochs=int(cfg["model"]["freeze_epochs"]),
        audio_unfreeze_num_layers=int(cfg["model"]["audio_unfreeze_num_layers"]),
        log_interval=int(cfg["training"]["log_interval"]),
        device=device_cfg,
        output_dir=output_dir,
    )

    trainer = Trainer(
        model=model,
        loss_fn=loss_fn,
        cfg=trainer_cfg,
    )

    if args.eval_only:
        if args.checkpoint is None:
            raise ValueError("--eval_only requires --checkpoint")
        print("Running evaluation on test set...")
        metrics = trainer.evaluate(test_loader, split_name="test")
        print(metrics)
        return

    # Train then evaluate on test set
    trainer.fit(train_loader, val_loader=val_loader)
    print("Training complete. Evaluating on test set with final checkpoint...")
    trainer.evaluate(test_loader, split_name="test")


if __name__ == "__main__":
    main()

