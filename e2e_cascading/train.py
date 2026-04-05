from __future__ import annotations

import argparse
import platform
from pathlib import Path
from typing import Dict, Any

import torch
from torch.utils.data import DataLoader
from transformers import WhisperProcessor, BertTokenizer

from e2e_cascading.src.dataset import (
    CharCTCTokenizer,
    TeleAntiFraudDataset,
    load_config,
    create_collate_fn,
)
from e2e_cascading.src.device_utils import resolve_runtime_device
from e2e_cascading.src.model import DifferentiableCascadeModel
from e2e_cascading.src.loss import JointCTCSLULoss
from e2e_cascading.src.trainer import Trainer, TrainerConfig


def build_label_mapping(cfg: Dict[str, Any]) -> Dict[str, int]:
    scam_label = cfg["dataset"]["scam_label"]
    non_scam_label = cfg["dataset"]["non_scam_label"]
    # Non-scam = 0, scam = 1
    return {non_scam_label: 0, scam_label: 1}


def get_dataset_audio_cfg(cfg: Dict[str, Any]) -> Dict[str, float]:
    dataset_cfg = cfg.get("dataset", {})
    return {
        "fixed_duration_seconds": float(dataset_cfg.get("fixed_duration_seconds", 15.0)),
        "train_noise_max_amp": float(dataset_cfg.get("train_noise_max_amp", 0.005)),
    }


def build_dataloader(
    dataset,
    *,
    batch_size: int,
    shuffle: bool,
    collate_fn,
    requested_workers: int,
    use_cuda: bool,
    cfg: Dict[str, Any],
) -> DataLoader:
    training_cfg = cfg.get("training", {})
    system = platform.system()
    if system == "Windows":
        num_workers = 0
    else:
        num_workers = max(0, int(requested_workers))

    loader_kwargs: Dict[str, Any] = {
        "dataset": dataset,
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": num_workers,
        "collate_fn": collate_fn,
        "pin_memory": bool(use_cuda and training_cfg.get("pin_memory", True)),
    }

    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = int(training_cfg.get("prefetch_factor", 2))
        loader_kwargs["persistent_workers"] = bool(training_cfg.get("persistent_workers", True))
        multiprocessing_context = training_cfg.get("multiprocessing_context")
        if multiprocessing_context:
            loader_kwargs["multiprocessing_context"] = str(multiprocessing_context)

    return DataLoader(**loader_kwargs)


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

    # Feature processor and tokenizers
    whisper_name = cfg["model"]["whisper_model_name"]
    bert_name = cfg["model"]["bert_model_name"]

    whisper_processor = WhisperProcessor.from_pretrained(whisper_name)
    # The BERT tokenizer is not used for CTC anymore; we keep the semantic
    # classifier vocabulary and use a lightweight character tokenizer for the
    # auxiliary speech loss to avoid frame-level OOM.
    BertTokenizer.from_pretrained(bert_name)

    # Datasets
    sr = int(cfg["dataset"]["sample_rate"])
    audio_cfg = get_dataset_audio_cfg(cfg)
    train_manifest = _resolve_manifest(cfg["dataset"].get("train_manifest", ""))
    val_manifest = _resolve_manifest(cfg["dataset"].get("val_manifest", ""))
    test_manifest = _resolve_manifest(cfg["dataset"].get("test_manifest", ""))
    ctc_blank_id = int(cfg["model"]["ctc_blank_token_id"])
    ctc_min_char_freq = int(cfg["model"].get("ctc_min_char_freq", 1))
    ctc_tokenizer = CharCTCTokenizer.build_from_manifest(
        train_manifest,
        blank_token_id=ctc_blank_id,
        min_char_freq=ctc_min_char_freq,
    )
    print(f"CTC char vocab size: {ctc_tokenizer.vocab_size}")

    train_ds = TeleAntiFraudDataset(
        manifest_path=train_manifest,
        tokenizer=ctc_tokenizer,
        sample_rate=sr,
        split="train",
        label_mapping=label_mapping,
        **audio_cfg,
    )
    val_ds = TeleAntiFraudDataset(
        manifest_path=val_manifest,
        tokenizer=ctc_tokenizer,
        sample_rate=sr,
        split="val",
        label_mapping=label_mapping,
        **audio_cfg,
    )
    test_ds = TeleAntiFraudDataset(
        manifest_path=test_manifest,
        tokenizer=ctc_tokenizer,
        sample_rate=sr,
        split="test",
        label_mapping=label_mapping,
        **audio_cfg,
    )

    # Collate function
    pad_token_id = ctc_tokenizer.pad_token_id
    collate_fn = create_collate_fn(
        processor=whisper_processor,
        pad_token_id=pad_token_id,
        sample_rate=sr,
    )

    # Model
    num_labels = int(cfg["model"]["num_labels"])
    ctc_vocab_size = ctc_tokenizer.vocab_size

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

    # Compute class weights from training set to handle imbalance
    from collections import Counter
    label_counts = Counter(ex.label_str for ex in train_ds.examples)
    total_samples = sum(label_counts.values())
    num_classes = len(label_mapping)
    class_weights = torch.zeros(num_classes, dtype=torch.float32)
    for lbl, idx in label_mapping.items():
        count = label_counts.get(lbl, 1)
        class_weights[idx] = total_samples / (num_classes * count)
    print(f"Class weights: {class_weights.tolist()}")

    # Loss and trainer
    loss_fn = JointCTCSLULoss(
        ctc_blank_id=ctc_blank_id,
        ctc_weight=cfg["training"]["ctc_weight"],
        slu_weight=cfg["training"]["slu_weight"],
        class_weights=class_weights,
    )

    output_dir = Path(cfg["training"]["output_dir"])

    # Device: keep the requested device and let actual initialization fail fast
    # if CUDA is unavailable, instead of silently falling back to CPU.
    device_cfg = resolve_runtime_device(str(cfg["training"].get("device", "cuda")))
    use_cuda = device_cfg.startswith("cuda")

    _requested_workers = int(cfg["training"].get("num_workers", 4))
    batch_size = int(cfg["training"]["batch_size"])
    print(
        f"Building DataLoaders with num_workers={_requested_workers}, "
        f"persistent_workers={bool(cfg['training'].get('persistent_workers', True))}, "
        f"prefetch_factor={int(cfg['training'].get('prefetch_factor', 2))}"
    )
    train_loader = build_dataloader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        requested_workers=_requested_workers,
        use_cuda=use_cuda,
        cfg=cfg,
    )
    val_loader = build_dataloader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        requested_workers=_requested_workers,
        use_cuda=use_cuda,
        cfg=cfg,
    )
    test_loader = build_dataloader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        requested_workers=_requested_workers,
        use_cuda=use_cuda,
        cfg=cfg,
    )

    # Effective steps per epoch for LR scheduler
    accum_steps = int(cfg["training"].get("gradient_accumulation_steps", 1))
    steps_per_epoch = len(train_loader)

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
        gradient_accumulation_steps=accum_steps,
        max_grad_norm=float(cfg["training"].get("max_grad_norm", 1.0)),
        warmup_ratio=float(cfg["training"].get("warmup_ratio", 0.05)),
        mixed_precision=str(cfg["training"].get("mixed_precision", "bf16")),
    )

    trainer = Trainer(
        model=model,
        loss_fn=loss_fn,
        cfg=trainer_cfg,
        steps_per_epoch=steps_per_epoch,
    )

    if args.eval_only:
        if args.checkpoint is None:
            raise ValueError("--eval_only requires --checkpoint")
        print("Running evaluation on test set...")
        metrics = trainer.evaluate(test_loader, split_name="test")
        print(metrics)
        return

    # Train then evaluate on test set
    fit_result = trainer.fit(train_loader, val_loader=val_loader)

    best_ckpt = fit_result.get("best_checkpoint")
    if best_ckpt is not None:
        state = torch.load(best_ckpt, map_location="cpu")
        model.load_state_dict(state)
        print(
            "Training complete. Evaluating on test set with best validation checkpoint "
            f"({best_ckpt})..."
        )
    else:
        print("Training complete. Evaluating on test set with final checkpoint...")
    trainer.evaluate(test_loader, split_name="test")


if __name__ == "__main__":
    main()

