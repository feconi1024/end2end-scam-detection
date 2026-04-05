from __future__ import annotations

import argparse
import platform
import time
from pathlib import Path
from typing import Any, Dict, List

import torch
from torch.utils.data import DataLoader
from transformers import WhisperProcessor, BertTokenizer
from sklearn.metrics import f1_score

from e2e_cascading.src.dataset import (
    CharCTCTokenizer,
    TeleAntiFraudDataset,
    load_config,
    create_collate_fn,
)
from e2e_cascading.src.device_utils import resolve_runtime_device
from e2e_cascading.src.model import DifferentiableCascadeModel


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
    parser = argparse.ArgumentParser(
        description="Evaluate a checkpoint on the test set and report F1 + latency per minute of audio."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("e2e_cascading/config/default_config.yaml"),
        help="Path to YAML config.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to model checkpoint (.pt) to evaluate.",
    )
    parser.add_argument(
        "--latency_num_workers",
        type=int,
        default=0,
        help="Num workers to use for latency measurement (default: 0 for true end-to-end timing).",
    )
    parser.add_argument(
        "--warmup_batches",
        type=int,
        default=2,
        help="Warmup batches to run (excluded from timing) to avoid first-batch overhead.",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    config_path = Path(args.config).resolve()
    config_dir = config_path.parent
    repo_root = config_dir.parent.parent

    def _resolve_manifest(p: str) -> str:
        path = Path(p)
        return str(repo_root / p) if not path.is_absolute() else p

    label_mapping = build_label_mapping(cfg)

    whisper_name = cfg["model"]["whisper_model_name"]
    bert_name = cfg["model"]["bert_model_name"]

    processor = WhisperProcessor.from_pretrained(whisper_name)
    BertTokenizer.from_pretrained(bert_name)

    sr = int(cfg["dataset"]["sample_rate"])
    audio_cfg = get_dataset_audio_cfg(cfg)
    ctc_blank_id = int(cfg["model"]["ctc_blank_token_id"])
    ctc_min_char_freq = int(cfg["model"].get("ctc_min_char_freq", 1))
    train_manifest = _resolve_manifest(cfg["dataset"].get("train_manifest", ""))
    ctc_tokenizer = CharCTCTokenizer.build_from_manifest(
        train_manifest,
        blank_token_id=ctc_blank_id,
        min_char_freq=ctc_min_char_freq,
    )
    test_manifest = _resolve_manifest(cfg["dataset"].get("test_manifest", ""))
    print(f"Tokenizer manifest: {train_manifest}")
    print(f"Test manifest: {test_manifest}")
    print(f"Checkpoint: {args.checkpoint}")

    test_ds = TeleAntiFraudDataset(
        manifest_path=test_manifest,
        tokenizer=ctc_tokenizer,
        sample_rate=sr,
        split="test",
        label_mapping=label_mapping,
        **audio_cfg,
    )

    pad_token_id = ctc_tokenizer.pad_token_id
    collate_fn = create_collate_fn(
        processor=processor,
        pad_token_id=pad_token_id,
        sample_rate=sr,
    )

    num_labels = int(cfg["model"]["num_labels"])
    ctc_vocab_size = ctc_tokenizer.vocab_size

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

    state = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(state)
    model.eval()

    device_str = resolve_runtime_device(str(cfg["training"].get("device", "cuda")))
    device = torch.device(device_str)
    model.to(device)
    use_cuda = device.type == "cuda"

    # For end-to-end latency (disk I/O + feature extraction + model), default to
    # single-process DataLoader timing by using num_workers=0.
    requested_workers = 0 if platform.system() == "Windows" else int(args.latency_num_workers)
    batch_size = int(cfg["training"]["batch_size"])
    test_loader = build_dataloader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        requested_workers=requested_workers,
        use_cuda=use_cuda,
        cfg=cfg,
    )

    all_labels: List[int] = []
    all_preds: List[int] = []
    total_infer_time = 0.0
    total_e2e_time = 0.0
    total_audio_seconds = 0.0

    with torch.no_grad():
        # Warmup (excluded from timing)
        for _i, batch in enumerate(test_loader):
            if _i >= args.warmup_batches:
                break
            _ = model(
                input_features=batch["input_features"].to(device),
                audio_attention_mask=batch["audio_attention_mask"].to(device),
            )
            if device.type == "cuda":
                torch.cuda.synchronize()

        # Full end-to-end timing includes waiting for DataLoader + preprocessing
        start_e2e = time.perf_counter()
        for batch in test_loader:
            start_batch = time.perf_counter()
            input_features = batch["input_features"].to(device)
            audio_attention_mask = batch["audio_attention_mask"].to(device)
            labels = batch["labels"].to(device)

            durations = batch.get("audio_durations")
            if durations is not None:
                total_audio_seconds += float(durations.sum().item())

            start = time.perf_counter()
            outputs = model(
                input_features=input_features,
                audio_attention_mask=audio_attention_mask,
            )
            if device.type == "cuda":
                torch.cuda.synchronize()
            end = time.perf_counter()
            total_infer_time += end - start
            total_e2e_time += time.perf_counter() - start_batch

            logits = outputs["classification_logits"]
            preds = logits.argmax(dim=-1)

            all_labels.extend(labels.cpu().tolist())
            all_preds.extend(preds.cpu().tolist())
        if device.type == "cuda":
            torch.cuda.synchronize()
        end_e2e = time.perf_counter()
        total_e2e_time = max(total_e2e_time, end_e2e - start_e2e)

    if len(all_labels) == 0:
        print("No test examples found.")
        return

    avg_mode = "binary" if len(set(all_labels)) == 2 else "macro"
    f1 = f1_score(all_labels, all_preds, average=avg_mode)

    audio_minutes = total_audio_seconds / 60.0 if total_audio_seconds > 0 else float("nan")
    if audio_minutes > 0:
        latency_per_min = total_infer_time / audio_minutes
    else:
        latency_per_min = float("nan")

    print(f"Test F1 ({avg_mode}): {f1:.4f}")
    print(f"Total inference time (model forward only): {total_infer_time:.3f} s")
    print(f"Total end-to-end time (data + features + model): {total_e2e_time:.3f} s")
    print(f"Total audio duration: {audio_minutes:.3f} min")
    if audio_minutes > 0:
        latency_per_min_e2e = total_e2e_time / audio_minutes
    else:
        latency_per_min_e2e = float("nan")
    print(f"Latency (forward-only): {latency_per_min:.3f} s per minute of audio")
    print(f"Latency (end-to-end): {latency_per_min_e2e:.3f} s per minute of audio")


if __name__ == "__main__":
    main()

