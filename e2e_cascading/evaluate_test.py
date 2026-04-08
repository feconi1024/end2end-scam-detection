from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping

os.environ.setdefault("PYTORCH_NVML_BASED_CUDA_CHECK", "1")

import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer, WhisperProcessor

_track_root = Path(__file__).resolve().parent
_repo_root = _track_root.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from e2e_cascading.src.dataset import (
    CharCTCTokenizer,
    TeleAntiFraudDataset,
    create_collate_fn,
    load_config,
)
from e2e_cascading.src.device_utils import resolve_runtime_device
from e2e_cascading.src.model import DifferentiableCascadeModel
from experiments.common_metrics import (
    build_standard_report,
    infer_eval_scope,
    infer_manifest_family,
    write_json,
    write_predictions_csv,
)


def build_label_mapping(cfg: Dict[str, Any]) -> Dict[str, int]:
    scam_label = cfg["dataset"]["scam_label"]
    non_scam_label = cfg["dataset"]["non_scam_label"]
    return {non_scam_label: 0, scam_label: 1}


def get_dataset_audio_cfg(cfg: Dict[str, Any]) -> Dict[str, float]:
    dataset_cfg = cfg.get("dataset", {})
    return {
        "fixed_duration_seconds": float(dataset_cfg.get("fixed_duration_seconds", 15.0)),
        "train_noise_max_amp": float(dataset_cfg.get("train_noise_max_amp", 0.005)),
    }


def extract_model_state_dict(payload: Any) -> Mapping[str, torch.Tensor]:
    if isinstance(payload, Mapping):
        for key in ("model_state_dict", "state_dict"):
            maybe = payload.get(key)
            if isinstance(maybe, Mapping):
                return maybe
    if isinstance(payload, Mapping):
        return payload
    raise TypeError(f"Unsupported checkpoint payload type: {type(payload)!r}")


def infer_checkpoint_ctc_vocab_size(state_dict: Mapping[str, torch.Tensor]) -> int | None:
    weight = state_dict.get("ctc_head.weight")
    if weight is None:
        return None
    if not hasattr(weight, "shape") or len(weight.shape) != 2:
        return None
    return int(weight.shape[0])


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
        loader_kwargs["multiprocessing_context"] = "spawn"
        loader_kwargs["prefetch_factor"] = int(training_cfg.get("prefetch_factor", 1))
        loader_kwargs["persistent_workers"] = bool(training_cfg.get("persistent_workers", False))

    return DataLoader(**loader_kwargs)


def infer_train_family(cfg: Dict[str, Any], repo_root: Path) -> str:
    train_manifest = Path(str(cfg["dataset"].get("train_manifest", "")))
    if not train_manifest.is_absolute():
        train_manifest = (repo_root / train_manifest).resolve()
    return infer_manifest_family(train_manifest)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate a checkpoint on a manifest and report standardized metrics + latency."
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
        "--manifest_override",
        type=Path,
        default=None,
        help="Optional manifest CSV to use instead of dataset.test_manifest from config.",
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
    parser.add_argument("--output_json", type=Path, default=None)
    parser.add_argument("--predictions_csv", type=Path, default=None)
    parser.add_argument("--model_name", type=str, default="e2e_cascading")
    parser.add_argument("--train_family", type=str, default=None)
    parser.add_argument("--eval_family", type=str, default=None)
    parser.add_argument("--eval_scope", type=str, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    config_path = Path(args.config).resolve()
    config_dir = config_path.parent
    repo_root = config_dir.parent.parent

    def _resolve_manifest(p: str | Path) -> Path:
        path = Path(p)
        return (repo_root / path).resolve() if not path.is_absolute() else path.resolve()

    label_mapping = build_label_mapping(cfg)
    id2label = {v: k for k, v in label_mapping.items()}

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
        str(train_manifest),
        blank_token_id=ctc_blank_id,
        min_char_freq=ctc_min_char_freq,
    )
    checkpoint_payload = torch.load(args.checkpoint, map_location="cpu")
    checkpoint_state = extract_model_state_dict(checkpoint_payload)
    checkpoint_ctc_vocab_size = infer_checkpoint_ctc_vocab_size(checkpoint_state)
    if checkpoint_ctc_vocab_size is not None and checkpoint_ctc_vocab_size != ctc_tokenizer.vocab_size:
        print(
            "Warning: checkpoint CTC vocab size "
            f"({checkpoint_ctc_vocab_size}) differs from manifest-derived vocab size "
            f"({ctc_tokenizer.vocab_size}). Using the checkpoint size for evaluation-time model loading."
        )
    test_manifest = _resolve_manifest(args.manifest_override or cfg["dataset"].get("test_manifest", ""))

    test_ds = TeleAntiFraudDataset(
        manifest_path=str(test_manifest),
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
    ctc_vocab_size = checkpoint_ctc_vocab_size or ctc_tokenizer.vocab_size
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

    model.load_state_dict(checkpoint_state)
    model.eval()

    device_str = resolve_runtime_device(str(cfg["training"].get("device", "cuda")))
    device = torch.device(device_str)
    model.to(device)
    use_cuda = device.type == "cuda"

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

    all_labels: List[str] = []
    all_preds: List[str] = []
    total_infer_time = 0.0
    total_e2e_time = 0.0
    total_audio_seconds = 0.0
    prediction_rows: List[Dict[str, Any]] = []

    with torch.no_grad():
        for warmup_index, batch in enumerate(test_loader):
            if warmup_index >= args.warmup_batches:
                break
            _ = model(
                input_features=batch["input_features"].to(device),
                audio_attention_mask=batch["audio_attention_mask"].to(device),
            )
            if device.type == "cuda":
                torch.cuda.synchronize()

        start_e2e = time.perf_counter()
        for batch in test_loader:
            start_batch = time.perf_counter()
            input_features = batch["input_features"].to(device)
            audio_attention_mask = batch["audio_attention_mask"].to(device)
            labels = batch["labels"].to(device)

            durations = batch.get("audio_durations")
            duration_list: List[float] = []
            if durations is not None:
                duration_list = [float(v) for v in durations.cpu().tolist()]
                total_audio_seconds += sum(duration_list)

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
            preds = logits.argmax(dim=-1).cpu().tolist()
            batch_labels = labels.cpu().tolist()

            for idx, (gold_id, pred_id) in enumerate(zip(batch_labels, preds)):
                gold = id2label[int(gold_id)]
                pred = id2label[int(pred_id)]
                all_labels.append(gold)
                all_preds.append(pred)
                prediction_rows.append(
                    {
                        'example_index': len(prediction_rows),
                        'gold_label': gold,
                        'predicted_label': pred,
                        'audio_duration_seconds': duration_list[idx] if idx < len(duration_list) else None,
                    }
                )
        if device.type == "cuda":
            torch.cuda.synchronize()
        end_e2e = time.perf_counter()
        total_e2e_time = max(total_e2e_time, end_e2e - start_e2e)

    if len(all_labels) == 0:
        print("No test examples found.")
        return

    report = build_standard_report(
        gold_labels=all_labels,
        predicted_labels=all_preds,
        model_name=args.model_name,
        train_family=args.train_family or infer_train_family(cfg, repo_root),
        eval_family=args.eval_family or infer_manifest_family(test_manifest),
        eval_scope=args.eval_scope or infer_eval_scope(test_manifest),
        total_runtime_sec=total_e2e_time,
        total_audio_seconds=total_audio_seconds,
        n_skipped=0,
        metadata={
            'config': str(args.config),
            'checkpoint': str(args.checkpoint),
            'device': device_str,
        },
        latency_breakdown={
            'forward_only_sec': total_infer_time,
            'end_to_end_sec': total_e2e_time,
        },
    )

    if args.output_json is not None:
        write_json(args.output_json, report)
    if args.predictions_csv is not None:
        write_predictions_csv(args.predictions_csv, prediction_rows)

    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
