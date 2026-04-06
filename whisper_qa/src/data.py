from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import numpy as np
import pandas as pd
import soundfile as sf
import torch
from torch.utils.data import Dataset
from transformers import WhisperProcessor


@dataclass(frozen=True)
class ManifestRecord:
    index: int
    audio_path: str
    label: str
    transcript: str
    audio_duration_seconds: float
    manifest_path: str
    split_name: str
    raw_path: str


class TeleAntiFraudManifestDataset(Dataset):
    def __init__(self, records: Sequence[ManifestRecord]):
        self.records = list(records)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        record = self.records[index]
        return {
            "index": record.index,
            "audio_path": record.audio_path,
            "label": record.label,
            "transcript": record.transcript,
            "audio_duration_seconds": record.audio_duration_seconds,
            "manifest_path": record.manifest_path,
            "split_name": record.split_name,
            "raw_path": record.raw_path,
        }


def create_processor(model_cfg: Mapping[str, Any]) -> WhisperProcessor:
    processor = WhisperProcessor.from_pretrained(
        model_cfg["name"],
        language=model_cfg.get("language"),
        task=str(model_cfg.get("task", "transcribe")),
    )
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    predict_timestamps = not bool(model_cfg.get("no_timestamps", True))
    if hasattr(processor.tokenizer, "set_prefix_tokens"):
        processor.tokenizer.set_prefix_tokens(
            language=model_cfg.get("language"),
            task=str(model_cfg.get("task", "transcribe")),
            predict_timestamps=predict_timestamps,
        )
    return processor


def resolve_audio_path(path_value: str | Path, root_dir: Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path.resolve()

    candidates = [
        (root_dir / path).resolve(),
        (root_dir.parent / path).resolve(),
    ]
    existing = next((candidate for candidate in candidates if candidate.exists()), None)
    return existing or candidates[0]


def _duration_seconds(path_value: Path) -> float:
    try:
        return float(sf.info(str(path_value)).duration)
    except Exception:
        return float("nan")


def _stringify(value: Any) -> str:
    if value is None:
        return ""
    try:
        is_na = pd.isna(value)
        if isinstance(is_na, (bool, np.bool_)) and is_na:
            return ""
    except Exception:
        pass
    return str(value)


def _resolve_manifest_path(dataset_dir: Path, split_name: str) -> Path:
    candidates = [dataset_dir / f"{split_name}_manifest.csv"]
    if split_name == "val":
        candidates.append(dataset_dir / "validation_manifest.csv")
    elif split_name == "validation":
        candidates.append(dataset_dir / "val_manifest.csv")
    candidates.append(dataset_dir / "manifest.csv")

    for candidate in candidates:
        if candidate.exists():
            return candidate

    candidate_names = ", ".join(path.name for path in candidates)
    raise FileNotFoundError(
        f"Could not find a manifest for split '{split_name}' under {dataset_dir}. Tried: {candidate_names}"
    )


def load_manifest_records(
    dataset_path: Path,
    split_name: str,
    label_column: str = "label",
    text_column: str = "transcript",
    max_rows: Optional[int] = None,
) -> List[ManifestRecord]:
    if dataset_path.is_dir():
        manifest_path = _resolve_manifest_path(dataset_path, split_name)
        root_dir = dataset_path
    elif dataset_path.suffix.lower() == ".csv":
        manifest_path = dataset_path
        root_dir = dataset_path.parent
    else:
        raise ValueError(f"Unsupported dataset path: {dataset_path}")

    df = pd.read_csv(manifest_path)
    if max_rows is not None and max_rows > 0:
        df = df.head(max_rows)

    if "path" not in df.columns or label_column not in df.columns:
        raise ValueError(
            f"Manifest {manifest_path} must contain at least 'path' and '{label_column}' columns."
        )

    records: List[ManifestRecord] = []
    for row_index, row in df.iterrows():
        resolved_audio_path = resolve_audio_path(_stringify(row["path"]), root_dir=root_dir)
        transcript = _stringify(row[text_column]) if text_column in row else ""
        records.append(
            ManifestRecord(
                index=int(row_index),
                audio_path=str(resolved_audio_path),
                label=_stringify(row[label_column]),
                transcript=transcript,
                audio_duration_seconds=_duration_seconds(resolved_audio_path),
                manifest_path=str(manifest_path.resolve()),
                split_name=split_name,
                raw_path=_stringify(row["path"]),
            )
        )
    return records


def load_audio(path: Path, sampling_rate: int) -> np.ndarray:
    audio, sr = sf.read(str(path))
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)

    if sr != sampling_rate:
        duration = audio.shape[0] / sr
        target_len = max(1, int(duration * sampling_rate))
        x_old = np.linspace(0.0, 1.0, num=audio.shape[0])
        x_new = np.linspace(0.0, 1.0, num=target_len)
        audio = np.interp(x_new, x_old, audio).astype(np.float32)

    return audio.astype(np.float32)


def _truncate_transcript_text(
    transcript: str,
    duration_seconds: float,
    audio_chunk_seconds: float,
) -> str:
    if not transcript:
        return ""
    if np.isnan(duration_seconds) or duration_seconds <= audio_chunk_seconds:
        return transcript

    ratio = max(0.05, min(1.0, audio_chunk_seconds / max(duration_seconds, 1e-6)))
    approx_chars = max(1, int(len(transcript) * ratio))
    return transcript[:approx_chars]


def _ensure_prefix_tokens(tokenizer: Any, model_cfg: Mapping[str, Any]) -> List[int]:
    predict_timestamps = not bool(model_cfg.get("no_timestamps", True))
    if hasattr(tokenizer, "set_prefix_tokens"):
        tokenizer.set_prefix_tokens(
            language=model_cfg.get("language"),
            task=str(model_cfg.get("task", "transcribe")),
            predict_timestamps=predict_timestamps,
        )
    return list(getattr(tokenizer, "prefix_tokens", []))


def encode_asr_label_ids(
    tokenizer: Any,
    transcript: str,
    model_cfg: Mapping[str, Any],
) -> List[int]:
    prefix_tokens = _ensure_prefix_tokens(tokenizer, model_cfg)
    text_ids = tokenizer(transcript, add_special_tokens=False)["input_ids"]
    eos_token_id = int(getattr(tokenizer, "eos_token_id"))
    ids = prefix_tokens + list(text_ids) + [eos_token_id]

    max_label_tokens = int(model_cfg.get("transcript_max_label_tokens", 128))
    if len(ids) > max_label_tokens:
        ids = ids[:max_label_tokens]
        ids[-1] = eos_token_id
    return ids


def encode_transcript_cache_ids(
    tokenizer: Any,
    transcript: str,
    model_cfg: Mapping[str, Any],
) -> List[int]:
    prefix_tokens = _ensure_prefix_tokens(tokenizer, model_cfg)
    text_ids = tokenizer(transcript, add_special_tokens=False)["input_ids"]
    max_cache_tokens = int(model_cfg.get("transcript_max_cache_tokens", 96))
    if len(text_ids) > max_cache_tokens:
        text_ids = text_ids[:max_cache_tokens]
    return prefix_tokens + list(text_ids)


class WhisperQACollator:
    def __init__(self, processor: Any, config: Mapping[str, Any]):
        self.processor = processor
        self.model_cfg = config.get("model", {})
        self.data_cfg = config.get("data", {})
        self.sampling_rate = int(self.model_cfg.get("sampling_rate", 16000))
        self.audio_chunk_seconds = float(self.model_cfg.get("audio_chunk_seconds", 30.0))
        self.align_transcript = bool(self.model_cfg.get("align_transcript_to_audio_chunk", True))

    def __call__(self, features: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
        input_features: List[np.ndarray] = []
        label_tensors: List[torch.Tensor] = []
        transcript_cache_ids: List[torch.Tensor] = []
        transcript_texts: List[str] = []

        for feature in features:
            audio = load_audio(Path(feature["audio_path"]), sampling_rate=self.sampling_rate)
            mel = self.processor.feature_extractor(
                audio,
                sampling_rate=self.sampling_rate,
                return_attention_mask=False,
            )["input_features"][0]
            input_features.append(np.asarray(mel, dtype=np.float32))

            transcript = str(feature.get("transcript", ""))
            duration_seconds = float(feature.get("audio_duration_seconds", float("nan")))
            if self.align_transcript:
                transcript = _truncate_transcript_text(
                    transcript=transcript,
                    duration_seconds=duration_seconds,
                    audio_chunk_seconds=self.audio_chunk_seconds,
                )
            transcript_texts.append(transcript)

            asr_ids = encode_asr_label_ids(self.processor.tokenizer, transcript, self.model_cfg)
            cache_ids = encode_transcript_cache_ids(self.processor.tokenizer, transcript, self.model_cfg)
            label_tensors.append(torch.tensor(asr_ids, dtype=torch.long))
            transcript_cache_ids.append(torch.tensor(cache_ids, dtype=torch.long))

        max_label_len = max(tensor.shape[0] for tensor in label_tensors)
        padded_labels = torch.full((len(label_tensors), max_label_len), fill_value=-100, dtype=torch.long)
        for idx, tensor in enumerate(label_tensors):
            padded_labels[idx, : tensor.shape[0]] = tensor

        stacked_inputs = torch.tensor(np.stack(input_features, axis=0), dtype=torch.float32)
        return {
            "input_features": stacked_inputs,
            "asr_labels": padded_labels,
            "transcript_cache_ids": transcript_cache_ids,
            "transcript_texts": transcript_texts,
            "labels": [str(feature["label"]) for feature in features],
            "audio_paths": [str(feature["audio_path"]) for feature in features],
            "audio_durations": [float(feature.get("audio_duration_seconds", float("nan"))) for feature in features],
            "manifest_paths": [str(feature.get("manifest_path", "")) for feature in features],
            "split_names": [str(feature.get("split_name", "")) for feature in features],
            "row_indices": [int(feature.get("index", 0)) for feature in features],
            "raw_paths": [str(feature.get("raw_path", feature["audio_path"])) for feature in features],
        }
