from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, MutableMapping, Optional, Tuple

import numpy as np
import pandas as pd
import soundfile as sf
import torch
from datasets import Audio, Dataset, DatasetDict, load_from_disk
from transformers import WhisperProcessor


DEFAULT_LABELS: Tuple[str, str] = ("non_scam", "scam")


def build_label_mappings(config: Mapping[str, Any]) -> Tuple[Dict[str, int], Dict[int, str]]:
    configured_labels = config.get("labels")
    if configured_labels is None:
        labels = list(DEFAULT_LABELS)
    else:
        labels = [str(label).strip() for label in configured_labels if str(label).strip()]

    if sorted(labels) != sorted(DEFAULT_LABELS):
        raise ValueError(
            "Classifier labels must contain exactly 'non_scam' and 'scam'. "
            f"Got: {labels}"
        )

    label2id = {label: idx for idx, label in enumerate(labels)}
    id2label = {idx: label for label, idx in label2id.items()}
    return label2id, id2label


def canonicalize_label(raw_label: Any) -> str:
    if raw_label is None:
        raise ValueError("Encountered empty label in classifier dataset.")

    text = str(raw_label).strip().lower()
    if not text:
        raise ValueError("Encountered blank label in classifier dataset.")

    if text in {"non_scam", "non scam", "ham", "pos"}:
        return "non_scam"
    if text in {"scam", "fraud", "neg"}:
        return "scam"

    raise ValueError(f"Unsupported classifier label: {raw_label}")


def _build_dataset_from_manifest(manifest: Path, root_dir: Path, sampling_rate: int) -> Dataset:
    df = pd.read_csv(manifest)

    if "path" not in df.columns or "label" not in df.columns:
        raise ValueError(
            f"Manifest {manifest} must contain at least 'path' and 'label' columns."
        )

    def _resolve_path(p: str) -> str:
        p_path = Path(p)
        if not p_path.is_absolute():
            candidates = [
                (root_dir / p).resolve(),
                (root_dir.parent / p).resolve(),
            ]
            existing = next((candidate for candidate in candidates if candidate.exists()), None)
            p_path = existing or candidates[0]
        else:
            p_path = p_path.resolve()
        return str(p_path)

    df["audio"] = df["path"].apply(_resolve_path)

    if "transcript" not in df.columns:
        df["transcript"] = ""

    def _duration_seconds(path_str: str) -> float:
        try:
            return float(sf.info(path_str).duration)
        except Exception:
            return float("nan")

    df["audio_duration_seconds"] = df["audio"].apply(_duration_seconds)

    ds = Dataset.from_pandas(
        df[["audio", "label", "transcript", "audio_duration_seconds"]],
        preserve_index=False,
    )
    ds = ds.cast_column("audio", Audio(sampling_rate=sampling_rate))
    return ds


def _resolve_manifest_path(dataset_dir: Path, split_name: str) -> Path:
    candidates = [dataset_dir / f"{split_name}_manifest.csv"]

    if split_name == "val":
        candidates.append(dataset_dir / "validation_manifest.csv")
    elif split_name == "validation":
        candidates.append(dataset_dir / "val_manifest.csv")

    for candidate in candidates:
        if candidate.exists():
            return candidate

    candidate_names = ", ".join(path.name for path in candidates)
    raise FileNotFoundError(
        f"Could not find a manifest for split '{split_name}' under {dataset_dir}. "
        f"Tried: {candidate_names}"
    )


def _load_dataset_dict_from_path(
    dataset_path: Path,
    sampling_rate: int,
    train_split: str,
    eval_split: str,
) -> Tuple[DatasetDict, str]:
    dataset_dict: DatasetDict

    if dataset_path.is_dir():
        try:
            dataset_dict = load_from_disk(str(dataset_path))
            dataset_dict = dataset_dict.cast_column("audio", Audio(sampling_rate=sampling_rate))

            if train_split not in dataset_dict:
                raise KeyError(
                    f"Split '{train_split}' not found in dataset at {dataset_path}. "
                    f"Available: {list(dataset_dict.keys())}"
                )

            resolved_eval_split = eval_split
            if resolved_eval_split not in dataset_dict:
                if resolved_eval_split == "val" and "validation" in dataset_dict:
                    resolved_eval_split = "validation"
                elif resolved_eval_split == "validation" and "val" in dataset_dict:
                    resolved_eval_split = "val"
                else:
                    raise KeyError(
                        f"Split '{eval_split}' not found in dataset at {dataset_path}. "
                        f"Available: {list(dataset_dict.keys())}"
                    )
            return dataset_dict, resolved_eval_split
        except Exception:
            train_manifest = _resolve_manifest_path(dataset_path, train_split)
            eval_manifest = _resolve_manifest_path(dataset_path, eval_split)
            root_dir = dataset_path
            train_ds = _build_dataset_from_manifest(
                train_manifest,
                root_dir=root_dir,
                sampling_rate=sampling_rate,
            )
            eval_ds = _build_dataset_from_manifest(
                eval_manifest,
                root_dir=root_dir,
                sampling_rate=sampling_rate,
            )
            dataset_dict = DatasetDict({train_split: train_ds, eval_split: eval_ds})
            return dataset_dict, eval_split

    if dataset_path.suffix.lower() == ".csv":
        root_dir = dataset_path.parent
        train_ds = _build_dataset_from_manifest(
            dataset_path,
            root_dir=root_dir,
            sampling_rate=sampling_rate,
        )
        dataset_dict = DatasetDict({train_split: train_ds, eval_split: train_ds})
        return dataset_dict, eval_split

    dataset_dict = load_from_disk(str(dataset_path))
    dataset_dict = dataset_dict.cast_column("audio", Audio(sampling_rate=sampling_rate))
    return dataset_dict, eval_split


def prepare_classification_mapping_fn(
    label2id: Mapping[str, int],
    label_column: str = "label",
) -> Any:
    def _map_batch(batch: MutableMapping[str, Any]) -> MutableMapping[str, Any]:
        raw_label = batch.get(label_column)
        if isinstance(raw_label, (list, tuple)):
            raw_label = raw_label[0] if raw_label else None

        canonical = canonicalize_label(raw_label)
        batch["label"] = canonical
        batch["class_label"] = int(label2id[canonical])
        return batch

    return _map_batch


def load_and_prepare_classification_datasets(
    dataset_path: Path,
    processor: WhisperProcessor,
    config: Mapping[str, Any],
    train_split: str = "train",
    eval_split: str = "val",
    label_column: str = "label",
    num_proc: Optional[int] = None,
) -> Tuple[Any, Any, Dict[str, int], Dict[int, str]]:
    sampling_rate = int(config.get("sampling_rate", 16000))
    label2id, id2label = build_label_mappings(config)

    dataset_dict, resolved_eval_split = _load_dataset_dict_from_path(
        dataset_path=dataset_path,
        sampling_rate=sampling_rate,
        train_split=train_split,
        eval_split=eval_split,
    )

    mapping_fn = prepare_classification_mapping_fn(
        label2id=label2id,
        label_column=label_column,
    )

    processed = dataset_dict.map(
        mapping_fn,
        remove_columns=[
            col
            for col in dataset_dict[train_split].column_names
            if col not in ("audio", "label", "class_label", "audio_duration_seconds")
        ],
        num_proc=num_proc,
    )

    return processed[train_split], processed[resolved_eval_split], label2id, id2label


@dataclass
class DataCollatorAudioClassificationWithPadding:
    processor: WhisperProcessor

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        fe = self.processor.feature_extractor
        input_feats_list: List[Dict[str, Any]] = []
        labels: List[int] = []

        for feature in features:
            audio = feature["audio"]
            array = audio["array"]
            sr = audio["sampling_rate"]

            sampling_rate = getattr(fe, "sampling_rate", sr)
            if sr != sampling_rate:
                duration = array.shape[0] / sr
                target_len = int(duration * sampling_rate)
                x_old = np.linspace(0, 1, num=array.shape[0])
                x_new = np.linspace(0, 1, num=target_len)
                array = np.interp(x_new, x_old, array).astype(np.float32)

            inputs = fe(
                array,
                sampling_rate=sampling_rate,
                return_attention_mask=False,
            )
            input_feats_list.append({"input_features": inputs["input_features"][0]})
            labels.append(int(feature["class_label"]))

        batch = fe.pad(
            input_feats_list,
            padding=True,
            return_tensors="pt",
        )
        batch["labels"] = torch.tensor(labels, dtype=torch.long)
        return batch
