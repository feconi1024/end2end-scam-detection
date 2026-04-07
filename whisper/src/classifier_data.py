from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path
from typing import Any, Dict, List, Mapping, MutableMapping, Optional, Tuple

import librosa
import numpy as np
import pandas as pd
import soundfile as sf
import torch
from datasets import Dataset, DatasetDict, load_from_disk
from transformers import WhisperProcessor


DEFAULT_LABELS: Tuple[str, str] = ("non_scam", "scam")
logger = logging.getLogger(__name__)


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


def build_family_mappings(
    train_dataset,
    eval_dataset,
) -> Tuple[Dict[str, int], Dict[int, str]]:
    families = []
    for dataset in (train_dataset, eval_dataset):
        if dataset is None or "family" not in dataset.column_names:
            continue
        families.extend(str(family) for family in dataset["family"])

    unique_families = sorted({family for family in families if family and family != "nan"})
    family2id = {family: idx for idx, family in enumerate(unique_families)}
    id2family = {idx: family for family, idx in family2id.items()}
    return family2id, id2family


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

    df["audio_path"] = df["path"].apply(_resolve_path)

    if "transcript" not in df.columns:
        df["transcript"] = ""
    if "family" not in df.columns:
        df["family"] = df["path"].astype(str).str.extract(r"merged_result/([^/]+)/")[0].fillna("")

    def _duration_seconds(path_str: str) -> float:
        try:
            return float(sf.info(path_str).duration)
        except Exception:
            return float("nan")

    df["audio_duration_seconds"] = df["audio_path"].apply(_duration_seconds)

    ds = Dataset.from_pandas(
        df[["audio_path", "label", "transcript", "family", "audio_duration_seconds"]],
        preserve_index=False,
    )
    return ds


def audit_manifest_family_overlap(train_manifest: Path, eval_manifest: Path) -> Dict[str, Any]:
    train_df = pd.read_csv(train_manifest, usecols=["path", "label"])
    eval_df = pd.read_csv(eval_manifest, usecols=["path", "label"])

    def _family(df: pd.DataFrame) -> pd.Series:
        return df["path"].astype(str).str.extract(r"merged_result/([^/]+)/")[0]

    train_families = _family(train_df)
    eval_families = _family(eval_df)

    train_family_set = set(train_families.dropna())
    eval_family_set = set(eval_families.dropna())
    overlap = sorted(train_family_set & eval_family_set)

    train_family_labels = (
        pd.DataFrame({"family": train_families, "label": train_df["label"].astype(str)})
        .dropna()
        .groupby("family")["label"]
        .agg(lambda values: sorted(set(values)))
        .to_dict()
    )
    eval_family_labels = (
        pd.DataFrame({"family": eval_families, "label": eval_df["label"].astype(str)})
        .dropna()
        .groupby("family")["label"]
        .agg(lambda values: sorted(set(values)))
        .to_dict()
    )

    pure_overlap = [
        family
        for family in overlap
        if len(train_family_labels.get(family, [])) == 1 and len(eval_family_labels.get(family, [])) == 1
    ]

    return {
        "train_num_families": len(train_family_set),
        "eval_num_families": len(eval_family_set),
        "overlap_num_families": len(overlap),
        "overlap_families_preview": overlap[:10],
        "pure_overlap_num_families": len(pure_overlap),
        "pure_overlap_families_preview": pure_overlap[:10],
    }


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
) -> Tuple[DatasetDict, str, Optional[Dict[str, Any]]]:
    dataset_dict: DatasetDict

    if dataset_path.is_dir():
        try:
            dataset_dict = load_from_disk(str(dataset_path))

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
            return dataset_dict, resolved_eval_split, None
        except Exception:
            train_manifest = _resolve_manifest_path(dataset_path, train_split)
            eval_manifest = _resolve_manifest_path(dataset_path, eval_split)
            audit = audit_manifest_family_overlap(train_manifest, eval_manifest)
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
            return dataset_dict, eval_split, audit

    if dataset_path.suffix.lower() == ".csv":
        root_dir = dataset_path.parent
        train_ds = _build_dataset_from_manifest(
            dataset_path,
            root_dir=root_dir,
            sampling_rate=sampling_rate,
        )
        dataset_dict = DatasetDict({train_split: train_ds, eval_split: train_ds})
        return dataset_dict, eval_split, None

    dataset_dict = load_from_disk(str(dataset_path))
    return dataset_dict, eval_split, None


def prepare_classification_mapping_fn(
    label2id: Mapping[str, int],
    family2id: Optional[Mapping[str, int]] = None,
    label_column: str = "label",
    family_column: str = "family",
) -> Any:
    def _map_batch(batch: MutableMapping[str, Any]) -> MutableMapping[str, Any]:
        raw_label = batch.get(label_column)
        if isinstance(raw_label, (list, tuple)):
            raw_label = raw_label[0] if raw_label else None

        canonical = canonicalize_label(raw_label)
        batch["label"] = canonical
        batch["class_label"] = int(label2id[canonical])

        if family2id is not None:
            family_value = batch.get(family_column, "")
            if isinstance(family_value, (list, tuple)):
                family_value = family_value[0] if family_value else ""
            family = str(family_value).strip()
            batch["family"] = family
            batch["family_id"] = int(family2id.get(family, -100))
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

    dataset_dict, resolved_eval_split, split_audit = _load_dataset_dict_from_path(
        dataset_path=dataset_path,
        sampling_rate=sampling_rate,
        train_split=train_split,
        eval_split=eval_split,
    )

    if split_audit is not None and split_audit["overlap_num_families"] > 0:
        logger.warning(
            "Manifest family overlap detected between train and %s: %d/%d eval families also appear in train. "
            "This can make classifier results unrealistically high if families carry label-specific acoustic shortcuts. "
            "Examples: %s",
            resolved_eval_split,
            split_audit["overlap_num_families"],
            split_audit["eval_num_families"],
            ", ".join(split_audit["overlap_families_preview"]) or "n/a",
        )

    classifier_cfg = config.get("classifier", {})
    family_adv_cfg = classifier_cfg.get("family_adversarial", {})
    use_family_adversarial = bool(family_adv_cfg.get("enabled", False))

    family2id: Optional[Dict[str, int]] = None
    id2family: Dict[int, str] = {}
    if use_family_adversarial:
        family2id, id2family = build_family_mappings(
            train_dataset=dataset_dict[train_split],
            eval_dataset=dataset_dict[resolved_eval_split],
        )

    mapping_fn = prepare_classification_mapping_fn(
        label2id=label2id,
        family2id=family2id,
        label_column=label_column,
    )

    processed = dataset_dict.map(
        mapping_fn,
        remove_columns=[
            col
            for col in dataset_dict[train_split].column_names
            if col not in ("audio", "audio_path", "label", "class_label", "family", "family_id", "audio_duration_seconds")
        ],
        num_proc=num_proc,
    )

    return processed[train_split], processed[resolved_eval_split], label2id, id2label, family2id or {}, id2family


@dataclass
class DataCollatorAudioClassificationWithPadding:
    processor: WhisperProcessor

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        fe = self.processor.feature_extractor
        input_feats_list: List[Dict[str, Any]] = []
        labels: List[int] = []
        family_labels: List[int] = []
        invalid_paths: List[str] = []

        for feature in features:
            audio_path = feature.get("audio_path")
            if audio_path is None:
                audio_value = feature.get("audio")
                if isinstance(audio_value, str):
                    audio_path = audio_value
                elif isinstance(audio_value, dict):
                    audio_path = audio_value.get("path")

            try:
                if not audio_path:
                    raise ValueError("Missing audio path.")
                if not Path(audio_path).exists():
                    raise FileNotFoundError(f"Audio file not found: {audio_path}")
                sampling_rate = int(getattr(fe, "sampling_rate", 16000))
                array, _ = librosa.load(
                    str(audio_path),
                    sr=sampling_rate,
                    mono=True,
                )
                if array.ndim > 1:
                    array = np.mean(array, axis=0)
                array = np.asarray(array, dtype=np.float32)

                inputs = fe(
                    array,
                    sampling_rate=sampling_rate,
                    return_attention_mask=False,
                )
                input_feats_list.append({"input_features": inputs["input_features"][0]})
                labels.append(int(feature["class_label"]))
                if "family_id" in feature:
                    family_labels.append(int(feature["family_id"]))
            except Exception:
                invalid_paths.append(str(audio_path))
                continue

        if invalid_paths:
            preview = ", ".join(invalid_paths[:3])
            suffix = "" if len(invalid_paths) <= 3 else f" (+{len(invalid_paths) - 3} more)"
            logger.warning("Skipping %d invalid audio file(s): %s%s", len(invalid_paths), preview, suffix)

        if not input_feats_list:
            sampling_rate = int(getattr(fe, "sampling_rate", 16000))
            dummy_audio = np.zeros(int(sampling_rate * 0.1), dtype=np.float32)
            inputs = fe(
                dummy_audio,
                sampling_rate=sampling_rate,
                return_attention_mask=False,
            )
            input_feats_list.append({"input_features": inputs["input_features"][0]})
            fallback_label = int(features[0].get("class_label", 0)) if features else 0
            labels.append(fallback_label)
            if features and "family_id" in features[0]:
                family_labels.append(int(features[0].get("family_id", -100)))

        batch = fe.pad(
            input_feats_list,
            padding=True,
            return_tensors="pt",
        )
        batch["labels"] = torch.tensor(labels, dtype=torch.long)
        if family_labels:
            batch["family_labels"] = torch.tensor(family_labels, dtype=torch.long)
        return batch
