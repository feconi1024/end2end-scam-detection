from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, MutableMapping, Optional, Tuple

import json
import numpy as np
import pandas as pd
import soundfile as sf
import yaml
from datasets import Audio, Dataset, DatasetDict, load_from_disk
from transformers import WhisperProcessor


def load_config(config_path: Path) -> Dict[str, Any]:
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def create_processor(config: Mapping[str, Any]) -> Tuple[WhisperProcessor, int]:
    """
    Create the WhisperProcessor used by the WhiSLU pipeline.

    The processor language is configurable because the local TeleAntiFraud
    manifests currently contain mostly Chinese transcripts.
    """
    model_name = config["model_name"]
    language = config.get("language")
    task = str(config.get("task", "transcribe"))

    processor_kwargs: Dict[str, Any] = {"task": task}
    if language is not None:
        processor_kwargs["language"] = str(language)

    processor = WhisperProcessor.from_pretrained(model_name, **processor_kwargs)

    # Whisper commonly reuses EOS as PAD.
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    return processor, 0


def prepare_dataset_mapping_fn(
    processor: WhisperProcessor,
    config: Mapping[str, Any],
    label_column: str = "label",
    text_column: Optional[str] = None,
) -> Any:
    """
    Build a mapping function for TeleAntiFraud-style examples.

    Default full target format:
      {
        "Text": "<transcript>",
        "Intent": "scam"
      }

    Optional constant/column-backed domain can be enabled with
    `include_domain: true`, which yields:
      {
        "Text": "<transcript>",
        "Domain": "telephony",
        "Intent": "scam"
      }

    This keeps the main SLU prediction last, matching the original WhiSLU
    sequence-level multitask ordering more closely than the previous
    intent-first format.
    """

    tokenizer = processor.tokenizer
    default_domain = str(config.get("domain", "telephony"))
    include_domain = bool(config.get("include_domain", False))
    intent_only = bool(config.get("train_intent_only", False))
    intent_first = bool(config.get("intent_first", True))
    max_label_tokens = int(config.get("max_label_tokens", 448))
    max_text_target_tokens_cfg = config.get("max_text_target_tokens")
    max_text_target_tokens = (
        int(max_text_target_tokens_cfg)
        if max_text_target_tokens_cfg is not None
        else None
    )
    align_transcript_to_audio_chunk = bool(config.get("align_transcript_to_audio_chunk", True))
    audio_chunk_seconds = float(config.get("audio_chunk_seconds", 30.0))
    intent_loss_weight = float(config.get("intent_loss_weight", 1.0))
    domain_column = config.get("domain_column")

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

    def _resolve_domain(batch: MutableMapping[str, Any]) -> str:
        if not include_domain:
            return ""
        if domain_column:
            candidate = _stringify(batch.get(domain_column))
            if candidate:
                return candidate
        return default_domain

    def _build_json_obj(
        *,
        transcript: str,
        intent: str,
        domain: str,
    ) -> Dict[str, str]:
        json_obj: Dict[str, str] = {}
        if intent_first:
            json_obj["Intent"] = intent
            if not intent_only:
                json_obj["Text"] = transcript
                if include_domain:
                    json_obj["Domain"] = domain
        else:
            if not intent_only:
                json_obj["Text"] = transcript
                if include_domain:
                    json_obj["Domain"] = domain
            json_obj["Intent"] = intent
        return json_obj

    def _encode_text(text: str) -> List[int]:
        return tokenizer(text, add_special_tokens=False)["input_ids"]

    def _map_batch(batch: MutableMapping[str, Any]) -> MutableMapping[str, Any]:
        raw_label = batch.get(label_column)
        if isinstance(raw_label, (list, tuple)):
            raw_label = raw_label[0] if raw_label else ""

        intent = _stringify(raw_label)
        domain = _resolve_domain(batch)

        transcript = ""
        raw_transcript = ""
        text_budget = 0

        if not intent_only:
            txt_col = text_column or "transcript"
            raw_transcript = _stringify(batch.get(txt_col))
            transcript = raw_transcript

            if align_transcript_to_audio_chunk and transcript:
                duration_seconds = batch.get("audio_duration_seconds")
                try:
                    duration_seconds = float(duration_seconds)
                except (TypeError, ValueError):
                    duration_seconds = None

                if duration_seconds is not None and duration_seconds > audio_chunk_seconds:
                    ratio = max(0.05, min(1.0, audio_chunk_seconds / duration_seconds))
                    approx_chars = max(1, int(len(transcript) * ratio))
                    transcript = transcript[:approx_chars]

            # Reserve explicit budget for the trailing Intent field so that
            # the transcript body does not crowd out the main label.
            envelope_obj = _build_json_obj(
                transcript="",
                intent=intent,
                domain=domain,
            )
            envelope_ids = _encode_text(json.dumps(envelope_obj, ensure_ascii=False))
            text_budget = max(0, max_label_tokens - len(envelope_ids) - 1 - 3)
            if max_text_target_tokens is not None:
                text_budget = min(text_budget, max_text_target_tokens)

            if transcript:
                text_ids = _encode_text(transcript)
                if len(text_ids) > text_budget:
                    transcript = tokenizer.decode(
                        text_ids[:text_budget],
                        skip_special_tokens=True,
                    )

        json_obj = _build_json_obj(
            transcript=transcript,
            intent=intent,
            domain=domain,
        )
        json_str = json.dumps(json_obj, ensure_ascii=False)
        ids = _encode_text(json_str) + [tokenizer.eos_token_id]
        loss_weights = [1.0] * len(ids)

        # Final safety pass: if BPE boundary effects still pushed the sequence
        # over budget, tighten the transcript budget rather than truncating the
        # JSON envelope blindly.
        if len(ids) > max_label_tokens and not intent_only and raw_transcript:
            excess = len(ids) - max_label_tokens
            tighter_budget = max(0, text_budget - excess - 2)
            tighter_ids = _encode_text(raw_transcript)[:tighter_budget]
            transcript = tokenizer.decode(
                tighter_ids,
                skip_special_tokens=True,
            )
            json_obj = _build_json_obj(
                transcript=transcript,
                intent=intent,
                domain=domain,
            )
            json_str = json.dumps(json_obj, ensure_ascii=False)
            ids = _encode_text(json_str) + [tokenizer.eos_token_id]
            loss_weights = [1.0] * len(ids)

        # Hard guardrail for pathological cases.
        if len(ids) > max_label_tokens:
            ids = ids[:max_label_tokens]
            ids[-1] = tokenizer.eos_token_id
            loss_weights = loss_weights[:max_label_tokens]
            if loss_weights:
                loss_weights[-1] = max(loss_weights[-1], 1.0)

        if intent_loss_weight > 1.0:
            intent_prefix_obj = _build_json_obj(
                transcript="",
                intent=intent,
                domain=domain,
            )
            intent_prefix_text = json.dumps(intent_prefix_obj, ensure_ascii=False)
            if not intent_only and '"Text": ""' in intent_prefix_text:
                intent_prefix_text = intent_prefix_text.replace('"Text": ""', '"Text": "')
            intent_prefix_len = len(_encode_text(intent_prefix_text))
            for idx in range(min(intent_prefix_len, len(loss_weights))):
                loss_weights[idx] = intent_loss_weight

        batch["labels"] = ids
        batch["loss_weights"] = loss_weights
        return batch

    return _map_batch


def load_and_prepare_datasets(
    dataset_path: Path,
    processor: WhisperProcessor,
    config: Mapping[str, Any],
    train_split: str = "train",
    eval_split: str = "val",
    label_column: str = "label",
    text_column: Optional[str] = None,
    num_proc: Optional[int] = None,
) -> Tuple[Any, Any]:
    """
    Load a TeleAntiFraud-28k dataset and prepare it for Whisper training.

    Supported inputs:
    1) A Hugging Face DatasetDict saved with `save_to_disk`.
    2) A folder containing CSV manifests such as:
       `train_manifest.csv`, `val_manifest.csv`, `test_manifest.csv`
       with columns `path,label,transcript`.
    3) A single CSV manifest path, which is treated as both train and eval.
    """
    sampling_rate = int(config.get("sampling_rate", 16000))

    def _build_dataset_from_manifest(manifest: Path, root_dir: Path) -> Dataset:
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

            if eval_split not in dataset_dict:
                if eval_split == "val" and "validation" in dataset_dict:
                    eval_split = "validation"
                elif eval_split == "validation" and "val" in dataset_dict:
                    eval_split = "val"
                else:
                    raise KeyError(
                        f"Split '{eval_split}' not found in dataset at {dataset_path}. "
                        f"Available: {list(dataset_dict.keys())}"
                    )
        except Exception:
            train_manifest = _resolve_manifest_path(dataset_path, train_split)
            eval_manifest = _resolve_manifest_path(dataset_path, eval_split)

            # Manifest paths are stored like
            # `TeleAntiFraud-28k/merged_result/...`, while the actual local
            # audio lives under `<repo>/TeleAntiFraud-28k/TeleAntiFraud-28k/...`.
            # `_resolve_path()` above tries both `root_dir / path` and
            # `root_dir.parent / path` so this also works when dataset_path is
            # `TeleAntiFraud-28k/corrected_manifests` or `hard_manifests`.
            root_dir = dataset_path
            train_ds = _build_dataset_from_manifest(train_manifest, root_dir=root_dir)
            eval_ds = _build_dataset_from_manifest(eval_manifest, root_dir=root_dir)
            dataset_dict = DatasetDict({train_split: train_ds, eval_split: eval_ds})
    else:
        if dataset_path.suffix.lower() == ".csv":
            root_dir = dataset_path.parent
            train_ds = _build_dataset_from_manifest(dataset_path, root_dir=root_dir)
            dataset_dict = DatasetDict({train_split: train_ds, eval_split: train_ds})
        else:
            dataset_dict = load_from_disk(str(dataset_path))
            dataset_dict = dataset_dict.cast_column("audio", Audio(sampling_rate=sampling_rate))

    mapping_fn = prepare_dataset_mapping_fn(
        processor=processor,
        config=config,
        label_column=label_column,
        text_column=text_column,
    )

    processed = dataset_dict.map(
        mapping_fn,
        remove_columns=[
            col
            for col in dataset_dict[train_split].column_names
            if col not in ("audio", "label", "labels", "loss_weights", "audio_duration_seconds")
        ],
        num_proc=num_proc,
    )

    train_dataset = processed[train_split]
    eval_dataset = processed[eval_split]
    return train_dataset, eval_dataset


def load_audio_for_inference(
    audio_path: Path,
    sampling_rate: int,
) -> np.ndarray:
    """
    Load raw audio from disk for inference-time processing.
    """
    audio, sr = sf.read(str(audio_path))

    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)

    if sr != sampling_rate:
        duration = audio.shape[0] / sr
        target_len = int(duration * sampling_rate)
        x_old = np.linspace(0, 1, num=audio.shape[0])
        x_new = np.linspace(0, 1, num=target_len)
        audio = np.interp(x_new, x_old, audio).astype(np.float32)

    return audio.astype(np.float32)


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    """
    Data collator that:
    - Pads log-Mel input features to the max length in the batch.
    - Pads labels and masks padding positions with -100.
    """

    processor: WhisperProcessor

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        fe = self.processor.feature_extractor

        input_feats_list: List[Dict[str, Any]] = []
        label_features: List[Dict[str, Any]] = []
        weight_features: List[Dict[str, Any]] = []

        for feature in features:
            if "labels" not in feature:
                continue

            try:
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
                label_features.append({"input_ids": feature["labels"]})
                weight_features.append({"input_ids": feature.get("loss_weights", [1.0] * len(feature["labels"]))})
            except Exception:
                continue

        if not input_feats_list:
            sampling_rate = getattr(fe, "sampling_rate", 16000)
            dummy_audio = np.zeros(int(sampling_rate * 0.1), dtype=np.float32)
            inputs = fe(
                dummy_audio,
                sampling_rate=sampling_rate,
                return_attention_mask=False,
            )
            input_feats_list.append({"input_features": inputs["input_features"][0]})
            label_features.append({"input_ids": features[0]["labels"]})
            weight_features.append({"input_ids": features[0].get("loss_weights", [1.0] * len(features[0]["labels"]))})

        batch = fe.pad(
            input_feats_list,
            padding=True,
            return_tensors="pt",
        )

        labels_batch = self.processor.tokenizer.pad(
            label_features,
            padding=True,
            return_tensors="pt",
        )
        weights_batch = self.processor.tokenizer.pad(
            weight_features,
            padding=True,
            return_tensors="pt",
        )

        labels = labels_batch["input_ids"]
        labels = labels.masked_fill(labels_batch.attention_mask.ne(1), -100)
        batch["labels"] = labels
        loss_weights = weights_batch["input_ids"].to(batch["input_features"].dtype)
        loss_weights = loss_weights.masked_fill(labels_batch.attention_mask.ne(1), 0.0)
        batch["loss_weights"] = loss_weights
        return batch
