from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np
import soundfile as sf
import yaml
from datasets import Audio, DatasetDict, load_from_disk
from transformers import WhisperFeatureExtractor, WhisperProcessor, WhisperTokenizer


def load_config(config_path: Path) -> Dict[str, Any]:
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def create_processor(config: Mapping[str, Any]) -> Tuple[WhisperProcessor, int]:
    """
    Create WhisperProcessor and extend tokenizer with fraud special tokens.

    Returns (processor, num_added_tokens).
    """
    model_name = config["model_name"]
    special_tokens: Sequence[str] = config.get("special_tokens", [])

    feature_extractor = WhisperFeatureExtractor.from_pretrained(model_name)
    tokenizer = WhisperTokenizer.from_pretrained(
        model_name,
        language="en",
        task="transcribe",
    )

    num_added = tokenizer.add_tokens(list(special_tokens))

    # Ensure we have a pad token; Whisper commonly reuses eos as pad.
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    processor = WhisperProcessor(feature_extractor=feature_extractor, tokenizer=tokenizer)
    return processor, num_added


def format_sml_sequence(
    intent_tokens: Sequence[str],
    tokenizer: WhisperTokenizer,
) -> str:
    """
    Format the multi-task target sequence as:
    <|startoftranscript|><|scam|><|impersonation|> <|endoftext|>
    """
    bos = tokenizer.bos_token or "<|startoftranscript|>"
    eos = tokenizer.eos_token or "<|endoftext|>"

    tag_str = "".join(intent_tokens) if intent_tokens else ""
    # Space before eos to separate from last tag.
    sequence = f"{bos}{tag_str} {eos}"
    return sequence


def labels_to_tokens(
    raw_label: Any,
    label2token: Mapping[str, str],
) -> List[str]:
    """
    Map dataset label(s) to the corresponding fraud intent special tokens.
    Supports single label or list of labels.
    """
    if raw_label is None:
        return []

    if isinstance(raw_label, str):
        labels = [raw_label]
    elif isinstance(raw_label, (list, tuple)):
        labels = list(raw_label)
    else:
        labels = [str(raw_label)]

    tokens: List[str] = []
    for lbl in labels:
        if lbl in label2token:
            tokens.append(label2token[lbl])
    return tokens


def prepare_dataset_mapping_fn(
    processor: WhisperProcessor,
    config: Mapping[str, Any],
    label_column: str = "label",
    text_column: Optional[str] = None,
) -> Any:
    """
    Build a mapping function to prepare TeleAntiFraud-style examples:
    - audio: Audio (array, sampling_rate)
    - transcript (optional): not directly used for targets here, because we enforce
      SML format intent-first.
    - label: label or list of labels, mapped via config.label2token.
    """

    sampling_rate: int = int(config.get("sampling_rate", 16000))
    label2token: Mapping[str, str] = config.get("label2token", {})

    feature_extractor = processor.feature_extractor
    tokenizer = processor.tokenizer

    def _map_batch(batch: MutableMapping[str, Any]) -> MutableMapping[str, Any]:
        audio = batch["audio"]

        # audio is expected in datasets.Audio format
        array = audio["array"]
        sr = audio["sampling_rate"]

        if sr != sampling_rate:
            # librosa-style resampling without importing librosa in the core path
            # to keep dependencies light in this module.
            # Users can pre-resample the dataset; here we fall back to simple
            # numpy interpolation as a generic solution.
            duration = array.shape[0] / sr
            target_len = int(duration * sampling_rate)
            x_old = np.linspace(0, 1, num=array.shape[0])
            x_new = np.linspace(0, 1, num=target_len)
            array = np.interp(x_new, x_old, array).astype(np.float32)

        inputs = feature_extractor(
            array,
            sampling_rate=sampling_rate,
            return_attention_mask=False,
        )

        # Build intent token sequence
        raw_label = batch.get(label_column)
        intent_tokens = labels_to_tokens(raw_label, label2token=label2token)
        target_text = format_sml_sequence(intent_tokens=intent_tokens, tokenizer=tokenizer)

        tokenized = tokenizer(
            target_text,
            add_special_tokens=False,
        )

        batch["input_features"] = inputs["input_features"][0]
        batch["labels"] = tokenized["input_ids"]
        return batch

    return _map_batch


def load_and_prepare_datasets(
    dataset_path: Path,
    processor: WhisperProcessor,
    config: Mapping[str, Any],
    train_split: str = "train",
    eval_split: str = "validation",
    label_column: str = "label",
    text_column: Optional[str] = None,
    num_proc: Optional[int] = None,
) -> Tuple[Any, Any]:
    """
    Load a TeleAntiFraud-28k dataset from disk and prepare it for Whisper training.
    """
    dataset_dict: DatasetDict = load_from_disk(str(dataset_path))

    sampling_rate: int = int(config.get("sampling_rate", 16000))
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
            col for col in dataset_dict[train_split].column_names if col not in ("input_features", "labels")
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
    - Pads log-Mel input features to max length in the batch.
    - Pads labels and replaces padding positions with -100 for CE loss masking.
    """

    processor: WhisperProcessor

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        input_features = [{"input_features": np.asarray(f["input_features"])} for f in features]
        label_features = [{"input_ids": f["labels"]} for f in features]

        batch = self.processor.feature_extractor.pad(
            input_features,
            padding=True,
            return_tensors="pt",
        )

        labels_batch = self.processor.tokenizer.pad(
            label_features,
            padding=True,
            return_tensors="pt",
        )

        labels = labels_batch["input_ids"]
        # Replace padding token id's with -100 so they are ignored by loss
        labels[labels == self.processor.tokenizer.pad_token_id] = -100
        batch["labels"] = labels
        return batch

