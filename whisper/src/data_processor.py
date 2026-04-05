from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

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
    Create WhisperProcessor for Whisper-based SLU.

    Returns (processor, num_added_tokens).
    """
    model_name = config["model_name"]

    processor = WhisperProcessor.from_pretrained(
        model_name,
        language="en",
        task="transcribe",
    )

    # Ensure we have a pad token; Whisper commonly reuses eos as pad.
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    # No additional special tokens are added when using JSON-formatted targets.
    return processor, 0


def prepare_dataset_mapping_fn(
    processor: WhisperProcessor,
    config: Mapping[str, Any],
    label_column: str = "label",
    text_column: Optional[str] = None,
) -> Any:
    """
    Build a mapping function to prepare TeleAntiFraud-style examples with
    JSON-formatted multitask targets.

    Two modes:
    - Full JSON (default):
      {
        "Intent": "scam",
        "Domain": "telephony",
        "Text": "<transcript>"
      }
    - Intent-only JSON (if config['train_intent_only'] is true):
      {
        "Intent": "scam"
      }

    The training sequence is:
      <|startoftranscript|>{...}<|endoftext|>
    """

    tokenizer = processor.tokenizer
    default_domain = str(config.get("domain", "telephony"))
    intent_only: bool = bool(config.get("train_intent_only", False))

    # Whisper decoder hard maximum.
    MAX_LABEL_TOKENS = 448

    # Pre-compute the token overhead of the JSON envelope (everything except
    # the transcript body).  We tokenize once with an empty "Text" field so
    # that we know exactly how many tokens are available for the transcript.
    if not intent_only:
        _envelope = json.dumps(
            {"Intent": "", "Domain": default_domain, "Text": ""},
            ensure_ascii=False,
        )
        _envelope_ids = tokenizer(_envelope, add_special_tokens=False)["input_ids"]
        # +1 for the EOS token we append; +3 safety margin for BPE boundary
        # effects when the transcript is re-encoded inside the JSON context.
        _envelope_overhead = len(_envelope_ids) + 1 + 3

    def _map_batch(batch: MutableMapping[str, Any]) -> MutableMapping[str, Any]:
        # Ground truth label (single-label classification)
        raw_label = batch.get(label_column)
        if isinstance(raw_label, (list, tuple)):
            raw_label = raw_label[0] if raw_label else ""
        intent = str(raw_label) if raw_label is not None else ""

        if intent_only:
            json_obj = {"Intent": intent}
        else:
            txt_col = text_column or "transcript"
            transcript = batch.get(txt_col) or ""

            # --- Compress transcript to fit within the 448-token budget ---
            # Tokenize the transcript in isolation and truncate to the
            # available budget, then decode back to text.  This guarantees
            # the final JSON is always structurally complete (closing `"}`
            # are never cut off) while preserving as much semantic content
            # as the token budget allows.
            text_budget = max(0, MAX_LABEL_TOKENS - _envelope_overhead)
            if transcript:
                text_ids = tokenizer(
                    transcript, add_special_tokens=False
                )["input_ids"]
                if len(text_ids) > text_budget:
                    text_ids = text_ids[:text_budget]
                    transcript = tokenizer.decode(
                        text_ids, skip_special_tokens=True
                    )

            json_obj = {
                "Intent": intent,
                "Domain": default_domain,
                "Text": transcript,
            }

        json_str = json.dumps(json_obj, ensure_ascii=False)

        # Tokenize the JSON text ONLY — do NOT embed special token strings
        # (like <|endoftext|>) in the text.  The Whisper fast tokenizer does
        # not recognise special-token strings inside input text when
        # add_special_tokens=False; it would BPE-encode "<|endoftext|>" as
        # regular characters instead of the single token ID 50257, so the
        # model would never learn to produce the real EOS and would loop
        # until max_length during generation.
        ids = tokenizer(json_str, add_special_tokens=False)["input_ids"]

        # Manually append the real EOS token ID.
        ids = ids + [tokenizer.eos_token_id]

        # Final safety: if BPE re-tokenization made the sequence slightly
        # longer than expected, trim transcript tokens from the end while
        # keeping the JSON envelope intact.  In practice the 3-token safety
        # margin above makes this a no-op.
        if len(ids) > MAX_LABEL_TOKENS:
            # The JSON closing '"}' is the last ~2 tokens before EOS.
            # Rebuild with a tighter budget rather than blindly truncating.
            if not intent_only and transcript:
                excess = len(ids) - MAX_LABEL_TOKENS
                tighter_budget = max(0, text_budget - excess - 2)
                text_ids = tokenizer(
                    batch.get(text_column or "transcript") or "",
                    add_special_tokens=False,
                )["input_ids"][:tighter_budget]
                trimmed = tokenizer.decode(text_ids, skip_special_tokens=True)
                json_obj["Text"] = trimmed
                json_str = json.dumps(json_obj, ensure_ascii=False)
                ids = tokenizer(json_str, add_special_tokens=False)["input_ids"]
                ids = ids + [tokenizer.eos_token_id]

        batch["labels"] = ids
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
    Load a TeleAntiFraud-28k dataset and prepare it for Whisper training.

    This supports two formats:
    1) A Hugging Face dataset directory created via `datasets.DatasetDict.save_to_disk`
       (loaded with `load_from_disk`).
    2) A plain folder containing CSV manifests (e.g. `train_manifest.csv`,
       `validation_manifest.csv`) with columns: `path,label,transcript`.
    """
    sampling_rate: int = int(config.get("sampling_rate", 16000))

    def _build_dataset_from_manifest(manifest: Path, root_dir: Path) -> Dataset:
        """
        Build a datasets.Dataset from a CSV manifest.
        Expected columns: `path,label,transcript`.
        `path` is interpreted relative to `root_dir` if not absolute.
        """
        df = pd.read_csv(manifest)

        if "path" not in df.columns or "label" not in df.columns:
            raise ValueError(f"Manifest {manifest} must contain at least 'path' and 'label' columns.")

        def _resolve_path(p: str) -> str:
            # Always return an absolute path so that audio loading works
            # correctly even in multiprocessing workers where the CWD may differ.
            p_path = Path(p)
            if not p_path.is_absolute():
                p_path = (root_dir / p).resolve()
            else:
                p_path = p_path.resolve()
            return str(p_path)

        df["audio"] = df["path"].apply(_resolve_path)

        # Ensure transcript column exists even if empty
        if "transcript" not in df.columns:
            df["transcript"] = ""

        ds = Dataset.from_pandas(
            df[["audio", "label", "transcript"]],
            preserve_index=False,
        )
        ds = ds.cast_column("audio", Audio(sampling_rate=sampling_rate))
        return ds

    dataset_dict: DatasetDict

    if dataset_path.is_dir():
        # First try: treat as a saved Hugging Face dataset directory
        try:
            dataset_dict = load_from_disk(str(dataset_path))
            dataset_dict = dataset_dict.cast_column("audio", Audio(sampling_rate=sampling_rate))
        except Exception:
            # Fallback: treat as TeleAntiFraud-style folder with CSV manifests
            train_manifest = dataset_path / f"{train_split}_manifest.csv"
            eval_manifest = dataset_path / f"{eval_split}_manifest.csv"

            if not train_manifest.exists():
                raise FileNotFoundError(
                    f"Expected training manifest not found: {train_manifest}. "
                    f"Provide a dataset directory created via `save_to_disk` or a folder "
                    f"containing '{train_split}_manifest.csv'."
                )

            # If a dedicated eval manifest is missing, try a common alternative or
            # fall back to using the training manifest for both.
            if not eval_manifest.exists():
                alt_eval = dataset_path / "validation_manifest.csv"
                if alt_eval.exists():
                    eval_manifest = alt_eval
                else:
                    eval_manifest = train_manifest

            # When dataset_path is a directory like `<root>/TeleAntiFraud-28k`
            # and CSV paths are `TeleAntiFraud-28k/merged_result/...`,
            # we need to join against `dataset_path` itself so that
            # final audio paths become `<root>/TeleAntiFraud-28k/TeleAntiFraud-28k/...`.
            root_dir = dataset_path
            train_ds = _build_dataset_from_manifest(train_manifest, root_dir=root_dir)
            eval_ds = _build_dataset_from_manifest(eval_manifest, root_dir=root_dir)
            dataset_dict = DatasetDict({train_split: train_ds, eval_split: eval_ds})
    else:
        # If a single CSV manifest is given, treat it as both train and eval.
        if dataset_path.suffix.lower() == ".csv":
            root_dir = dataset_path.parent
            train_ds = _build_dataset_from_manifest(dataset_path, root_dir=root_dir)
            dataset_dict = DatasetDict({train_split: train_ds, eval_split: train_ds})
        else:
            # Fallback: assume this is a `save_to_disk` directory path
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
            # Keep audio + original label + generated labels.
            col for col in dataset_dict[train_split].column_names if col not in ("audio", "label", "labels")
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
        """
        Build batch from raw audio + label ids.
        Invalid audio examples are skipped at collator time so training
        is robust to occasional corrupt files.
        """
        fe = self.processor.feature_extractor

        input_feats_list: List[Dict[str, Any]] = []
        label_features: List[Dict[str, Any]] = []

        for f in features:
            # Skip examples without labels
            if "labels" not in f:
                continue

            try:
                audio = f["audio"]
                array = audio["array"]
                sr = audio["sampling_rate"]

                # Rely on feature extractor's expected sampling rate
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
                label_features.append({"input_ids": f["labels"]})
            except Exception:
                # Corrupt or unreadable audio: drop this example from batch
                continue

        if not input_feats_list:
            # Extremely unlikely, but as a safety net we synthesize a short
            # dummy waveform and run it through the feature extractor so that
            # the resulting mel features have the exact expected shape
            # (including Whisper's required length of 3000 frames).
            sampling_rate = getattr(fe, "sampling_rate", 16000)
            dummy_audio = np.zeros(int(sampling_rate * 0.1), dtype=np.float32)
            inputs = fe(
                dummy_audio,
                sampling_rate=sampling_rate,
                return_attention_mask=False,
            )
            input_feats_list.append({"input_features": inputs["input_features"][0]})
            # Reuse the first example's labels so loss is still well-defined.
            label_features.append({"input_ids": features[0]["labels"]})

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

        labels = labels_batch["input_ids"]
        # Mask padding positions with -100 so they are ignored by the loss.
        # IMPORTANT: use the attention mask, NOT token-ID comparison.
        # For Whisper pad_token_id == eos_token_id (both 50257), so comparing
        # by ID would also mask the legitimate EOS at the end of each label
        # sequence — the model would never learn to stop generating.
        labels = labels.masked_fill(labels_batch.attention_mask.ne(1), -100)
        batch["labels"] = labels
        return batch

