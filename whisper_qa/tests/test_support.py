from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Dict

import numpy as np
import soundfile as sf
from transformers import WhisperConfig, WhisperForConditionalGeneration


class ToyTokenizer:
    def __init__(self):
        self.pad_token_id = 0
        self.bos_token_id = 1
        self.eos_token_id = 2
        self.pad_token = "<pad>"
        self.eos_token = "<eos>"
        self.prefix_tokens = [1, 3, 4, 5]

    def set_prefix_tokens(self, language=None, task=None, predict_timestamps=None):
        self.prefix_tokens = [1, 3, 4, 5 if not predict_timestamps else 6]

    def _encode_chars(self, text: str):
        return [6 + (ord(ch) % 20) for ch in text][:12]

    def __call__(self, text: str, add_special_tokens: bool = False):
        ids = self._encode_chars(text)
        if add_special_tokens:
            ids = self.prefix_tokens + ids + [self.eos_token_id]
        return {"input_ids": ids}

    def decode(self, ids, skip_special_tokens: bool = False) -> str:
        special = {self.pad_token_id, self.bos_token_id, 3, 4, 5, 6, self.eos_token_id}
        chars = []
        for token_id in ids:
            token_value = int(token_id)
            if skip_special_tokens and token_value in special:
                continue
            if token_value in special:
                chars.append("")
            else:
                chars.append(chr(ord("a") + (token_value - 6) % 20))
        return "".join(chars)

    def batch_decode(self, batch_ids, skip_special_tokens: bool = False):
        return [self.decode(ids.tolist() if hasattr(ids, "tolist") else ids, skip_special_tokens=skip_special_tokens) for ids in batch_ids]


class ToyFeatureExtractor:
    def __init__(self, feature_length: int):
        self.sampling_rate = 16000
        self.feature_length = feature_length

    def __call__(self, audio, sampling_rate: int, return_attention_mask: bool = False):
        del sampling_rate, return_attention_mask
        mean_value = float(np.mean(audio)) if len(audio) else 0.0
        features = np.full((1, 80, self.feature_length), fill_value=mean_value, dtype=np.float32)
        return {"input_features": features}


class ToyProcessor:
    def __init__(self, feature_length: int = 8):
        self.tokenizer = ToyTokenizer()
        self.feature_extractor = ToyFeatureExtractor(feature_length=feature_length)

    def get_decoder_prompt_ids(self, task=None, language=None, no_timestamps=True):
        del task, language, no_timestamps
        return [(1, 3), (2, 4), (3, 5)]

    def batch_decode(self, batch_ids, skip_special_tokens: bool = False):
        return self.tokenizer.batch_decode(batch_ids, skip_special_tokens=skip_special_tokens)

    def save_pretrained(self, path: str):
        target = Path(path)
        target.mkdir(parents=True, exist_ok=True)
        with (target / "toy_processor.json").open("w", encoding="utf-8") as f:
            json.dump({"feature_length": self.feature_extractor.feature_length}, f)


def build_tiny_whisper_model() -> WhisperForConditionalGeneration:
    config = WhisperConfig(
        vocab_size=32,
        num_mel_bins=80,
        encoder_layers=1,
        decoder_layers=1,
        encoder_attention_heads=2,
        decoder_attention_heads=2,
        encoder_ffn_dim=16,
        decoder_ffn_dim=16,
        d_model=8,
        max_source_positions=4,
        max_target_positions=64,
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=0,
        decoder_start_token_id=1,
        begin_suppress_tokens=[],
        suppress_tokens=[],
    )
    return WhisperForConditionalGeneration(config)


def build_test_config(tmp_path: Path) -> Dict[str, Any]:
    return {
        "model": {
            "name": "toy-whisper",
            "language": "zh",
            "task": "transcribe",
            "sampling_rate": 16000,
            "audio_chunk_seconds": 30.0,
            "transcript_max_label_tokens": 16,
            "transcript_max_cache_tokens": 8,
            "align_transcript_to_audio_chunk": True,
            "transcript_generation_max_new_tokens": 4,
            "no_timestamps": True,
        },
        "data": {
            "dataset_path": str(tmp_path / "TeleAntiFraud-28k" / "corrected_manifests"),
            "train_split": "train",
            "eval_split": "val",
            "label_column": "label",
            "text_column": "transcript",
            "num_workers": 0,
        },
        "questions": {
            "bank_path": "whisper_qa/config/questions_zh.yaml",
        },
        "tuning": {
            "mode": "prefix",
            "prefix": {
                "encoder_prefix_length": 2,
                "decoder_prefix_length": 2,
                "init_std": 0.02,
            },
        },
        "training": {
            "output_dir": str(tmp_path / "outputs"),
            "per_device_train_batch_size": 1,
            "per_device_eval_batch_size": 1,
            "gradient_accumulation_steps": 1,
            "num_train_epochs": 1,
            "learning_rate": 1.0e-3,
            "weight_decay": 0.0,
            "warmup_steps": 0,
            "max_grad_norm": 1.0,
            "log_every_steps": 1,
            "eval_each_epoch": False,
            "save_each_epoch": True,
            "asr_loss_weight": 0.5,
            "qa_loss_weight": 1.0,
            "gradient_checkpointing": False,
            "bf16": False,
            "fp16": False,
            "seed": 7,
            "device": "cpu",
        },
        "evaluation": {
            "output_dir": str(tmp_path / "outputs" / "eval"),
            "max_eval_samples": 2,
        },
        "inference": {
            "transcript_source": "toy_whisper_asr_greedy",
        },
    }


def write_tiny_wav(path: Path, seconds: float = 0.05) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sr = 16000
    audio = np.zeros(int(sr * seconds), dtype=np.float32)
    sf.write(str(path), audio, sr)

