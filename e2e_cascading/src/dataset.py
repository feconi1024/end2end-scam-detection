from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import math
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import Dataset
import torchaudio
import yaml


ConfigDict = Dict[str, Any]


def load_config(config_path: Union[str, Path]) -> ConfigDict:
    """
    Load a YAML configuration file into a nested dict.
    """
    path = Path(config_path)
    with path.open("r", encoding="utf-8") as f:
        cfg: ConfigDict = yaml.safe_load(f)
    return cfg


@dataclass
class TeleAntiFraudExample:
    audio_path: Path
    label_str: str
    transcript: Optional[str]


@dataclass
class AudioPreprocessResult:
    audio: Tensor
    original_num_samples: int


def load_audio_tensor(audio_path: Union[str, Path], sample_rate: int) -> Tensor:
    """
    Load audio as mono float32 at the target sample rate.
    """
    path = Path(audio_path)
    waveform, sr = torchaudio.load(path.as_posix())  # (channels, time)
    if waveform.dim() == 2 and waveform.size(0) > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sr != sample_rate:
        waveform = torchaudio.functional.resample(
            waveform,
            sr,
            sample_rate,
        )
    return waveform.squeeze(0).float()


def prepare_audio_tensor(
    audio_path: Union[str, Path],
    sample_rate: int,
    split: str = "train",
    fixed_duration_seconds: float = 15.0,
    train_noise_max_amp: float = 0.005,
) -> AudioPreprocessResult:
    """
    Load audio and apply the same preprocessing used by dataset training/eval.

    Every example is normalized to the same fixed duration so downstream models
    do not receive raw duration as an easy shortcut feature.
    """
    path = Path(audio_path)
    split = str(split).lower()
    target_length = int(sample_rate * fixed_duration_seconds)

    try:
        audio_tensor = load_audio_tensor(path, sample_rate=sample_rate)
    except Exception as e_torch:
        # Substitute 1 second of silence so that a handful of unreadable
        # files do not abort training or evaluation.
        print(
            "[TeleAntiFraudDataset] Warning: failed to load audio with torchaudio, "
            f"substituting silence. path={path.as_posix()}, error={repr(e_torch)}"
        )
        audio_tensor = torch.zeros(sample_rate, dtype=torch.float32)

    original_num_samples = int(audio_tensor.numel())

    if audio_tensor.size(0) > target_length:
        if split == "train":
            max_start = audio_tensor.size(0) - target_length
            start = int(torch.randint(0, max_start + 1, (1,)).item())
        else:
            start = 0
        audio_tensor = audio_tensor[start : start + target_length]
    else:
        pad_amount = target_length - audio_tensor.size(0)
        if pad_amount > 0:
            audio_tensor = F.pad(audio_tensor, (0, pad_amount))

    if split == "train" and train_noise_max_amp > 0.0:
        noise_amp = float(train_noise_max_amp) * torch.rand(1).item()
        audio_tensor = audio_tensor + noise_amp * torch.randn_like(audio_tensor)

    return AudioPreprocessResult(
        audio=audio_tensor,
        original_num_samples=original_num_samples,
    )


class TeleAntiFraudDataset(Dataset):
    """
    Dataset for TeleAntiFraud-28k-style manifests.

    Expects a CSV manifest with at least:
      - path: absolute or relative path to the audio file
      - label: string label (e.g. "scam", "non_scam")
    Optionally:
      - transcript: ground-truth text transcript (for CTC)
    """

    def __init__(
        self,
        manifest_path: Union[str, Path],
        tokenizer,
        sample_rate: int,
        split: str = "train",
        label_mapping: Optional[Dict[str, int]] = None,
        fixed_duration_seconds: float = 15.0,
        train_noise_max_amp: float = 0.005,
    ) -> None:
        import csv

        self.manifest_path = Path(manifest_path)
        self.sample_rate = int(sample_rate)
        self.split = str(split).lower()
        self.tokenizer = tokenizer
        self.fixed_duration_seconds = float(fixed_duration_seconds)
        self.train_noise_max_amp = float(train_noise_max_amp)

        self.examples: List[TeleAntiFraudExample] = []
        with self.manifest_path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                path_str = row.get("path") or row.get("audio_path") or row.get("wav_path")
                if not path_str:
                    raise ValueError(
                        "Manifest row missing audio path. Expected one of: 'path', 'audio_path', 'wav_path'."
                    )
                label_str = row.get("label")
                if label_str is None:
                    raise ValueError("Manifest row missing 'label' column.")
                transcript = row.get("transcript") or None
                audio_path = Path(path_str)
                if not audio_path.is_absolute():
                    audio_path = (self.manifest_path.parent / audio_path).resolve()

                if not audio_path.exists():
                    # Skip missing files but warn once per missing path.
                    print(
                        f"[TeleAntiFraudDataset] Warning: audio file does not exist, skipping: "
                        f"{audio_path.as_posix()}"
                    )
                    continue

                self.examples.append(
                    TeleAntiFraudExample(
                        audio_path=audio_path,
                        label_str=label_str,
                        transcript=transcript,
                    )
                )

        # Build label mapping if not provided
        if label_mapping is None:
            unique_labels = sorted({ex.label_str for ex in self.examples})
            self.label2id: Dict[str, int] = {lbl: i for i, lbl in enumerate(unique_labels)}
        else:
            self.label2id = dict(label_mapping)
        self.id2label: Dict[int, str] = {v: k for k, v in self.label2id.items()}

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        ex = self.examples[idx]
        audio_result = prepare_audio_tensor(
            audio_path=ex.audio_path,
            sample_rate=self.sample_rate,
            split=self.split,
            fixed_duration_seconds=self.fixed_duration_seconds,
            train_noise_max_amp=self.train_noise_max_amp,
        )
        audio_tensor = audio_result.audio

        label_id = self.label2id[ex.label_str]

        token_ids: Optional[Tensor] = None
        if ex.transcript is not None and ex.transcript.strip():
            encoded = self.tokenizer(
                ex.transcript,
                add_special_tokens=False,
                return_attention_mask=False,
                return_token_type_ids=False,
                truncation=True,
                max_length=448,
            )
            token_ids = torch.tensor(encoded["input_ids"], dtype=torch.long)

        return {
            "audio": audio_tensor,  # 1D waveform
            "label": label_id,
            "transcript_ids": token_ids,  # 1D token sequence or None
            "original_num_samples": audio_result.original_num_samples,
        }


class WhisperCollator:
    """
    Picklable collator for batching waveforms into Whisper-compatible mel features
    and CTC targets. Use this so DataLoader with num_workers > 0 works on both
    Windows (spawn) and Linux/Slurm (fork or spawn).
    """

    def __init__(self, processor, pad_token_id: int, sample_rate: int) -> None:
        self.processor = processor
        self.pad_token_id = pad_token_id
        self.sample_rate = sample_rate
        self.hop_length = int(
            getattr(getattr(processor, "feature_extractor", None), "hop_length", 160)
        )

    def __call__(self, batch: Sequence[Dict[str, Any]]) -> Dict[str, Tensor]:
        # Waveforms (list of 1D tensors) -> list of 1D numpy arrays
        audios_torch: List[Tensor] = [b["audio"] for b in batch]
        audios = [a.cpu().numpy() for a in audios_torch]
        labels = torch.tensor([b["label"] for b in batch], dtype=torch.long)
        audio_durations = torch.tensor(
            [
                float(b.get("original_num_samples", a.numel())) / float(self.sample_rate)
                for a, b in zip(audios_torch, batch)
            ],
            dtype=torch.float32,
        )

        # Use the full WhisperProcessor so that input_features are padded/
        # truncated to the fixed length expected by the Whisper encoder
        # (e.g., 3000 frames for 30s of audio).
        processed = self.processor(
            audios,
            sampling_rate=self.sample_rate,
            return_tensors="pt",
        )
        input_features: Tensor = processed.input_features  # (B, n_mels, T_acoustic)
        bsz, _, t_acoustic = input_features.shape
        # Build a correct attention mask so padding isn't treated as speech.
        # Whisper uses hop_length=160 at 16kHz, i.e. 10ms per frame.
        # valid_frames ≈ ceil(num_samples / hop_length), clipped to T_acoustic.
        audio_attention_mask = torch.zeros((bsz, t_acoustic), dtype=torch.long)
        for i, wav in enumerate(audios_torch):
            valid_frames = int(math.ceil(wav.numel() / float(self.hop_length)))
            valid_frames = max(0, min(t_acoustic, valid_frames))
            if valid_frames > 0:
                audio_attention_mask[i, :valid_frames] = 1

        # CTC targets from tokenized transcripts
        token_seqs: List[Tensor] = [
            b["transcript_ids"] for b in batch if b["transcript_ids"] is not None
        ]
        if len(token_seqs) == len(batch):
            lengths = torch.tensor([t.size(0) for t in token_seqs], dtype=torch.long)
            max_len = int(lengths.max().item()) if lengths.numel() > 0 else 0
            if max_len > 0:
                padded = torch.full(
                    (len(batch), max_len),
                    fill_value=self.pad_token_id,
                    dtype=torch.long,
                )
                for i, t in enumerate(token_seqs):
                    padded[i, : t.size(0)] = t
            else:
                padded = torch.empty((len(batch), 0), dtype=torch.long)
            ctc_targets = padded
            ctc_target_lengths = lengths
        else:
            # At least one example is missing a transcript; disable CTC for this batch.
            ctc_targets = torch.empty((len(batch), 0), dtype=torch.long)
            ctc_target_lengths = torch.zeros(len(batch), dtype=torch.long)

        return {
            "input_features": input_features,
            "audio_attention_mask": audio_attention_mask,
            "labels": labels,
            "ctc_targets": ctc_targets,
            "ctc_target_lengths": ctc_target_lengths,
            "audio_durations": audio_durations,
        }


def create_collate_fn(
    processor,
    pad_token_id: int,
    sample_rate: int,
) -> Callable[[Sequence[Dict[str, Any]]], Dict[str, Tensor]]:
    """
    Returns a picklable collator (WhisperCollator) so that DataLoader with
    num_workers > 0 works on Linux and Slurm. On Windows, num_workers is
    typically set to 0 in the training script to avoid spawn/pickle issues.
    """
    return WhisperCollator(
        processor=processor,
        pad_token_id=pad_token_id,
        sample_rate=sample_rate,
    )

