from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import librosa
import torch
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
        label_mapping: Optional[Dict[str, int]] = None,
    ) -> None:
        import csv

        self.manifest_path = Path(manifest_path)
        self.sample_rate = int(sample_rate)
        self.tokenizer = tokenizer

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

        # Load audio as float32 mono at the target sample rate.
        # Prefer torchaudio (ships its own codecs, robust for mp3), fall back
        # to librosa if torchaudio fails for any reason. If both backends
        # fail, replace with a short silence segment to avoid crashing
        # training on a few problematic files.
        path_str = ex.audio_path.as_posix()
        try:
            waveform, sr = torchaudio.load(path_str)  # (channels, time)
            if waveform.dim() == 2 and waveform.size(0) > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
            if sr != self.sample_rate:
                waveform = torchaudio.functional.resample(
                    waveform, sr, self.sample_rate
                )
            audio_tensor = waveform.squeeze(0).float()
        except Exception as e_torch:
            try:
                # Fallback to librosa (may rely on system codecs for mp3).
                audio, sr = librosa.load(path_str, sr=self.sample_rate, mono=True)
                audio_tensor = torch.from_numpy(audio).float()
            except Exception as e_librosa:
                # As a last resort, substitute 1 second of silence so that
                # a handful of unreadable files do not abort training.
                print(
                    "[TeleAntiFraudDataset] Warning: failed to load audio with both "
                    f"torchaudio and librosa, substituting silence. path={path_str}, "
                    f"torchaudio_error={repr(e_torch)}, librosa_error={repr(e_librosa)}"
                )
                audio_tensor = torch.zeros(self.sample_rate, dtype=torch.float32)

        label_id = self.label2id[ex.label_str]

        token_ids: Optional[Tensor] = None
        if ex.transcript is not None and ex.transcript.strip():
            encoded = self.tokenizer(
                ex.transcript,
                add_special_tokens=False,
                return_attention_mask=False,
                return_token_type_ids=False,
            )
            token_ids = torch.tensor(encoded["input_ids"], dtype=torch.long)

        return {
            "audio": audio_tensor,  # 1D waveform
            "label": label_id,
            "transcript_ids": token_ids,  # 1D token sequence or None
        }


def create_collate_fn(
    processor,
    pad_token_id: int,
    sample_rate: int,
) -> Callable[[Sequence[Dict[str, Any]]], Dict[str, Tensor]]:
    """
    Collate function that:
      - Pads variable-length waveforms and converts them to Whisper log-Mel features.
      - Pads variable-length token sequences for CTC targets.

    Returns a dict with:
      - input_features: (B, T_acoustic, n_mels)
      - audio_attention_mask: (B, T_acoustic)
      - labels: (B,)
      - ctc_targets: (B, T_text) or None
      - ctc_target_lengths: (B,) or None
    """

    def collate(batch: Sequence[Dict[str, Any]]) -> Dict[str, Tensor]:
        # Waveforms (list of 1D tensors) -> list of 1D numpy arrays
        audios_torch: List[Tensor] = [b["audio"] for b in batch]
        audios = [a.cpu().numpy() for a in audios_torch]
        labels = torch.tensor([b["label"] for b in batch], dtype=torch.long)

        # Use the full WhisperProcessor so that input_features are padded/
        # truncated to the fixed length expected by the Whisper encoder
        # (e.g., 3000 frames for 30s of audio).
        processed = processor(
            audios,
            sampling_rate=sample_rate,
            return_tensors="pt",
        )
        input_features: Tensor = processed.input_features  # (B, n_mels, T_acoustic)
        # WhisperProcessor already handles padding/truncation; we treat all
        # frames as valid here.
        bsz, _, t_acoustic = input_features.shape
        audio_attention_mask: Tensor = torch.ones(
            (bsz, t_acoustic), dtype=torch.long
        )

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
                    fill_value=pad_token_id,
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
        }

    return collate

