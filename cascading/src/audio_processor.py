from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import librosa
import numpy as np

from .config_loader import AudioConfig


class AudioLoadingError(Exception):
    """Raised when audio cannot be loaded or validated."""


@dataclass
class AudioProcessor:
    config: AudioConfig

    def _validate_path(self, path: os.PathLike | str) -> Path:
        p = Path(path)
        if not p.exists():
            raise AudioLoadingError(f"Audio file does not exist: {p}")
        if not p.is_file():
            raise AudioLoadingError(f"Path is not a file: {p}")
        if p.stat().st_size == 0:
            raise AudioLoadingError(f"Audio file is empty: {p}")

        if not any(str(p).lower().endswith(ext.lower()) for ext in self.config.supported_extensions):
            exts = ", ".join(self.config.supported_extensions)
            raise AudioLoadingError(f"Unsupported audio extension for {p}. Supported: {exts}")
        return p

    def _normalize_to_minus_3db(self, audio: np.ndarray) -> np.ndarray:
        """
        Normalize the audio signal to approximately -3 dBFS peak.
        """
        if audio.size == 0:
            raise AudioLoadingError("Loaded audio array is empty.")

        peak = np.max(np.abs(audio))
        if peak == 0:
            return audio

        target_db = -3.0
        target_linear = 10 ** (target_db / 20.0)
        gain = target_linear / peak
        return audio * gain

    def load_and_prep_audio(self, path: os.PathLike | str) -> np.ndarray:
        """
        Load audio from disk, resample to target rate, convert to mono, and normalize.

        Tries loading with target sample rate and mono first (most robust); if that
        fails, falls back to load at native rate then resample/convert to mono.
        """
        file_path = self._validate_path(path)
        target_sr = self.config.target_sample_rate

        # First try: load directly at target rate and mono (avoids resample/channel issues).
        try:
            audio_mono, sr = librosa.load(file_path, sr=target_sr, mono=True)
            if audio_mono.size == 0:
                raise AudioLoadingError(f"Loaded audio is empty: {file_path}")
            audio_norm = self._normalize_to_minus_3db(audio_mono.astype(np.float32))
            return audio_norm
        except AudioLoadingError:
            raise
        except Exception:  # noqa: BLE001
            pass

        # Fallback: load at native rate and channels, then resample and convert to mono.
        try:
            audio, sr = librosa.load(file_path, sr=None, mono=False)
        except Exception as exc:  # noqa: BLE001
            raise AudioLoadingError(f"Failed to load audio: {file_path}") from exc

        if audio.size == 0:
            raise AudioLoadingError(f"Loaded audio is empty: {file_path}")

        if audio.ndim > 1:
            resampled_channels = []
            for ch in audio:
                resampled_ch = librosa.resample(ch, orig_sr=sr, target_sr=target_sr)
                resampled_channels.append(resampled_ch)
            max_len = max(len(ch) for ch in resampled_channels)
            padded = [
                np.pad(ch, (0, max_len - len(ch)), mode="constant")
                if len(ch) < max_len
                else ch[:max_len]
                for ch in resampled_channels
            ]
            audio_mono = np.mean(np.stack(padded, axis=0), axis=0)
        else:
            audio_mono = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)

        audio_norm = self._normalize_to_minus_3db(audio_mono.astype(np.float32))
        return audio_norm


__all__ = ["AudioProcessor", "AudioLoadingError"]

