"""
Audio loading and resampling utilities for Qwen2-Audio.
"""

from pathlib import Path

import librosa
import numpy as np


def load_audio_for_qwen(path: str | Path, target_sr: int) -> np.ndarray:
    """
    Load an audio file and resample it for Qwen2-Audio.

    Args:
        path: Path to the audio file (.wav, .mp3, .flac, .m4a, .ogg, etc.).
        target_sr: Target sample rate (from processor.feature_extractor.sampling_rate).

    Returns:
        Mono audio as a 1D numpy array of shape (n_samples,), dtype float32,
        normalized to [-1, 1].

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the audio is empty or corrupted.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {path}")

    try:
        # Load as mono; sr=None preserves original rate for resampling
        y, sr = librosa.load(str(path), sr=None, mono=True)
    except Exception as e:
        raise ValueError(f"Failed to load audio from {path}: {e}") from e

    if y.size == 0:
        raise ValueError(f"Audio file is empty: {path}")

    if sr != target_sr:
        y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)

    return y.astype(np.float32)
