from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
from faster_whisper import WhisperModel

from .config_loader import ASRConfig


@dataclass
class ASRResult:
    text: str
    audio_duration_s: float
    asr_time_s: float


@dataclass
class ASREngine:
    """
    Wrapper around faster-whisper for streamlined transcription.
    """

    config: ASRConfig
    _model: WhisperModel | None = None

    @property
    def model(self) -> WhisperModel:
        if self._model is None:
            # Note: if faster-whisper was installed with Flash Attention support,
            # it will be enabled automatically by the underlying implementation.
            self._model = WhisperModel(
                self.config.model_size,
                device=self.config.device,
                compute_type=self.config.compute_type,
            )
        return self._model

    def transcribe(self, audio_array: np.ndarray) -> ASRResult:
        """
        Transcribe an audio array into a single combined string.
        Uses VAD to filter silence and beam search for robustness.
        """
        start_time = time.time()
        segments, info = self.model.transcribe(
            audio_array,
            beam_size=self.config.beam_size,
            vad_filter=self.config.vad_filter,
        )

        all_text_parts = []
        for segment in segments:
            if segment.text:
                all_text_parts.append(segment.text.strip())

        transcription = " ".join(all_text_parts).strip()
        elapsed = time.time() - start_time

        detected_lang = getattr(info, "language", "unknown")
        duration = float(getattr(info, "duration", 0.0) or 0.0)
        print(
            f"[ASR] Language: {detected_lang} | Duration: {duration:.2f}s | "
            f"Inference time: {elapsed:.2f}s"
        )

        return ASRResult(
            text=transcription,
            audio_duration_s=duration,
            asr_time_s=elapsed,
        )


__all__ = ["ASREngine", "ASRResult"]

