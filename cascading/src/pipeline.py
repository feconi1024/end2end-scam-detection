from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time

from .audio_processor import AudioProcessor, AudioLoadingError
from .asr_engine import ASREngine, ASRResult
from .config_loader import AppConfig
from .llm_engine import ScamAnalysis, get_llm_engine


@dataclass
class PipelineResult:
    analysis: ScamAnalysis
    asr_time_s: float
    llm_time_s: float
    total_time_s: float
    audio_duration_s: float


@dataclass
class ScamDetectionPipeline:
    config: AppConfig

    def __post_init__(self) -> None:
        self.audio_processor = AudioProcessor(self.config.audio)
        self.asr_engine = ASREngine(self.config.asr)
        self.llm_engine = get_llm_engine(self.config.llm)

    def run(self, audio_path: str | Path) -> PipelineResult:
        """
        End-to-end processing: audio file -> transcription -> scam analysis.
        Also measures timing for ASR and LLM steps.
        """
        start_total = time.time()
        asr_time_s = 0.0
        llm_time_s = 0.0
        audio_duration_s = 0.0

        try:
            audio = self.audio_processor.load_and_prep_audio(audio_path)
        except AudioLoadingError:
            analysis = ScamAnalysis(
                is_scam=False,
                risk_score=0,
                category="Unknown",
                reasoning="Audio could not be processed; returning default safe result.",
                urgency_detected=False,
                flagged_phrases=[],
            )
            total_time_s = time.time() - start_total
            return PipelineResult(
                analysis=analysis,
                asr_time_s=asr_time_s,
                llm_time_s=llm_time_s,
                total_time_s=total_time_s,
                audio_duration_s=audio_duration_s,
            )

        asr_result: ASRResult = self.asr_engine.transcribe(audio)
        audio_duration_s = asr_result.audio_duration_s
        asr_time_s = asr_result.asr_time_s

        if not asr_result.text:
            analysis = ScamAnalysis(
                is_scam=False,
                risk_score=0,
                category="Unknown",
                reasoning="Transcription was empty or unintelligible.",
                urgency_detected=False,
                flagged_phrases=[],
            )
            total_time_s = time.time() - start_total
            return PipelineResult(
                analysis=analysis,
                asr_time_s=asr_time_s,
                llm_time_s=llm_time_s,
                total_time_s=total_time_s,
                audio_duration_s=audio_duration_s,
            )

        llm_start = time.time()
        analysis = self.llm_engine.analyze_transcript(asr_result.text)
        llm_time_s = time.time() - llm_start

        total_time_s = time.time() - start_total

        return PipelineResult(
            analysis=analysis,
            asr_time_s=asr_time_s,
            llm_time_s=llm_time_s,
            total_time_s=total_time_s,
            audio_duration_s=audio_duration_s,
        )


__all__ = ["ScamDetectionPipeline", "PipelineResult"]

