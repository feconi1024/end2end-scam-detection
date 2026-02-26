"""
Main orchestration logic for the SLM scam detection pipeline.
"""

import time
from pathlib import Path

from .audio_utils import load_audio_for_qwen
from .config_loader import load_settings
from .slm_engine import SpeechLanguageModel


def run_pipeline(
    audio_path: str | Path,
    config_path: str | Path | None = None,
    prompt_path: str | Path | None = None,
) -> dict:
    """
    Run the full scam detection pipeline: load audio -> SLM inference -> JSON result.

    Args:
        audio_path: Path to the target audio file (.wav, .mp3, etc.).
        config_path: Optional path to settings.yaml.
        prompt_path: Optional path to system prompt.txt.

    Returns:
        Parsed JSON result from the model, with added keys:
        - inference_time_sec: total wall-clock inference time (seconds)
        - inference_time_per_min_audio_sec: inference seconds per 1 minute of audio
        - audio_duration_sec: duration of the input audio in seconds
    """
    slm = SpeechLanguageModel(config_path=config_path)
    target_sr = slm.sampling_rate

    audio = load_audio_for_qwen(audio_path, target_sr=target_sr)
    audio_duration_sec = len(audio) / float(target_sr)
    duration_min = audio_duration_sec / 60.0

    if prompt_path is None:
        prompt_path = Path(__file__).resolve().parents[1] / "config" / "prompt.txt"
    else:
        prompt_path = Path(prompt_path)

    system_prompt = prompt_path.read_text(encoding="utf-8").strip()
    user_prompt = (
        "Analyze this telephone call for scam indicators. "
        "Output the JSON block as specified in the system instructions."
    )

    t0 = time.perf_counter()
    result = slm.analyze_audio(
        audio_array=audio,
        prompt_text=user_prompt,
        system_prompt=system_prompt,
    )
    inference_time_sec = time.perf_counter() - t0

    result["inference_time_sec"] = round(inference_time_sec, 3)
    result["inference_time_per_min_audio_sec"] = (
        round(inference_time_sec / duration_min, 3) if duration_min > 0 else None
    )
    result["audio_duration_sec"] = round(audio_duration_sec, 3)

    return result
