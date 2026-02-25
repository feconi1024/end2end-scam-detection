"""
Main orchestration logic for the SLM scam detection pipeline.
"""

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
        Parsed JSON result from the model.
    """
    slm = SpeechLanguageModel(config_path=config_path)
    target_sr = slm.sampling_rate

    audio = load_audio_for_qwen(audio_path, target_sr=target_sr)

    if prompt_path is None:
        prompt_path = Path(__file__).resolve().parents[1] / "config" / "prompt.txt"
    else:
        prompt_path = Path(prompt_path)

    system_prompt = prompt_path.read_text(encoding="utf-8").strip()
    user_prompt = (
        "Analyze this telephone call for scam indicators. "
        "Output the JSON block as specified in the system instructions."
    )

    result = slm.analyze_audio(
        audio_array=audio,
        prompt_text=user_prompt,
        system_prompt=system_prompt,
    )
    return result
