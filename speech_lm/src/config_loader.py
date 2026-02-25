"""
Configuration loader for the Speech Language Model scam detector.
Loads settings from YAML and validates with Pydantic.
"""

import os
from pathlib import Path
from typing import List, Literal

import yaml
from pydantic import BaseModel, Field, ValidationError


class ModelConfig(BaseModel):
    """Model loading parameters for Qwen2-Audio."""

    model_id: str = Field(
        default="Qwen/Qwen2-Audio-7B-Instruct",
        description="Hugging Face model ID.",
    )
    device_map: str = Field(
        default="auto",
        description="Device mapping for model distribution across GPUs.",
    )
    torch_dtype: Literal["float16", "bfloat16"] = Field(
        default="bfloat16",
        description="Torch dtype for model weights.",
    )
    load_in_8bit: bool = Field(
        default=False,
        description="Load model in 8-bit for memory-limited hardware.",
    )
    trust_remote_code: bool = Field(default=True)


class GenerationConfig(BaseModel):
    """Text generation parameters for deterministic scam classification."""

    max_new_tokens: int = Field(default=512, ge=1, le=2048)
    temperature: float = Field(default=0.1, ge=0.0, le=2.0)
    do_sample: bool = Field(default=True)
    pad_token_id: int | None = Field(default=None)


class AudioConfig(BaseModel):
    """Audio file handling settings."""

    supported_extensions: List[str] = Field(
        default_factory=lambda: [".wav", ".mp3", ".flac", ".m4a", ".ogg"]
    )


class SLMConfig(BaseModel):
    """Root configuration for the SLM scam detector."""

    model: ModelConfig = Field(default_factory=ModelConfig)
    generation: GenerationConfig = Field(default_factory=GenerationConfig)
    audio: AudioConfig = Field(default_factory=AudioConfig)


def load_settings(config_path: os.PathLike | str | None = None) -> SLMConfig:
    """
    Load SLM settings from a YAML file into a validated Pydantic model.

    Args:
        config_path: Path to settings.yaml. If None, uses config/settings.yaml
            relative to the speech_lm package root.

    Returns:
        Validated SLMConfig instance.

    Raises:
        FileNotFoundError: If config file does not exist.
        RuntimeError: If validation fails.
    """
    if config_path is None:
        config_path = Path(__file__).resolve().parents[1] / "config" / "settings.yaml"
    else:
        config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found at {config_path}")

    with config_path.open("r", encoding="utf-8") as f:
        raw_cfg = yaml.safe_load(f) or {}

    try:
        return SLMConfig(**raw_cfg)
    except ValidationError as e:
        raise RuntimeError(f"Invalid configuration: {e}") from e


__all__ = [
    "ModelConfig",
    "GenerationConfig",
    "AudioConfig",
    "SLMConfig",
    "load_settings",
]
