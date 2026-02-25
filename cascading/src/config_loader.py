import os
from pathlib import Path
from typing import List, Literal

import yaml
from pydantic import BaseModel, Field, ValidationError


class ASRConfig(BaseModel):
    model_size: str = Field(default="large-v3")
    device: str = Field(default="cuda")  # "cuda" or "cpu"
    compute_type: str = Field(default="float16")
    beam_size: int = Field(default=5)
    vad_filter: bool = Field(default=True)


class LLMConfig(BaseModel):
    backend: Literal["openai", "huggingface"] = Field(
        default="huggingface",
        description="LLM backend: 'openai' for OpenAI-compatible API, 'huggingface' for Transformers.",
    )
    # OpenAI-compatible API (used when backend == "openai")
    base_url: str = Field(default="http://localhost:8000/v1")
    api_key: str = Field(default="YOUR_API_KEY_HERE")
    model: str = Field(default="Qwen/Qwen2.5-72B-Instruct")
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    use_structured_output: bool = Field(
        default=True,
        description="If True, use response_format=json_object (OpenAI only). "
        "If False, rely on prompt-instructed raw JSON and parse manually.",
    )
    max_input_chars: int = Field(
        default=16000,
        description="Safety limit for very long transcripts before sending to LLM.",
    )
    # Hugging Face Transformers (used when backend == "huggingface")
    device_map: str | None = Field(
        default=None,
        description="Device for HF models: 'auto', 'cuda', 'cpu', or null for auto.",
    )
    max_new_tokens: int = Field(default=1024, description="Max tokens to generate (HF).")
    torch_dtype: str | None = Field(
        default="auto",
        description="Torch dtype for HF: 'auto', 'float16', 'bfloat16', or null.",
    )


class AudioConfig(BaseModel):
    target_sample_rate: int = Field(default=16000)
    supported_extensions: List[str] = Field(
        default_factory=lambda: [".wav", ".mp3", ".m4a"]
    )


class AppConfig(BaseModel):
    asr: ASRConfig = ASRConfig()
    llm: LLMConfig = LLMConfig()
    audio: AudioConfig = AudioConfig()


def load_settings(config_path: os.PathLike | str | None = None) -> AppConfig:
    """
    Load application settings from a YAML file into a validated Pydantic model.

    Environment overrides:
      - LLM_API_KEY: overrides llm.api_key
      - LLM_BASE_URL: overrides llm.base_url
      - LLM_MODEL: overrides llm.model
    """
    if config_path is None:
        config_path = Path(__file__).resolve().parents[1] / "config" / "settings.yaml"
    else:
        config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found at {config_path}")

    with config_path.open("r", encoding="utf-8") as f:
        raw_cfg = yaml.safe_load(f) or {}

    # Environment overrides
    llm_section = raw_cfg.setdefault("llm", {})
    api_key_env = os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY")
    base_url_env = os.getenv("LLM_BASE_URL")
    model_env = os.getenv("LLM_MODEL")

    if api_key_env:
        llm_section["api_key"] = api_key_env
    if base_url_env:
        llm_section["base_url"] = base_url_env
    if model_env:
        llm_section["model"] = model_env

    try:
        return AppConfig(**raw_cfg)
    except ValidationError as e:
        raise RuntimeError(f"Invalid configuration: {e}") from e


__all__ = ["ASRConfig", "LLMConfig", "AudioConfig", "AppConfig", "load_settings"]

