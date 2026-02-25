from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from openai import OpenAI
from pydantic import BaseModel, Field, ValidationError, field_validator

from .config_loader import LLMConfig


class ScamAnalysis(BaseModel):
    is_scam: bool = Field(description="True if the conversation is fraudulent.")
    risk_score: int = Field(ge=0, le=100, description="0-100 confidence score.")
    category: str = Field(description="Category label for the conversation.")
    reasoning: str = Field(description="Concise explanation, max 50 words.")
    urgency_detected: bool = Field(
        description="Whether the caller pressured the victim with urgency."
    )
    flagged_phrases: list[str] = Field(
        default_factory=list, description="Key phrases supporting the decision."
    )

    @field_validator("flagged_phrases", mode="before")
    @classmethod
    def coerce_flagged_phrases_to_list(cls, v: object) -> list[str]:
        """Accept str or non-list from LLM (e.g. empty string) and coerce to list."""
        if isinstance(v, list):
            return [str(x).strip() for x in v if str(x).strip()]
        if v is None:
            return []
        s = str(v).strip()
        if not s:
            return []
        # Single string: treat as one phrase or split by comma
        if "," in s:
            return [p.strip() for p in s.split(",") if p.strip()]
        return [s]


def get_llm_engine(config: LLMConfig):
    """Return the appropriate LLM engine for the configured backend."""
    if config.backend == "huggingface":
        from .llm_engine_hf import HuggingFaceLLMEngine
        return HuggingFaceLLMEngine(config)
    return OpenAILLMEngine(config)


@dataclass
class OpenAILLMEngine:
    config: LLMConfig
    prompt_path: Path | None = None

    def __post_init__(self) -> None:
        self._client = OpenAI(
            base_url=self.config.base_url,
            api_key=self.config.api_key,
        )
        if self.prompt_path is None:
            self.prompt_path = (
                Path(__file__).resolve().parents[1] / "config" / "prompt.txt"
            )
        self._system_prompt = self._load_prompt(self.prompt_path)

    @staticmethod
    def _load_prompt(path: Path) -> str:
        if not path.exists():
            raise FileNotFoundError(f"LLM system prompt not found at {path}")
        return path.read_text(encoding="utf-8")

    def _truncate_transcript(self, transcript_text: str) -> str:
        max_chars = max(1000, self.config.max_input_chars)
        if len(transcript_text) <= max_chars:
            return transcript_text
        # Simple truncation strategy; can be replaced by more advanced chunking if needed.
        return transcript_text[:max_chars]

    def _build_messages(self, transcript_text: str) -> list[dict[str, str]]:
        truncated = self._truncate_transcript(transcript_text)
        system_msg = self._system_prompt.strip()

        # Explicit JSON output instruction if structured outputs are not used.
        if not self.config.use_structured_output:
            system_msg += (
                "\n\nYou must output ONLY a single valid JSON object. "
                "Do not include any markdown code fences or extra text."
            )

        messages = [
            {"role": "system", "content": system_msg},
            {
                "role": "user",
                "content": (
                    "Analyze the following phone call transcript for potential scam "
                    "behavior. The text may contain ASR errors and may be multilingual. "
                    "Assess the semantic intent regardless of language.\n\n"
                    f"Transcript:\n{truncated}"
                ),
            },
        ]
        return messages

    def _parse_response_content(self, content: str) -> ScamAnalysis:
        try:
            payload = json.loads(content)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"LLM returned non-JSON output: {content}") from exc

        try:
            return ScamAnalysis(**payload)
        except ValidationError as exc:
            raise RuntimeError(f"Invalid ScamAnalysis schema: {exc}") from exc

    def analyze_transcript(self, transcript_text: str) -> ScamAnalysis:
        """
        Call the LLM to analyze an ASR transcript and return a validated ScamAnalysis.
        Supports both response_format-based structured output and prompt-enforced JSON.
        """
        messages = self._build_messages(transcript_text)

        kwargs: Dict[str, Any] = {
            "model": self.config.model,
            "messages": messages,
            "temperature": self.config.temperature,
        }

        # Use response_format for models that support it (e.g., GPT-4o).
        if self.config.use_structured_output:
            kwargs["response_format"] = {"type": "json_object"}

        response = self._client.chat.completions.create(**kwargs)
        content = response.choices[0].message.content or ""

        return self._parse_response_content(content)


# Alias for backward compatibility when using OpenAI backend
LLMEngine = OpenAILLMEngine


__all__ = ["LLMEngine", "OpenAILLMEngine", "ScamAnalysis", "get_llm_engine"]

