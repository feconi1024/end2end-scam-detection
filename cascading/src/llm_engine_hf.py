"""
Hugging Face Transformers backend for the scam detection LLM.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch
from pydantic import ValidationError
from transformers import AutoModelForCausalLM, AutoTokenizer

from .config_loader import LLMConfig
from .llm_engine import ScamAnalysis


def _extract_json_from_text(text: str) -> str:
    """Extract the first complete JSON object from model output (handles markdown/prefix)."""
    text = text.strip()
    # Remove markdown code fence if present
    if "```json" in text:
        start = text.find("```json") + 7
        end = text.find("```", start)
        if end != -1:
            return text[start:end].strip()
    if "```" in text:
        start = text.find("```") + 3
        end = text.find("```", start)
        if end != -1:
            return text[start:end].strip()
    # Find first { and matching }
    start = text.find("{")
    if start == -1:
        raise ValueError(f"No JSON object found in output: {text[:500]!r}")
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    raise ValueError(f"Incomplete JSON in output: {text[start:start+500]!r}")


def _get_torch_dtype(dtype_str: str | None) -> torch.dtype | None:
    if dtype_str is None or dtype_str == "auto":
        return None
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }.get(dtype_str, None)


@dataclass
class HuggingFaceLLMEngine:
    """LLM engine using Hugging Face Transformers (local models)."""

    config: LLMConfig
    prompt_path: Path | None = None

    def __post_init__(self) -> None:
        if self.prompt_path is None:
            self.prompt_path = (
                Path(__file__).resolve().parents[1] / "config" / "prompt.txt"
            )
        self._system_prompt = self._load_prompt(self.prompt_path)
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.config.model,
            trust_remote_code=True,
        )
        device_map = self.config.device_map or "auto"
        torch_dtype = _get_torch_dtype(self.config.torch_dtype)
        self._model = AutoModelForCausalLM.from_pretrained(
            self.config.model,
            torch_dtype=torch_dtype if torch_dtype is not None else "auto",
            device_map=device_map,
            trust_remote_code=True,
        )
        self._model.eval()

    @staticmethod
    def _load_prompt(path: Path) -> str:
        if not path.exists():
            raise FileNotFoundError(f"LLM system prompt not found at {path}")
        return path.read_text(encoding="utf-8")

    def _truncate_transcript(self, transcript_text: str) -> str:
        max_chars = max(1000, self.config.max_input_chars)
        if len(transcript_text) <= max_chars:
            return transcript_text
        return transcript_text[:max_chars]

    def _build_messages(self, transcript_text: str) -> list[dict[str, str]]:
        truncated = self._truncate_transcript(transcript_text)
        system_msg = self._system_prompt.strip()
        system_msg += (
            "\n\nYou must output ONLY a single valid JSON object. "
            "Do not include any markdown code fences or extra text."
        )
        user_content = (
            "Analyze the following phone call transcript for potential scam "
            "behavior. The text may contain ASR errors and may be multilingual. "
            "Assess the semantic intent regardless of language.\n\n"
            f"Transcript:\n{truncated}"
        )
        return [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_content},
        ]

    def _parse_response_content(self, content: str) -> ScamAnalysis:
        json_str = _extract_json_from_text(content)
        try:
            payload = json.loads(json_str)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"LLM returned invalid JSON: {content[:500]!r}") from exc
        try:
            return ScamAnalysis(**payload)
        except ValidationError as exc:
            raise RuntimeError(f"Invalid ScamAnalysis schema: {exc}") from exc

    def analyze_transcript(self, transcript_text: str) -> ScamAnalysis:
        """Run scam analysis using the local Hugging Face model."""
        messages = self._build_messages(transcript_text)
        try:
            prompt = self._tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            # Fallback for tokenizers without chat template: concatenate messages
            prompt = "\n\n".join(m.get("content", "") for m in messages)
        inputs = self._tokenizer(prompt, return_tensors="pt")
        if hasattr(self._model, "device"):
            inputs = {k: v.to(self._model.device) for k, v in inputs.items()}
        else:
            inputs = {k: v.to(next(self._model.parameters()).device) for k, v in inputs.items()}
        with torch.no_grad():
            out = self._model.generate(
                **inputs,
                max_new_tokens=self.config.max_new_tokens,
                temperature=self.config.temperature if self.config.temperature > 0 else None,
                do_sample=self.config.temperature > 0,
                pad_token_id=self._tokenizer.eos_token_id or self._tokenizer.pad_token_id,
            )
        # Decode only the generated part (after input length)
        input_len = inputs["input_ids"].shape[1]
        generated = out[0][input_len:]
        content = self._tokenizer.decode(generated, skip_special_tokens=True).strip()
        return self._parse_response_content(content)


__all__ = ["HuggingFaceLLMEngine"]
