"""
Speech Language Model engine wrapping Qwen2-Audio for scam detection.
"""

import ast
import json
from pathlib import Path
from typing import Any

import torch
from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration

from .config_loader import GenerationConfig, ModelConfig, load_settings

# Default sampling rate for Whisper-based feature extractor (Qwen2-Audio)
DEFAULT_SAMPLING_RATE = 16000


def _get_torch_dtype(dtype_str: str) -> torch.dtype:
    """Resolve torch dtype, preferring bfloat16 when available."""
    if dtype_str == "bfloat16" and torch.cuda.is_available():
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16
    if dtype_str == "float16":
        return torch.float16
    return torch.float16


def _extract_json_from_text(text: str) -> dict[str, Any] | None:
    """
    Extract a JSON or Python-dict object from model output.
    SLMs may prepend text before the block or use single-quoted Python dicts.
    Uses brace matching to find the outermost {...} block, then tries
    json.loads and ast.literal_eval as fallback.
    """
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    for i, c in enumerate(text[start:], start=start):
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                block = text[start : i + 1]
                try:
                    return json.loads(block)
                except json.JSONDecodeError:
                    pass
                try:
                    return ast.literal_eval(block)
                except (ValueError, SyntaxError):
                    pass
                return None
    return None


def _map_to_scam_schema(parsed: dict[str, Any]) -> dict[str, Any]:
    """
    Map model output that uses different keys (e.g. scam_indicators, spoken_text)
    to the expected scam-detection schema so the pipeline always returns
    is_scam, fraud_type, acoustic_analysis, semantic_analysis, confidence_score.
    """
    # Already in expected shape
    if "is_scam" in parsed and "fraud_type" in parsed:
        return parsed

    result: dict[str, Any] = {
        "is_scam": False,
        "fraud_type": "Normal",
        "acoustic_analysis": "",
        "semantic_analysis": "",
        "confidence_score": 50,
    }

    # Map from common alternative outputs
    semantic = (
        parsed.get("semantic_analysis")
        or parsed.get("scam_indicators")
        or parsed.get("reasoning")
        or ""
    )
    if semantic:
        result["semantic_analysis"] = semantic
        # Infer is_scam from semantic content
        low = semantic.lower()
        if any(
            w in low
            for w in (
                "scam",
                "phishing",
                "impersonat",
                "fraud",
                "deceptive",
                "fake",
                "urgent",
                "pressure",
            )
        ):
            result["is_scam"] = True

    result["acoustic_analysis"] = parsed.get("acoustic_analysis") or parsed.get(
        "speaker_gender", ""
    )
    if parsed.get("age_group"):
        result["acoustic_analysis"] = (
            (result["acoustic_analysis"] + f" (age: {parsed['age_group']})").strip()
        )

    if "fraud_type" in parsed:
        result["fraud_type"] = parsed["fraud_type"]
    elif result["is_scam"] and "impersonat" in result["semantic_analysis"].lower():
        result["fraud_type"] = "Impersonation"
    elif result["is_scam"]:
        result["fraud_type"] = "Impersonation"  # default when semantic suggests scam

    if "confidence_score" in parsed:
        try:
            result["confidence_score"] = min(100, max(0, int(parsed["confidence_score"])))
        except (TypeError, ValueError):
            result["confidence_score"] = 80 if result["is_scam"] else 20

    return result


class SpeechLanguageModel:
    """
    Wrapper for Qwen2-Audio model for multimodal (audio + text) scam analysis.
    """

    def __init__(self, config_path: str | Path | None = None):
        """
        Initialize the SLM by loading the processor and model.

        Args:
            config_path: Optional path to settings.yaml. Uses default if None.
        """
        cfg = load_settings(config_path)
        model_cfg = cfg.model
        self.generation_cfg = cfg.generation

        dtype = _get_torch_dtype(model_cfg.torch_dtype)
        load_kwargs: dict[str, Any] = {
            "device_map": model_cfg.device_map,
            "torch_dtype": dtype,
            "trust_remote_code": model_cfg.trust_remote_code,
        }
        if model_cfg.load_in_8bit:
            load_kwargs["load_in_8bit"] = True
        else:
            load_kwargs["low_cpu_mem_usage"] = True

        self.processor = AutoProcessor.from_pretrained(
            model_cfg.model_id,
            trust_remote_code=model_cfg.trust_remote_code,
        )
        # Avoid "pass sampling_rate to WhisperFeatureExtractor" warning
        if getattr(
            self.processor.feature_extractor,
            "sampling_rate",
            None,
        ) is None:
            self.processor.feature_extractor.sampling_rate = DEFAULT_SAMPLING_RATE

        # Avoid "Sliding Window Attention is enabled but not implemented for sdpa" warning
        load_kwargs["attn_implementation"] = "eager"

        self.model = Qwen2AudioForConditionalGeneration.from_pretrained(
            model_cfg.model_id,
            **load_kwargs,
        )

    @property
    def sampling_rate(self) -> int:
        """Sample rate required by the model's feature extractor."""
        return self.processor.feature_extractor.sampling_rate

    def analyze_audio(
        self,
        audio_array: "np.ndarray",
        prompt_text: str,
        system_prompt: str | None = None,
    ) -> dict[str, Any]:
        """
        Run multimodal inference: audio + text prompt -> structured JSON.

        Args:
            audio_array: Mono audio as float32 numpy array at the model's sampling rate.
            prompt_text: User instruction / analysis question.
            system_prompt: Optional system instruction (e.g. scam detection role).

        Returns:
            Parsed JSON dict from model output. If parsing fails, returns
            {"raw_response": <text>, "parse_error": true}.
        """
        messages: list[dict[str, Any]] = []

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        # ChatML format: user content can be list of {type, audio_url/text}
        # Use placeholder for audio_url; actual audio is passed separately
        messages.append({
            "role": "user",
            "content": [
                {"type": "audio", "audio_url": "local"},
                {"type": "text", "text": prompt_text},
            ],
        })

        text = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
        )
        audios = [audio_array]

        inputs = self.processor(
            text=text,
            audio=audios,
            return_tensors="pt",
            padding=True,
        )

        device = self.model.device
        inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}

        input_length = inputs["input_ids"].shape[1]
        gen_kwargs = {
            "max_new_tokens": self.generation_cfg.max_new_tokens,
            "temperature": self.generation_cfg.temperature,
            "do_sample": self.generation_cfg.do_sample,
            "pad_token_id": self.generation_cfg.pad_token_id or self.processor.tokenizer.pad_token_id,
        }
        if gen_kwargs["pad_token_id"] is None:
            gen_kwargs["pad_token_id"] = self.processor.tokenizer.eos_token_id

        generate_ids = self.model.generate(**inputs, **gen_kwargs)
        generate_ids = generate_ids[:, input_length:]

        response = self.processor.batch_decode(
            generate_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

        parsed = _extract_json_from_text(response)
        if parsed is not None:
            return _map_to_scam_schema(parsed)
        return {"raw_response": response.strip(), "parse_error": True}
