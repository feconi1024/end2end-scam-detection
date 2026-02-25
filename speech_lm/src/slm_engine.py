"""
Speech Language Model engine wrapping Qwen2-Audio for scam detection.
"""

import json
from pathlib import Path
from typing import Any

import torch
from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration

from .config_loader import GenerationConfig, ModelConfig, load_settings


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
    Extract a JSON object from model output.
    SLMs may prepend text like 'Here is the analysis:' before the JSON block.
    Uses brace matching to find the outermost {...} block.
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
                try:
                    return json.loads(text[start : i + 1])
                except json.JSONDecodeError:
                    return None
    return None


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
            return parsed
        return {"raw_response": response.strip(), "parse_error": True}
