from __future__ import annotations

from typing import Any, Mapping

import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor


def initialize_whislu_model(
    model_name: str,
    processor: WhisperProcessor,
    config: Mapping[str, Any],
    device: str | None = None,
) -> WhisperForConditionalGeneration:
    """
    Initialize Whisper for WhiSLU using the "frozen encoder, full decoder"
    strategy inspired by WhiSLU/Whisper-SLU:

    - Loads a full Whisper model (e.g., openai/whisper-medium) in float32.
    - Freezes the entire encoder so only the decoder is fine-tuned.
    - Does not use LoRA/PEFT; full decoder parameters are trainable.
    - Disables forced decoder ids and token suppression so the model can
      freely emit JSON-formatted SLU outputs.
    """
    model = WhisperForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
    )

    # Do NOT resize token embeddings: we keep the original vocabulary and
    # express SLU targets as JSON strings within the existing token set.

    # Disable Whisper's default forced decoder ids and token suppression so that
    # the model is free to emit custom JSON immediately after <|startoftranscript|>.
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []

    # Enable gradient checkpointing to significantly reduce memory usage.
    model.config.use_cache = False
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()

    # Freeze the encoder; only the decoder (and lm_head) are updated.
    if hasattr(model, "model") and hasattr(model.model, "encoder"):
        model.model.encoder.requires_grad_(False)

    if device is not None:
        model.to(device)

    return model


def load_whislu_for_inference(
    model_dir: str,
    processor: WhisperProcessor,
) -> WhisperForConditionalGeneration:
    """
    Load a fully fine-tuned WhiSLU model (no LoRA) for inference.
    """
    model = WhisperForConditionalGeneration.from_pretrained(
        model_dir,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
    )

    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []

    model.eval()
    return model

