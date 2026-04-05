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
    Initialize Whisper for WhiSLU using a fully trainable encoder+decoder:

    - Loads a full Whisper model (e.g., openai/whisper-medium) in float32.
    - Allows the encoder to adapt (no freezing) so acoustic cues can be
      leveraged for scam detection.
    - Does not use LoRA/PEFT; full model parameters are trainable.
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
    # IMPORTANT: Must clear BOTH model.config AND model.generation_config, because
    # model.generate() reads from generation_config, not model.config.
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []

    if hasattr(model, "generation_config"):
        model.generation_config.forced_decoder_ids = None
        model.generation_config.suppress_tokens = []
        # Also clear begin_suppress_tokens which can suppress JSON-critical tokens
        # like '{' and '"' at the start of generation.
        if hasattr(model.generation_config, "begin_suppress_tokens"):
            model.generation_config.begin_suppress_tokens = []
        # Whisper's custom generate() reconstructs forced_decoder_ids from
        # language/task/is_multilingual even when forced_decoder_ids is None.
        # Clear these so the decoder is truly free to emit JSON from the start.
        model.generation_config.language = None
        model.generation_config.task = None
        model.generation_config.is_multilingual = False

    # Enable gradient checkpointing to significantly reduce memory usage.
    model.config.use_cache = False
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()

    # Encoder and decoder are both trainable in this configuration.

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

    if hasattr(model, "generation_config"):
        model.generation_config.forced_decoder_ids = None
        model.generation_config.suppress_tokens = []
        if hasattr(model.generation_config, "begin_suppress_tokens"):
            model.generation_config.begin_suppress_tokens = []
        model.generation_config.language = None
        model.generation_config.task = None
        model.generation_config.is_multilingual = False

    model.eval()
    return model

