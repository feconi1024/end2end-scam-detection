from __future__ import annotations

from types import MethodType
from typing import Any, Mapping

import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor


def configure_generation_for_json(
    model: WhisperForConditionalGeneration,
) -> WhisperForConditionalGeneration:
    """
    Align Whisper generation with the training target format.

    Our labels are trained as:
      <|startoftranscript|>{...JSON...}<|endoftext|>

    Hugging Face Whisper generation otherwise auto-injects prompt tokens such
    as language, task, and <|notimestamps|>, which creates a train/inference
    mismatch and can easily corrupt the opening JSON tokens.
    """
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = None
    if hasattr(model.config, "begin_suppress_tokens"):
        model.config.begin_suppress_tokens = None

    if getattr(model, "generation_config", None) is not None:
        model.generation_config.forced_decoder_ids = None
        model.generation_config.suppress_tokens = None
        if hasattr(model.generation_config, "begin_suppress_tokens"):
            model.generation_config.begin_suppress_tokens = None
        if hasattr(model.generation_config, "language"):
            model.generation_config.language = None
        if hasattr(model.generation_config, "task"):
            model.generation_config.task = None
        if hasattr(model.generation_config, "return_timestamps"):
            model.generation_config.return_timestamps = False
        if hasattr(model.generation_config, "is_multilingual"):
            # Keep the decoder prompt to SOT-only so generation matches
            # the SOT-only supervision used during training.
            model.generation_config.is_multilingual = False

    if not hasattr(model, "_whislu_original_retrieve_init_tokens"):
        model._whislu_original_retrieve_init_tokens = model._retrieve_init_tokens

        def _retrieve_init_tokens_without_notimestamps(self, *args, **kwargs):
            init_tokens = self._whislu_original_retrieve_init_tokens(*args, **kwargs)
            generation_config = kwargs.get("generation_config")
            if generation_config is None and len(args) >= 3:
                generation_config = args[2]

            no_timestamps_token_id = getattr(generation_config, "no_timestamps_token_id", None)
            if no_timestamps_token_id is None:
                return init_tokens

            if (
                isinstance(init_tokens, torch.Tensor)
                and init_tokens.ndim == 2
                and init_tokens.shape[1] > 1
                and torch.all(init_tokens[:, -1] == no_timestamps_token_id)
            ):
                # HF Whisper auto-appends <|notimestamps|> when
                # return_timestamps=False. Our labels were trained from SOT-only,
                # so remove that trailing prompt token while keeping
                # `no_timestamps_token_id` intact for internal generation logic.
                init_tokens = init_tokens[:, :-1]

            return init_tokens

        model._retrieve_init_tokens = MethodType(
            _retrieve_init_tokens_without_notimestamps,
            model,
        )

    return model


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

    configure_generation_for_json(model)

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

    configure_generation_for_json(model)
    model.eval()
    return model

