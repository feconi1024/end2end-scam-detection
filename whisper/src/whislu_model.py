from __future__ import annotations

from typing import Any, Mapping, Sequence

import torch
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import WhisperForConditionalGeneration, WhisperProcessor


def create_lora_config(lora_cfg: Mapping[str, Any]) -> LoraConfig:
    return LoraConfig(
        r=int(lora_cfg.get("r", 32)),
        lora_alpha=int(lora_cfg.get("lora_alpha", 64)),
        lora_dropout=float(lora_cfg.get("lora_dropout", 0.05)),
        target_modules=list(lora_cfg.get("target_modules", ["q_proj", "v_proj"])),
        bias="none",
        task_type="SEQ_2_SEQ_LM",
        modules_to_save=list(lora_cfg.get("modules_to_save", ["embed_tokens", "lm_head"])),
    )


def initialize_whislu_model(
    model_name: str,
    processor: WhisperProcessor,
    config: Mapping[str, Any],
    device: str | None = None,
) -> WhisperForConditionalGeneration:
    """
    Initialize Whisper + LoRA for WhiSLU.

    - Resizes token embeddings for newly added fraud tokens.
    - Applies LoRA on decoder attention (q_proj, v_proj by default).
    - Keeps adapter weights and embedding/lm_head trainable.
    """
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    model = WhisperForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    )

    # IMPORTANT: resize embeddings to account for added special tokens
    model.resize_token_embeddings(len(processor.tokenizer))

    # Disable Whisper's default forced decoder ids and token suppression so that
    # the model is free to emit custom fraud tokens after <|startoftranscript|>.
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []

    lora_cfg = config.get("lora", {})
    peft_config = create_lora_config(lora_cfg)

    model = get_peft_model(model, peft_config)

    # Some trainer versions may still try to pass `input_ids` to the model
    # (text-style interface). Whisper expects `input_features` instead, so
    # we defensively drop any stray `input_ids` argument at the PEFT wrapper
    # level to avoid runtime errors.
    def _patched_forward(self, *args, **kwargs):
        kwargs.pop("input_ids", None)
        return super(type(self), self).forward(*args, **kwargs)

    # Bind the patched forward to this model instance
    model.forward = _patched_forward.__get__(model, type(model))

    # Ensure that only LoRA adapters and the resized embeddings / lm_head are trainable.
    for name, param in model.named_parameters():
        if "lora_" in name or any(
            key in name for key in ("embed_tokens", "lm_head")
        ):
            param.requires_grad = True
        else:
            param.requires_grad = False

    if device is not None:
        model.to(device)

    return model


def load_model_with_lora_for_inference(
    base_model_name: str,
    processor: WhisperProcessor,
    peft_checkpoint: str,
    merge_adapters: bool = True,
) -> WhisperForConditionalGeneration:
    """
    Load base Whisper model + trained LoRA adapters for inference.
    Optionally merges adapters back into the base weights for faster inference.
    """
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    model = WhisperForConditionalGeneration.from_pretrained(
        base_model_name,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    )
    model.resize_token_embeddings(len(processor.tokenizer))

    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []

    model = PeftModel.from_pretrained(model, peft_checkpoint)

    if merge_adapters:
        model = model.merge_and_unload()

    # When not merged, we may still get stray `input_ids` from some pipelines;
    # drop them defensively.
    if isinstance(model, PeftModel):
        def _patched_forward(self, *args, **kwargs):
            kwargs.pop("input_ids", None)
            return super(type(self), self).forward(*args, **kwargs)

        model.forward = _patched_forward.__get__(model, type(model))

    model.eval()
    return model

