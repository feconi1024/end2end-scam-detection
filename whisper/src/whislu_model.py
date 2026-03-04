from __future__ import annotations

from typing import Any, Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import WhisperForConditionalGeneration, WhisperProcessor


class WhisluWithIntentHead(nn.Module):
    """
    Wrapper around WhisperForConditionalGeneration that adds an auxiliary
    intent classification head on top of the decoder hidden state at the
    first position. This encourages the model to learn the intent label
    explicitly, in addition to generating the JSON-formatted sequence.
    """

    def __init__(
        self,
        base_model: WhisperForConditionalGeneration,
        num_intents: int,
        intent_loss_weight: float = 1.0,
    ) -> None:
        super().__init__()
        self.base_model = base_model
        self.intent_head = nn.Linear(base_model.config.d_model, num_intents)
        self.intent_loss_weight = float(intent_loss_weight)

        # Expose config and generation_config so that Seq2SeqTrainer and
        # generation utilities can treat this wrapper like the underlying
        # Whisper model.
        self.config = base_model.config
        self.generation_config = getattr(base_model, "generation_config", None)

    def forward(
        self,
        input_features=None,
        labels=None,
        intent_labels: torch.Tensor | None = None,
        **kwargs: Any,
    ):
        # Some Trainer/Accelerate versions pass extra kwargs such as
        # `num_items_in_batch` that the underlying model does not accept.
        kwargs.pop("num_items_in_batch", None)

        outputs = self.base_model(
            input_features=input_features,
            labels=labels,
            output_hidden_states=True,
            **kwargs,
        )

        loss = outputs.loss
        intent_logits = None

        if intent_labels is not None:
            # Use the last decoder hidden state at position 0 as intent
            # representation.
            decoder_hiddens = outputs.decoder_hidden_states[-1]  # (B, T, D)
            cls_state = decoder_hiddens[:, 0, :]  # (B, D)
            intent_logits = self.intent_head(cls_state)

            valid_mask = intent_labels >= 0
            if valid_mask.any():
                aux_loss = F.cross_entropy(
                    intent_logits[valid_mask],
                    intent_labels[valid_mask],
                )
                loss = loss + self.intent_loss_weight * aux_loss

        outputs.loss = loss
        outputs.intent_logits = intent_logits
        return outputs

    def generate(self, *args: Any, **kwargs: Any):
        """
        Delegate generation to the underlying Whisper model, ignoring
        auxiliary-only arguments like `intent_labels` that are not used
        during decoding.
        """
        kwargs.pop("intent_labels", None)
        return self.base_model.generate(*args, **kwargs)


def initialize_whislu_model(
    model_name: str,
    processor: WhisperProcessor,
    config: Mapping[str, Any],
    device: str | None = None,
) -> nn.Module:
    """
    Initialize Whisper for WhiSLU using the "frozen encoder, full decoder"
    strategy plus an auxiliary intent head:

    - Loads a full Whisper model (e.g., openai/whisper-medium) in float32.
    - Freezes the entire encoder so only the decoder is fine-tuned.
    - Adds a small intent classification head trained jointly with the
      sequence loss to improve intent F1.
    - Disables forced decoder ids and token suppression so the model can
      freely emit JSON-formatted SLU outputs.
    """
    base = WhisperForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
    )

    # Do NOT resize token embeddings: we keep the original vocabulary and
    # express SLU targets as JSON strings within the existing token set.

    # Disable Whisper's default forced decoder ids and token suppression so that
    # the model is free to emit custom JSON immediately after <|startoftranscript|>.
    base.config.forced_decoder_ids = None
    base.config.suppress_tokens = []

    # Enable gradient checkpointing to significantly reduce memory usage.
    base.config.use_cache = False
    if hasattr(base, "gradient_checkpointing_enable"):
        base.gradient_checkpointing_enable()

    # Freeze the encoder; only the decoder (and lm_head) are updated.
    if hasattr(base, "model") and hasattr(base.model, "encoder"):
        base.model.encoder.requires_grad_(False)

    intent_classes: list[str] = list(config.get("intents", ["scam", "non_scam"]))
    intent_loss_weight: float = float(config.get("intent_loss_weight", 1.0))

    model: nn.Module = WhisluWithIntentHead(
        base_model=base,
        num_intents=len(intent_classes),
        intent_loss_weight=intent_loss_weight,
    )

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

