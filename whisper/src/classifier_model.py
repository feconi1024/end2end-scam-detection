from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Optional

import torch
import torch.nn as nn
from transformers import WhisperModel, WhisperPreTrainedModel
from transformers.modeling_outputs import SequenceClassifierOutput


class WhisperEncoderForScamClassification(WhisperPreTrainedModel):
    """
    Whisper encoder + lightweight classification head.

    This avoids the brittle full-sequence JSON decoding objective and is a
    better fit for the repo's primary binary scam-detection task.
    """

    def __init__(
        self,
        config,
        classifier_dropout: float | None = None,
        classifier_pooling: str | None = None,
        **kwargs,
    ):
        del kwargs
        super().__init__(config)
        if classifier_dropout is not None:
            config.classifier_dropout = float(classifier_dropout)
        if classifier_pooling is not None:
            config.classifier_pooling = str(classifier_pooling)

        self.num_labels = int(getattr(config, "num_labels", 2))
        self.pooling = str(getattr(config, "classifier_pooling", "mean"))
        dropout = float(getattr(config, "classifier_dropout", 0.1))

        self.whisper = WhisperModel(config)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(config.d_model, self.num_labels)
        self.class_weights: Optional[torch.Tensor] = None

        self.post_init()

    def set_class_weights(self, class_weights: Optional[torch.Tensor]) -> None:
        self.class_weights = class_weights

    def _pool_hidden_states(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.pooling == "first":
            return hidden_states[:, 0, :]
        if self.pooling == "max":
            return hidden_states.max(dim=1).values
        return hidden_states.mean(dim=1)

    def forward(
        self,
        input_features: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> SequenceClassifierOutput:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        encoder_outputs = self.whisper.encoder(
            input_features=input_features,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )

        pooled = self._pool_hidden_states(encoder_outputs.last_hidden_state)
        logits = self.classifier(self.dropout(pooled))

        loss = None
        if labels is not None:
            class_weights = None
            if self.class_weights is not None:
                class_weights = self.class_weights.to(logits.device, dtype=logits.dtype)
            loss_fct = nn.CrossEntropyLoss(weight=class_weights)
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits, encoder_outputs.hidden_states, encoder_outputs.attentions)
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


def initialize_classifier_model(
    model_name: str,
    label2id: Mapping[str, int],
    id2label: Mapping[int, str],
    config: Mapping[str, Any],
    device: str | None = None,
) -> WhisperEncoderForScamClassification:
    classifier_cfg = config.get("classifier", {})
    model = WhisperEncoderForScamClassification.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
        num_labels=len(label2id),
        label2id=dict(label2id),
        id2label={int(idx): label for idx, label in id2label.items()},
        classifier_dropout=float(classifier_cfg.get("dropout", 0.1)),
        classifier_pooling=str(classifier_cfg.get("pooling", "mean")),
    )

    # The decoder is not used in the encoder-classifier path.
    for parameter in model.whisper.decoder.parameters():
        parameter.requires_grad = False

    if bool(classifier_cfg.get("gradient_checkpointing", True)) and hasattr(
        model.whisper.encoder, "gradient_checkpointing"
    ):
        model.config.use_cache = False
        model.gradient_checkpointing_enable()

    if bool(classifier_cfg.get("freeze_encoder", False)):
        for parameter in model.whisper.encoder.parameters():
            parameter.requires_grad = False

    if device is not None:
        model.to(device)

    return model


def load_classifier_for_inference(model_dir: str | Path) -> WhisperEncoderForScamClassification:
    model = WhisperEncoderForScamClassification.from_pretrained(
        str(model_dir),
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
    )
    for parameter in model.whisper.decoder.parameters():
        parameter.requires_grad = False
    model.eval()
    return model
