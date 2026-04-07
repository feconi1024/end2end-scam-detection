from __future__ import annotations

import logging
import json
from pathlib import Path
from typing import Any, Mapping, Optional

import torch
import torch.nn as nn
from transformers import WhisperConfig, WhisperModel, WhisperPreTrainedModel
from transformers.modeling_outputs import SequenceClassifierOutput

logger = logging.getLogger(__name__)


class GradientReversal(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_tensor: torch.Tensor, alpha: float) -> torch.Tensor:
        ctx.alpha = float(alpha)
        return input_tensor.view_as(input_tensor)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return -ctx.alpha * grad_output, None


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
        num_family_labels: int | None = None,
        family_adversarial_weight: float | None = None,
        family_gradient_scale: float | None = None,
        compute_family_loss_on_eval: bool | None = None,
        **kwargs,
    ):
        del kwargs
        super().__init__(config)
        if classifier_dropout is not None:
            config.classifier_dropout = float(classifier_dropout)
        if classifier_pooling is not None:
            config.classifier_pooling = str(classifier_pooling)
        if num_family_labels is not None:
            config.num_family_labels = int(num_family_labels)
        if family_adversarial_weight is not None:
            config.family_adversarial_weight = float(family_adversarial_weight)
        if family_gradient_scale is not None:
            config.family_gradient_scale = float(family_gradient_scale)
        if compute_family_loss_on_eval is not None:
            config.compute_family_loss_on_eval = bool(compute_family_loss_on_eval)

        self.num_labels = int(getattr(config, "num_labels", 2))
        self.pooling = str(getattr(config, "classifier_pooling", "mean"))
        dropout = float(getattr(config, "classifier_dropout", 0.1))
        self.num_family_labels = int(getattr(config, "num_family_labels", 0))
        self.family_adversarial_weight = float(getattr(config, "family_adversarial_weight", 0.0))
        self.family_gradient_scale = float(getattr(config, "family_gradient_scale", 1.0))
        self.compute_family_loss_on_eval = bool(getattr(config, "compute_family_loss_on_eval", False))

        self.whisper = WhisperModel(config)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(config.d_model, self.num_labels)
        self.family_classifier: Optional[nn.Linear]
        if self.num_family_labels > 0 and self.family_adversarial_weight > 0.0:
            self.family_classifier = nn.Linear(config.d_model, self.num_family_labels)
        else:
            self.family_classifier = None
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
        family_labels: Optional[torch.LongTensor] = None,
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

        if (
            self.family_classifier is not None
            and family_labels is not None
            and (self.training or self.compute_family_loss_on_eval)
        ):
            reversed_features = GradientReversal.apply(pooled, self.family_gradient_scale)
            family_logits = self.family_classifier(self.dropout(reversed_features))
            family_loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            family_loss = family_loss_fct(
                family_logits.view(-1, self.num_family_labels),
                family_labels.view(-1),
            )
            weighted_family_loss = self.family_adversarial_weight * family_loss
            loss = weighted_family_loss if loss is None else loss + weighted_family_loss

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
    family2id: Optional[Mapping[str, int]] = None,
    device: str | None = None,
) -> WhisperEncoderForScamClassification:
    classifier_cfg = config.get("classifier", {})
    family_adv_cfg = classifier_cfg.get("family_adversarial", {})
    base_config = WhisperConfig.from_pretrained(model_name)
    base_config.num_labels = len(label2id)
    base_config.label2id = dict(label2id)
    base_config.id2label = {int(idx): label for idx, label in id2label.items()}
    base_config.classifier_dropout = float(classifier_cfg.get("dropout", 0.1))
    base_config.classifier_pooling = str(classifier_cfg.get("pooling", "mean"))
    base_config.num_family_labels = int(len(family2id or {})) if bool(family_adv_cfg.get("enabled", False)) else 0
    base_config.family_adversarial_weight = float(family_adv_cfg.get("weight", 0.0))
    base_config.family_gradient_scale = float(family_adv_cfg.get("gradient_scale", 1.0))
    base_config.compute_family_loss_on_eval = bool(family_adv_cfg.get("compute_on_eval", False))

    model = WhisperEncoderForScamClassification(base_config)
    pretrained_whisper = WhisperModel.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
    )
    missing_keys, unexpected_keys = model.whisper.load_state_dict(
        pretrained_whisper.state_dict(),
        strict=True,
    )
    if missing_keys or unexpected_keys:
        raise RuntimeError(
            "Failed to load pretrained Whisper weights cleanly into the classifier model. "
            f"Missing: {missing_keys} | Unexpected: {unexpected_keys}"
        )
    logger.info("Loaded pretrained Whisper encoder/decoder weights into classifier backbone.")
    del pretrained_whisper

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


def _has_model_weights(model_dir: Path) -> bool:
    return any(
        (model_dir / filename).exists()
        for filename in (
            "pytorch_model.bin",
            "model.safetensors",
            "tf_model.h5",
            "model.ckpt.index",
            "flax_model.msgpack",
        )
    )


def resolve_classifier_model_dir(model_dir: str | Path) -> Path:
    requested = Path(model_dir)
    candidates = []

    if requested.exists():
        candidates.append(requested)

    if requested.name == "model":
        candidates.append(requested.parent)
    else:
        candidates.append(requested / "model")

    trainer_state_path = requested.parent / "trainer_state.json" if requested.name == "model" else requested / "trainer_state.json"
    if trainer_state_path.exists():
        try:
            trainer_state = json.loads(trainer_state_path.read_text(encoding="utf-8"))
            best_checkpoint = trainer_state.get("best_model_checkpoint")
            if best_checkpoint:
                candidates.append(Path(best_checkpoint))
        except Exception:
            pass

    checkpoint_root = requested.parent if requested.name == "model" else requested
    if checkpoint_root.exists():
        checkpoint_dirs = sorted(
            (
                path for path in checkpoint_root.glob("checkpoint-*")
                if path.is_dir()
            ),
            key=lambda path: int(path.name.split("-")[-1]) if path.name.split("-")[-1].isdigit() else -1,
            reverse=True,
        )
        candidates.extend(checkpoint_dirs)

    seen = set()
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        if candidate.is_dir() and _has_model_weights(candidate):
            return candidate

    raise OSError(
        f"Could not find a saved classifier model under {requested}. "
        "Checked the requested directory, its parent/output dir, and available checkpoint-* folders."
    )


def load_classifier_for_inference(model_dir: str | Path) -> WhisperEncoderForScamClassification:
    resolved_model_dir = resolve_classifier_model_dir(model_dir)
    if Path(model_dir) != resolved_model_dir:
        logger.warning("Classifier model dir %s was missing; loading fallback checkpoint from %s", model_dir, resolved_model_dir)
    model = WhisperEncoderForScamClassification.from_pretrained(
        str(resolved_model_dir),
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
    )
    for parameter in model.whisper.decoder.parameters():
        parameter.requires_grad = False
    model.eval()
    return model
