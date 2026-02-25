from __future__ import annotations

from typing import Any, Dict, Optional

import torch
from torch import Tensor, nn
from transformers import (
    BertForSequenceClassification,
    BertConfig,
    WhisperModel,
)

from .projector import ModalityProjector, ProjectorConfig


class DifferentiableCascadeModel(nn.Module):
    """
    Whisper encoder -> ModalityProjector -> BERT classifier (+ CTC head).

    The projector output is fed directly as `inputs_embeds` into BERT to
    preserve gradient flow from the classifier back into the acoustic space.
    """

    def __init__(
        self,
        whisper_model_name: str,
        bert_model_name: str,
        num_labels: int,
        ctc_vocab_size: int,
        ctc_blank_id: int,
        projector_cfg_overrides: Optional[Dict[str, Any]] = None,
        label2id: Optional[Dict[str, int]] = None,
        id2label: Optional[Dict[int, str]] = None,
    ) -> None:
        super().__init__()

        # Load Whisper encoder
        whisper = WhisperModel.from_pretrained(whisper_model_name)
        self.audio_encoder = whisper.encoder
        acoustic_dim = whisper.config.d_model

        # Build BERT classifier (semantic decoder)
        bert_config = BertConfig.from_pretrained(
            bert_model_name,
            num_labels=num_labels,
            label2id=label2id,
            id2label=id2label,
        )
        self.semantic_decoder = BertForSequenceClassification.from_pretrained(
            bert_model_name,
            config=bert_config,
        )
        semantic_dim = bert_config.hidden_size

        # Modality projector
        base_proj_cfg = ProjectorConfig(
            acoustic_dim=acoustic_dim,
            semantic_dim=semantic_dim,
        )
        if projector_cfg_overrides:
            for k, v in projector_cfg_overrides.items():
                if hasattr(base_proj_cfg, k):
                    setattr(base_proj_cfg, k, v)
        self.projector = ModalityProjector(base_proj_cfg)

        # CTC head maps projector output to tokenizer vocabulary
        self.ctc_head = nn.Linear(semantic_dim, ctc_vocab_size)
        self.ctc_blank_id = int(ctc_blank_id)

    def forward(
        self,
        input_features: Tensor,
        audio_attention_mask: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        """
        Args:
            input_features: (B, T_acoustic, n_mels) from WhisperFeatureExtractor.
            audio_attention_mask: (B, T_acoustic) with 1 = valid, 0 = padding.

        Returns:
            {
              "classification_logits": (B, num_labels),
              "ctc_logits": (B, T_semantic, vocab_size),
              "projected_attention_mask": (B, T_semantic),
            }
        """
        # Whisper encoder
        enc_outputs = self.audio_encoder(
            input_features=input_features,
            attention_mask=audio_attention_mask,
        )
        hidden_states: Tensor = enc_outputs.last_hidden_state  # (B, T_acoustic, d_model)

        # Project to semantic space and downsample in time
        soft_embeddings, projected_attention_mask = self.projector(
            hidden_states, attention_mask=audio_attention_mask
        )  # (B, T_semantic, semantic_dim), (B, T_semantic)

        # Semantic decoder using inputs_embeds to preserve differentiability.
        # We do not pass an attention_mask here to avoid shape mismatches
        # between Whisper time steps and the projected sequence length.
        decoder_outputs = self.semantic_decoder(
            inputs_embeds=soft_embeddings,
            return_dict=True,
        )
        classification_logits: Tensor = decoder_outputs.logits  # (B, num_labels)

        # CTC logits over tokenizer vocabulary
        ctc_logits: Tensor = self.ctc_head(soft_embeddings)  # (B, T_semantic, vocab)

        return {
            "classification_logits": classification_logits,
            "ctc_logits": ctc_logits,
            "projected_attention_mask": projected_attention_mask,
        }

