from __future__ import annotations

from typing import Any, Dict, Optional

import torch
from torch import Tensor, nn
import torch.nn.functional as F
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

        # Learnable [CLS] embedding for BERT-style pooling on token 0.
        # Without this, BERT would pool from the first acoustic frame.
        self.cls_token = nn.Parameter(torch.zeros(1, 1, semantic_dim))
        nn.init.normal_(self.cls_token, mean=0.0, std=0.02)

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
        hidden_states: Tensor = enc_outputs.last_hidden_state  # (B, T_enc, d_model)

        # Whisper encoder can change the time resolution (e.g., T_enc != T_mel).
        # Ensure the mask we pass into the projector matches hidden_states length.
        encoder_attention_mask: Optional[Tensor]
        if audio_attention_mask is None:
            encoder_attention_mask = None
        else:
            mask = audio_attention_mask
            if mask.dim() != 2:
                raise ValueError(f"audio_attention_mask must be (B, T), got {tuple(mask.shape)}")
            t_mel = mask.size(1)
            t_enc = hidden_states.size(1)
            if t_mel == t_enc:
                encoder_attention_mask = mask
            else:
                # Downsample by max-pooling in time, then crop/pad to t_enc.
                # Commonly, Whisper reduces length by ~2.
                ratio = max(1, int(round(t_mel / float(t_enc))))
                pooled = F.max_pool1d(
                    mask.unsqueeze(1).float(),
                    kernel_size=ratio,
                    stride=ratio,
                    ceil_mode=True,
                ).squeeze(1).long()
                if pooled.size(1) < t_enc:
                    pad = t_enc - pooled.size(1)
                    pooled = F.pad(pooled, (0, pad), value=0)
                encoder_attention_mask = pooled[:, :t_enc]

        # Project to semantic space and downsample in time
        soft_embeddings, projected_attention_mask = self.projector(
            hidden_states, attention_mask=encoder_attention_mask
        )  # (B, T_semantic, semantic_dim), (B, T_semantic)

        # Prepend a learnable [CLS] token so BERT pools from index 0.
        bsz = soft_embeddings.size(0)
        cls = self.cls_token.expand(bsz, -1, -1)  # (B, 1, semantic_dim)
        soft_embeddings_for_bert = torch.cat([cls, soft_embeddings], dim=1)

        if projected_attention_mask is None:
            bert_attention_mask = None
        else:
            cls_mask = projected_attention_mask.new_ones((bsz, 1))
            bert_attention_mask = torch.cat([cls_mask, projected_attention_mask], dim=1)

        # --- ABLATION FIX: Prevent length leakage ---
        # Intentionally do not pass the attention mask to the semantic decoder.
        # This prevents BERT from exploiting exact sequence length as a shortcut.
        decoder_outputs = self.semantic_decoder(
            inputs_embeds=soft_embeddings_for_bert,
            attention_mask=None,
            return_dict=True,
        )
        classification_logits: Tensor = decoder_outputs.logits  # (B, num_labels)

        # CTC logits over tokenizer vocabulary
        ctc_logits: Tensor = self.ctc_head(soft_embeddings)  # (B, T_semantic, vocab)

        return {
            "classification_logits": classification_logits,
            "ctc_logits": ctc_logits,
            # For CTC / length accounting (no CLS)
            "projected_attention_mask": projected_attention_mask,
            # For BERT (with CLS prepended)
            "bert_attention_mask": bert_attention_mask,
        }

