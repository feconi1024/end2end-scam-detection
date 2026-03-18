from __future__ import annotations

from typing import Dict, Optional

import torch
from torch import Tensor, nn


class JointCTCSLULoss(nn.Module):
    """
    Composite loss: L_total = lambda_CTC * L_CTC + lambda_SLU * L_CE.
    """

    def __init__(
        self,
        ctc_blank_id: int,
        ctc_weight: float = 0.3,
        slu_weight: float = 0.7,
        class_weights: Optional[Tensor] = None,
    ) -> None:
        super().__init__()
        self.ctc_weight = float(ctc_weight)
        self.slu_weight = float(slu_weight)

        self.ctc_loss_fn = nn.CTCLoss(
            blank=ctc_blank_id,
            zero_infinity=True,
        )
        self.ce_loss_fn = nn.CrossEntropyLoss(
            weight=class_weights,
        )

    def forward(
        self,
        classification_logits: Tensor,
        ctc_logits: Tensor,
        labels: Tensor,
        ctc_targets: Tensor,
        ctc_input_lengths: Tensor,
        ctc_target_lengths: Tensor,
    ) -> Dict[str, Tensor]:
        """
        Args:
            classification_logits: (B, num_labels)
            ctc_logits: (B, T_ctc, vocab)
            labels: (B,)
            ctc_targets: (B, T_text) padded with a non-used token id
            ctc_input_lengths: (B,) lengths in time after projector downsampling
            ctc_target_lengths: (B,) target lengths per example
        """
        # Cross-entropy for classification
        slu_loss = self.ce_loss_fn(classification_logits, labels)

        # CTC loss
        has_ctc_targets = (ctc_target_lengths > 0).any().item()
        if has_ctc_targets and self.ctc_weight > 0.0:
            # Prepare shapes for nn.CTCLoss: (T, N, C)
            log_probs = ctc_logits.log_softmax(dim=-1).transpose(0, 1)  # (T, B, vocab)

            # Flatten padded targets to 1D
            # Only the first target_lengths[i] of each row will be used.
            targets_flat = ctc_targets.contiguous().view(-1)

            ctc_loss = self.ctc_loss_fn(
                log_probs,
                targets_flat,
                ctc_input_lengths,
                ctc_target_lengths,
            )
            total_loss = self.ctc_weight * ctc_loss + self.slu_weight * slu_loss
        else:
            ctc_loss = classification_logits.new_tensor(0.0)
            # No CTC targets: use unweighted classification loss so that the
            # effective learning rate is not silently reduced by slu_weight.
            total_loss = slu_loss

        return {
            "loss": total_loss,
            "ctc_loss": ctc_loss.detach(),
            "slu_loss": slu_loss.detach(),
        }

