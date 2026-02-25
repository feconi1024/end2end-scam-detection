from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch import Tensor, nn
import torch.nn.functional as F


@dataclass
class ProjectorConfig:
    acoustic_dim: int
    semantic_dim: int
    conv_kernel_size: int = 3
    conv_stride: int = 4
    conv_padding: int = 1
    conv_dilation: int = 1


class ModalityProjector(nn.Module):
    """
    Differentiable bridge between acoustic and semantic spaces.

    Input:
        hidden_states: (B, T_acoustic, acoustic_dim)
        attention_mask: (B, T_acoustic) with 1 = valid, 0 = padding

    Steps:
        1) Linear projection to semantic_dim.
        2) Temporal downsampling via Conv1d with stride k.
        3) GELU activation.
        4) Attention mask is downsampled using max-pooling with the same
           kernel/stride/padding so that a position is valid if ANY source
           position in its receptive field was valid.

    Output:
        soft_embeddings: (B, T_semantic, semantic_dim)
        downsampled_attention_mask: (B, T_semantic)
    """

    def __init__(self, config: ProjectorConfig) -> None:
        super().__init__()
        self.config = config

        self.linear = nn.Linear(config.acoustic_dim, config.semantic_dim)
        self.conv = nn.Conv1d(
            in_channels=config.semantic_dim,
            out_channels=config.semantic_dim,
            kernel_size=config.conv_kernel_size,
            stride=config.conv_stride,
            padding=config.conv_padding,
            dilation=config.conv_dilation,
        )
        self.activation = nn.GELU()

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """
        Args:
            hidden_states: (B, T_acoustic, acoustic_dim)
            attention_mask: (B, T_acoustic) or None

        Returns:
            soft_embeddings: (B, T_semantic, semantic_dim)
            downsampled_attention_mask: (B, T_semantic) or None
        """
        # Project to semantic dimension
        x = self.linear(hidden_states)  # (B, T_acoustic, semantic_dim)

        # Conv1d expects (B, C, T)
        x = x.transpose(1, 2)  # (B, C=semantic_dim, T_acoustic)
        x = self.conv(x)  # (B, semantic_dim, T_semantic)
        x = x.transpose(1, 2)  # (B, T_semantic, semantic_dim)
        x = self.activation(x)

        downsampled_mask: Optional[Tensor]
        if attention_mask is not None:
            # attention_mask: (B, T_acoustic) -> (B, 1, T_acoustic)
            mask = attention_mask.unsqueeze(1).float()
            # Use max-pooling to propagate "any valid" within receptive field.
            mask = F.max_pool1d(
                mask,
                kernel_size=self.config.conv_kernel_size,
                stride=self.config.conv_stride,
                padding=self.config.conv_padding,
            )  # (B, 1, T_semantic)
            downsampled_mask = (mask > 0.0).long().squeeze(1)  # (B, T_semantic)
        else:
            downsampled_mask = None

        return x, downsampled_mask

