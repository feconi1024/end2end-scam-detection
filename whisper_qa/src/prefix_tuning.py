from __future__ import annotations

from types import MethodType
from typing import Any, Dict, Tuple

import torch
from torch import nn
from transformers.cache_utils import EncoderDecoderCache


class WhisperPrefixTuningManager(nn.Module):
    def __init__(
        self,
        encoder_layers: int,
        decoder_layers: int,
        encoder_heads: int,
        decoder_heads: int,
        d_model: int,
        encoder_prefix_length: int,
        decoder_prefix_length: int,
        init_std: float = 0.02,
    ):
        super().__init__()
        self.encoder_layers = encoder_layers
        self.decoder_layers = decoder_layers
        self.encoder_heads = encoder_heads
        self.decoder_heads = decoder_heads
        self.d_model = d_model
        self.encoder_prefix_length = encoder_prefix_length
        self.decoder_prefix_length = decoder_prefix_length
        self.head_dim = d_model // encoder_heads
        self.init_std = float(init_std)

        self.encoder_key_prefix = nn.ParameterList(
            [self._make_prefix(encoder_heads, encoder_prefix_length) for _ in range(encoder_layers)]
        )
        self.encoder_value_prefix = nn.ParameterList(
            [self._make_prefix(encoder_heads, encoder_prefix_length) for _ in range(encoder_layers)]
        )
        self.decoder_key_prefix = nn.ParameterList(
            [self._make_prefix(decoder_heads, decoder_prefix_length) for _ in range(decoder_layers)]
        )
        self.decoder_value_prefix = nn.ParameterList(
            [self._make_prefix(decoder_heads, decoder_prefix_length) for _ in range(decoder_layers)]
        )

    def _make_prefix(self, num_heads: int, prefix_length: int) -> nn.Parameter:
        tensor = torch.empty(1, num_heads, prefix_length, self.head_dim)
        nn.init.normal_(tensor, mean=0.0, std=self.init_std)
        return nn.Parameter(tensor)

    def get_prefix(
        self,
        role: str,
        layer_idx: int,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if role == "encoder":
            key = self.encoder_key_prefix[layer_idx]
            value = self.encoder_value_prefix[layer_idx]
        elif role == "decoder":
            key = self.decoder_key_prefix[layer_idx]
            value = self.decoder_value_prefix[layer_idx]
        else:
            raise ValueError(f"Unsupported prefix role: {role}")

        return (
            key.to(device=device, dtype=dtype).expand(batch_size, -1, -1, -1),
            value.to(device=device, dtype=dtype).expand(batch_size, -1, -1, -1),
        )

    def summary(self) -> Dict[str, int]:
        return {
            "encoder_layers": self.encoder_layers,
            "decoder_layers": self.decoder_layers,
            "encoder_prefix_length": self.encoder_prefix_length,
            "decoder_prefix_length": self.decoder_prefix_length,
            "trainable_params": count_trainable_parameters(self),
        }

    def attach(self, whisper_model: Any) -> None:
        whisper_model.whisper_qa_prefix_manager = self

        for layer_idx, layer in enumerate(whisper_model.model.encoder.layers):
            _patch_attention_forward(layer.self_attn, manager=self, role="encoder", layer_idx=layer_idx)

        for layer_idx, layer in enumerate(whisper_model.model.decoder.layers):
            _patch_attention_forward(layer.self_attn, manager=self, role="decoder", layer_idx=layer_idx)


def freeze_base_model_parameters(whisper_model: nn.Module) -> None:
    for parameter in whisper_model.parameters():
        parameter.requires_grad = False


def count_trainable_parameters(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad)


def _patch_attention_forward(attention_module: nn.Module, manager: WhisperPrefixTuningManager, role: str, layer_idx: int) -> None:
    if getattr(attention_module, "_whisper_qa_prefix_patched", False):
        return

    def _forward_with_prefix(
        self,
        hidden_states: torch.Tensor,
        key_value_states: torch.Tensor | None = None,
        past_key_value: EncoderDecoderCache | None = None,
        attention_mask: torch.Tensor | None = None,
        layer_head_mask: torch.Tensor | None = None,
        output_attentions: bool = False,
        cache_position: torch.LongTensor | None = None,
    ):
        is_cross_attention = key_value_states is not None
        bsz, tgt_len, _ = hidden_states.size()

        query_states = self._shape(self.q_proj(hidden_states) * self.scaling, tgt_len, bsz)

        cache_ref = None
        is_updated = False
        if past_key_value is not None:
            is_updated = past_key_value.is_updated.get(self.layer_idx, False)
            if is_cross_attention:
                past_key_value.is_updated[self.layer_idx] = True
                cache_ref = past_key_value.cross_attention_cache
            else:
                cache_ref = past_key_value.self_attention_cache

        current_states = key_value_states if is_cross_attention else hidden_states
        if is_cross_attention and cache_ref is not None and is_updated:
            key_states = cache_ref.key_cache[self.layer_idx]
            value_states = cache_ref.value_cache[self.layer_idx]
        else:
            key_states = self._shape(self.k_proj(current_states), -1, bsz)
            value_states = self._shape(self.v_proj(current_states), -1, bsz)

            apply_prefix = False
            if not is_cross_attention:
                cached_length = cache_ref.get_seq_length(self.layer_idx) if cache_ref is not None else 0
                apply_prefix = cached_length == 0

            if apply_prefix:
                prefix_key, prefix_value = manager.get_prefix(
                    role=role,
                    layer_idx=layer_idx,
                    batch_size=bsz,
                    device=hidden_states.device,
                    dtype=key_states.dtype,
                )
                key_states = torch.cat([prefix_key, key_states], dim=2)
                value_states = torch.cat([prefix_value, value_states], dim=2)

                if attention_mask is not None:
                    prefix_mask = torch.zeros(
                        attention_mask.shape[0],
                        attention_mask.shape[1],
                        attention_mask.shape[2],
                        prefix_key.shape[2],
                        device=attention_mask.device,
                        dtype=attention_mask.dtype,
                    )
                    attention_mask = torch.cat([prefix_mask, attention_mask], dim=-1)

            if cache_ref is not None:
                update_position = cache_position if not is_cross_attention else None
                key_states, value_states = cache_ref.update(
                    key_states,
                    value_states,
                    self.layer_idx,
                    {"cache_position": update_position},
                )

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3))
        if attention_mask is not None:
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask

        attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        if layer_head_mask is not None:
            if layer_head_mask.size() != (self.num_heads,):
                raise ValueError(
                    f"Head mask for a single layer should be of size {(self.num_heads,)}, but is {layer_head_mask.size()}"
                )
            attn_weights = layer_head_mask.view(1, -1, 1, 1) * attn_weights

        attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
        attn_output = torch.matmul(attn_probs, value_states)

        if attn_output.size() != (bsz, self.num_heads, tgt_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, tgt_len, self.head_dim)}, but is {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
        attn_output = self.out_proj(attn_output)
        return attn_output, attn_weights if output_attentions else None, past_key_value

    attention_module._whisper_qa_prefix_patched = True
    attention_module._whisper_qa_prefix_role = role
    attention_module._whisper_qa_prefix_layer_idx = layer_idx
    attention_module.forward = MethodType(_forward_with_prefix, attention_module)
