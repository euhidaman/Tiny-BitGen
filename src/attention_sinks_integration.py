"""
Attention Sinks Integration for BitMar
Implements streaming attention with attention sinks for endless fluent generation
while preserving all BitMar features including episodic memory and quantization
"""

import math
import types
from dataclasses import dataclass
from typing import Callable, Optional, Tuple, Dict, Any

import torch
import torch.nn.functional as F
from torch import nn


def slice1d(x, start, end):
    """Slice tensor along dimension 1"""
    return x[:, start:end, ...]


def slice2d(x, start, end):
    """Slice tensor along dimension 2"""
    return x[:, :, start:end, ...]


def slice3d(x, start, end):
    """Slice tensor along dimension 3"""
    return x[:, :, :, start:end, ...]


DIM_TO_SLICE = {
    1: slice1d,
    2: slice2d,
    3: slice3d,
}


@dataclass
class AttentionSinkKVCache:
    """
    Attention Sink KV Cache for maintaining constant memory usage
    during long sequence generation while preserving fluency
    """
    attention_sink_size: int = 4
    attention_sink_window_size: int = 1020
    k_seq_dim: int = 2
    v_seq_dim: int = 2

    def __post_init__(self):
        self.cache_size = self.attention_sink_size + self.attention_sink_window_size
        self.k_slice = DIM_TO_SLICE[self.k_seq_dim]
        self.v_slice = DIM_TO_SLICE[self.v_seq_dim]

    def __call__(self, past_key_values):
        """Apply attention sink cache to past key values"""
        if past_key_values is None:
            return None

        seq_len = past_key_values[0][0].size(self.k_seq_dim)
        if seq_len <= self.cache_size:
            return past_key_values

        return [
            [
                torch.cat(
                    [
                        self.k_slice(k, 0, self.attention_sink_size),
                        self.k_slice(k, seq_len - self.attention_sink_window_size, seq_len),
                    ],
                    dim=self.k_seq_dim,
                ),
                torch.cat(
                    [
                        self.v_slice(v, 0, self.attention_sink_size),
                        self.v_slice(v, seq_len - self.attention_sink_window_size, seq_len),
                    ],
                    dim=self.v_seq_dim,
                ),
            ]
            for k, v in past_key_values
        ]

    def evict_for_space(self, past_key_values, num_coming):
        """Evict KV cache entries to make space for new tokens"""
        if past_key_values is None:
            return None

        seq_len = past_key_values[0][0].size(self.k_seq_dim)
        if seq_len + num_coming <= self.cache_size:
            return past_key_values

        return [
            [
                torch.cat(
                    [
                        self.k_slice(k, 0, self.attention_sink_size),
                        self.k_slice(
                            k,
                            seq_len - self.attention_sink_window_size + num_coming,
                            seq_len,
                        ),
                    ],
                    dim=self.k_seq_dim,
                ),
                torch.cat(
                    [
                        self.v_slice(v, 0, self.attention_sink_size),
                        self.v_slice(
                            v,
                            seq_len - self.attention_sink_window_size + num_coming,
                            seq_len,
                        ),
                    ],
                    dim=self.v_seq_dim,
                ),
            ]
            for k, v in past_key_values
        ]

    def evict_range(self, past_key_values, start, end):
        """Evict a specific range from KV cache"""
        if past_key_values is None:
            return None

        seq_len = past_key_values[0][0].size(self.k_seq_dim)
        assert start <= end and end <= seq_len

        return [
            [
                torch.cat(
                    [
                        self.k_slice(k, 0, start),
                        self.k_slice(k, end, seq_len),
                    ],
                    dim=self.k_seq_dim,
                ),
                torch.cat(
                    [
                        self.v_slice(v, 0, start),
                        self.v_slice(v, end, seq_len),
                    ],
                    dim=self.v_seq_dim,
                ),
            ]
            for k, v in past_key_values
        ]


class BitMarAttentionSinksMixin:
    """
    Mixin class to add attention sinks capability to BitMar models
    while preserving all existing functionality
    """

    def __init__(self, *args, **kwargs):
        # Extract attention sink parameters
        self.attention_sink_size = kwargs.pop('attention_sink_size', 4)
        self.attention_sink_window_size = kwargs.pop('attention_sink_window_size', 1020)
        self.enable_attention_sinks = kwargs.pop('enable_attention_sinks', True)

        super().__init__(*args, **kwargs)

        if self.enable_attention_sinks:
            self._setup_attention_sinks()

    def _setup_attention_sinks(self):
        """Setup attention sinks for the model"""
        # Initialize attention sink cache for text encoder
        if hasattr(self, 'text_encoder'):
            self._inject_attention_sinks_to_encoder(self.text_encoder)

        # Initialize attention sink cache for text decoder if it exists
        if hasattr(self, 'text_decoder'):
            self._inject_attention_sinks_to_encoder(self.text_decoder)

    def _inject_attention_sinks_to_encoder(self, encoder):
        """Inject attention sinks into a transformer encoder"""
        if not hasattr(encoder, 'layers'):
            return

        for layer in encoder.layers:
            if hasattr(layer, 'self_attn'):
                # Create attention sink cache for this layer
                layer.attention_sink_kv_cache = AttentionSinkKVCache(
                    attention_sink_size=self.attention_sink_size,
                    attention_sink_window_size=self.attention_sink_window_size,
                    k_seq_dim=2,  # Standard for transformers
                    v_seq_dim=2
                )

                # Wrap the attention layer's forward method
                self._wrap_attention_forward(layer.self_attn, layer.attention_sink_kv_cache)

    def _wrap_attention_forward(self, attention_module, kv_cache):
        """Wrap attention forward pass to use attention sinks"""
        original_forward = attention_module.forward

        def attention_sinks_forward(self, *args, **kwargs):
            # Call original attention forward
            outputs = original_forward(*args, **kwargs)

            # Apply attention sink cache if past_key_values exist
            if hasattr(outputs, 'past_key_values') and outputs.past_key_values is not None:
                outputs.past_key_values = kv_cache(outputs.past_key_values)
            elif isinstance(outputs, tuple) and len(outputs) > 1:
                # Handle tuple outputs where past_key_values might be at index 1
                output_list = list(outputs)
                if output_list[1] is not None:
                    output_list[1] = kv_cache(output_list[1])
                outputs = tuple(output_list)

            return outputs

        # Replace the forward method
        attention_module.forward = types.MethodType(attention_sinks_forward, attention_module)

    def apply_rotary_pos_emb_with_sinks(self, q, k, cos, sin, position_ids, past_kv_len=0):
        """
        Apply rotary positional embedding with attention sinks support
        Handles position shifting for the sliding window
        """
        if not self.enable_attention_sinks:
            # Fall back to standard rotary embedding
            return self._apply_standard_rotary_pos_emb(q, k, cos, sin, position_ids)

        # For attention sinks, we need to handle position shifting
        cache_size = self.attention_sink_size + self.attention_sink_window_size

        # Query positions are clamped to cache size
        query_position_ids = torch.clamp(position_ids, max=cache_size - 1)

        # Apply rotary embedding to queries with clamped positions
        q_cos = cos[query_position_ids].unsqueeze(1) if cos.dim() > 2 else cos
        q_sin = sin[query_position_ids].unsqueeze(1) if sin.dim() > 2 else sin
        q_embed = (q * q_cos) + (self._rotate_half(q) * q_sin)

        # For keys, use actual positions in cache
        kv_seq_len = k.shape[-2] + past_kv_len
        key_position_ids = torch.arange(kv_seq_len, device=position_ids.device)

        k_cos = cos[key_position_ids].unsqueeze(1) if cos.dim() > 2 else cos
        k_sin = sin[key_position_ids].unsqueeze(1) if sin.dim() > 2 else sin
        k_embed = (k * k_cos) + (self._rotate_half(k) * k_sin)

        return q_embed, k_embed

    def _apply_standard_rotary_pos_emb(self, q, k, cos, sin, position_ids):
        """Standard rotary positional embedding for fallback"""
        cos = cos[position_ids].unsqueeze(1) if cos.dim() > 2 else cos
        sin = sin[position_ids].unsqueeze(1) if sin.dim() > 2 else sin

        q_embed = (q * cos) + (self._rotate_half(q) * sin)
        k_embed = (k * cos) + (self._rotate_half(k) * sin)

        return q_embed, k_embed

    def _rotate_half(self, x):
        """Rotate half the hidden dims of the input"""
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    def get_attention_sinks_stats(self):
        """Get statistics about attention sinks usage"""
        if not self.enable_attention_sinks:
            return {"attention_sinks_enabled": False}

        stats = {
            "attention_sinks_enabled": True,
            "attention_sink_size": self.attention_sink_size,
            "attention_sink_window_size": self.attention_sink_window_size,
            "cache_size": self.attention_sink_size + self.attention_sink_window_size,
        }

        # Count number of attention layers with sinks
        sink_layers = 0
        if hasattr(self, 'text_encoder') and hasattr(self.text_encoder, 'layers'):
            for layer in self.text_encoder.layers:
                if hasattr(layer, 'attention_sink_kv_cache'):
                    sink_layers += 1

        if hasattr(self, 'text_decoder') and hasattr(self.text_decoder, 'layers'):
            for layer in self.text_decoder.layers:
                if hasattr(layer, 'attention_sink_kv_cache'):
                    sink_layers += 1

        stats["layers_with_attention_sinks"] = sink_layers

        return stats


def update_model_kwargs_for_generation_with_sinks(
    self, outputs, model_kwargs, is_encoder_decoder=False, standardize_cache_format=False
):
    """
    Updated model kwargs for generation that handles attention sinks properly
    This prevents indexing errors when generating with attention sinks
    """
    # Update past_key_values
    if "past_key_values" in outputs:
        model_kwargs["past_key_values"] = outputs.past_key_values
    elif "mems" in outputs:
        model_kwargs["past_key_values"] = outputs.mems
    elif "past_buckets_states" in outputs:
        model_kwargs["past_key_values"] = outputs.past_buckets_states
    else:
        model_kwargs["past_key_values"] = None

    # Update token_type_ids with last value
    if "token_type_ids" in model_kwargs:
        token_type_ids = model_kwargs["token_type_ids"]
        model_kwargs["token_type_ids"] = torch.cat([token_type_ids, token_type_ids[:, -1].unsqueeze(-1)], dim=-1)

    if not is_encoder_decoder:
        # Update attention_mask for attention sinks
        if "attention_mask" in model_kwargs:
            attention_mask = model_kwargs["attention_mask"]
            # For attention sinks, we maintain the attention mask size based on cache size
            if hasattr(self, 'enable_attention_sinks') and self.enable_attention_sinks:
                cache_size = getattr(self, 'attention_sink_size', 4) + getattr(self, 'attention_sink_window_size', 1020)
                current_length = attention_mask.shape[-1]

                if current_length >= cache_size:
                    # Maintain cache size for attention mask
                    model_kwargs["attention_mask"] = torch.cat([
                        attention_mask[:, :getattr(self, 'attention_sink_size', 4)],  # Keep sink tokens
                        attention_mask[:, -(getattr(self, 'attention_sink_window_size', 1020) - 1):],  # Keep window
                        torch.ones((attention_mask.shape[0], 1), dtype=attention_mask.dtype, device=attention_mask.device)  # New token
                    ], dim=-1)
                else:
                    # Standard behavior when not at cache limit
                    model_kwargs["attention_mask"] = torch.cat(
                        [attention_mask, torch.ones((attention_mask.shape[0], 1), dtype=attention_mask.dtype, device=attention_mask.device)], dim=-1
                    )
            else:
                # Standard behavior without attention sinks
                model_kwargs["attention_mask"] = torch.cat(
                    [attention_mask, torch.ones((attention_mask.shape[0], 1), dtype=attention_mask.dtype, device=attention_mask.device)], dim=-1
                )

    return model_kwargs


class AttentionSinksConfig:
    """Configuration for attention sinks integration"""

    def __init__(
        self,
        enable_attention_sinks: bool = True,
        attention_sink_size: int = 4,
        attention_sink_window_size: int = 1020,
        inject_to_text_encoder: bool = True,
        inject_to_text_decoder: bool = True,
        position_shift_enabled: bool = True,
    ):
        self.enable_attention_sinks = enable_attention_sinks
        self.attention_sink_size = attention_sink_size
        self.attention_sink_window_size = attention_sink_window_size
        self.inject_to_text_encoder = inject_to_text_encoder
        self.inject_to_text_decoder = inject_to_text_decoder
        self.position_shift_enabled = position_shift_enabled

    @property
    def cache_size(self):
        return self.attention_sink_size + self.attention_sink_window_size

    def to_dict(self):
        return {
            'enable_attention_sinks': self.enable_attention_sinks,
            'attention_sink_size': self.attention_sink_size,
            'attention_sink_window_size': self.attention_sink_window_size,
            'inject_to_text_encoder': self.inject_to_text_encoder,
            'inject_to_text_decoder': self.inject_to_text_decoder,
            'position_shift_enabled': self.position_shift_enabled,
        }

    @classmethod
    def from_dict(cls, config_dict):
        return cls(**config_dict)


def apply_attention_sinks_to_bitmar_model(model, attention_sinks_config: AttentionSinksConfig):
    """
    Apply attention sinks to an existing BitMar model
    This function can be used to retrofit attention sinks to existing models
    """
    if not attention_sinks_config.enable_attention_sinks:
        return model

    # Add attention sinks attributes
    model.attention_sink_size = attention_sinks_config.attention_sink_size
    model.attention_sink_window_size = attention_sinks_config.attention_sink_window_size
    model.enable_attention_sinks = True

    # Inject to text encoder
    if attention_sinks_config.inject_to_text_encoder and hasattr(model, 'text_encoder'):
        _inject_attention_sinks_to_transformer(model.text_encoder, attention_sinks_config)

    # Inject to text decoder
    if attention_sinks_config.inject_to_text_decoder and hasattr(model, 'text_decoder'):
        _inject_attention_sinks_to_transformer(model.text_decoder, attention_sinks_config)

    # Update generation method for attention sinks
    model._update_model_kwargs_for_generation = types.MethodType(
        update_model_kwargs_for_generation_with_sinks, model
    )

    return model


def _inject_attention_sinks_to_transformer(transformer, config: AttentionSinksConfig):
    """Inject attention sinks into a transformer module"""
    if not hasattr(transformer, 'layers'):
        return

    for layer in transformer.layers:
        if hasattr(layer, 'self_attn'):
            # Create attention sink cache
            layer.attention_sink_kv_cache = AttentionSinkKVCache(
                attention_sink_size=config.attention_sink_size,
                attention_sink_window_size=config.attention_sink_window_size,
                k_seq_dim=2,
                v_seq_dim=2
            )

            # Wrap attention forward
            _wrap_attention_forward_with_sinks(layer.self_attn, layer.attention_sink_kv_cache)


def _wrap_attention_forward_with_sinks(attention_module, kv_cache):
    """Wrap attention forward method with attention sinks"""
    original_forward = attention_module.forward

    def attention_sinks_forward(*args, **kwargs):
        # Call original attention
        outputs = original_forward(*args, **kwargs)

        # Apply attention sink cache
        if hasattr(outputs, 'past_key_values') and outputs.past_key_values is not None:
            outputs.past_key_values = kv_cache(outputs.past_key_values)
        elif isinstance(outputs, tuple) and len(outputs) > 1:
            output_list = list(outputs)
            if output_list[1] is not None:
                output_list[1] = kv_cache(output_list[1])
            outputs = tuple(output_list)

        return outputs

    attention_module.forward = attention_sinks_forward
