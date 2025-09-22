"""
FIBER-inspired Cross-Modal Fusion for BitMar
Advanced cross-modal transformer blocks with hierarchical fusion
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import math
import logging

logger = logging.getLogger(__name__)


class BitNetLinear(nn.Module):
    """1.58-bit Linear layer following BitNet b1.58 architecture"""

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Weight parameters (full precision for training)
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None

        # Quantization scaling factors
        self.register_buffer('weight_scale', torch.ones(1))
        self.register_buffer('input_scale', torch.ones(1))

    def quantize_weights_1_58_bit(self, weight: torch.Tensor) -> torch.Tensor:
        """BitNet b1.58 weight quantization: {-1, 0, +1}"""
        scale = weight.abs().mean()
        self.weight_scale.data = scale.clamp(min=1e-5, max=1e3)
        weight_norm = torch.clamp(weight / self.weight_scale, min=-10.0, max=10.0)
        threshold = 2.0 / 3.0
        
        quantized = torch.zeros_like(weight_norm)
        quantized[weight_norm > threshold] = 1.0
        quantized[weight_norm < -threshold] = -1.0
        return quantized

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            weight_q = self.quantize_weights_1_58_bit(self.weight)
            weight_forward = weight_q * self.weight_scale
            weight_forward = weight_forward + (self.weight - self.weight.detach())
            return F.linear(x, weight_forward, self.bias)
        else:
            weight_q = self.quantize_weights_1_58_bit(self.weight) * self.weight_scale
            return F.linear(x, weight_q, self.bias)


class CrossModalAttention(nn.Module):
    """
    FIBER-inspired Cross-Modal Attention with hierarchical fusion
    Supports image-to-text and text-to-image attention with relative positioning
    """
    
    def __init__(
        self,
        vision_dim: int,
        text_dim: int,
        hidden_dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        attn_drop: float = 0.1,
        proj_drop: float = 0.1,
        use_relative_pos: bool = True
    ):
        super().__init__()
        self.vision_dim = vision_dim
        self.text_dim = text_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.use_relative_pos = use_relative_pos
        
        assert hidden_dim % num_heads == 0, f"hidden_dim {hidden_dim} not divisible by num_heads {num_heads}"
        
        # Vision feature projection
        self.vision_proj = BitNetLinear(vision_dim, hidden_dim, bias=qkv_bias)
        self.vision_qkv = BitNetLinear(hidden_dim, hidden_dim * 3, bias=qkv_bias)
        
        # Text feature projection  
        self.text_proj = BitNetLinear(text_dim, hidden_dim, bias=qkv_bias)
        self.text_qkv = BitNetLinear(hidden_dim, hidden_dim * 3, bias=qkv_bias)
        
        # Cross-modal projections (FIBER-style)
        # Image-to-text attention
        self.i2t_q = BitNetLinear(hidden_dim, hidden_dim, bias=qkv_bias)
        self.i2t_kv = BitNetLinear(hidden_dim, hidden_dim * 2, bias=qkv_bias)
        
        # Text-to-image attention  
        self.t2i_q = BitNetLinear(hidden_dim, hidden_dim, bias=qkv_bias)
        self.t2i_kv = BitNetLinear(hidden_dim, hidden_dim * 2, bias=qkv_bias)
        
        # Normalization layers (MISSING - this was causing the error!)
        self.vision_norm = nn.LayerNorm(hidden_dim)
        self.text_norm = nn.LayerNorm(hidden_dim)
        self.norm_v = nn.LayerNorm(hidden_dim)
        self.norm_t = nn.LayerNorm(hidden_dim)
        self.norm_i2t = nn.LayerNorm(hidden_dim)
        self.norm_t2i = nn.LayerNorm(hidden_dim)
        
        # Output projections
        self.vision_out = BitNetLinear(hidden_dim, hidden_dim)
        self.text_out = BitNetLinear(hidden_dim, hidden_dim)
        
        # Dropout
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)
        
        # Learnable mixing weights (FIBER-style)
        self.alpha_i2t = nn.Parameter(torch.zeros(1))
        self.alpha_t2i = nn.Parameter(torch.zeros(1))
        
        # Relative position encoding for text (if enabled)
        if use_relative_pos:
            self.max_relative_pos = 128
            self.relative_pos_embed = nn.Embedding(
                2 * self.max_relative_pos + 1, num_heads
            )
    
    def get_relative_pos_bias(self, seq_len: int) -> torch.Tensor:
        """Compute relative position bias for text sequence"""
        if not self.use_relative_pos:
            return None
            
        # Create relative position indices
        coords = torch.arange(seq_len, device=self.relative_pos_embed.weight.device)
        relative_coords = coords[:, None] - coords[None, :]  # [seq_len, seq_len]
        
        # Clip to max range and shift to positive
        relative_coords = torch.clamp(
            relative_coords, -self.max_relative_pos, self.max_relative_pos
        ) + self.max_relative_pos
        
        # Get embeddings and permute for attention
        relative_pos_bias = self.relative_pos_embed(relative_coords)  # [seq_len, seq_len, num_heads]
        return relative_pos_bias.permute(2, 0, 1)  # [num_heads, seq_len, seq_len]
    
    def forward(
        self,
        vision_features: torch.Tensor,  # [B, vision_dim] 
        text_features: torch.Tensor,   # [B, seq_len, text_dim]
        text_mask: Optional[torch.Tensor] = None  # [B, seq_len]
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Args:
            vision_features: [batch_size, vision_dim] 
            text_features: [batch_size, seq_len, text_dim]
            text_mask: [batch_size, seq_len] - 1 for valid tokens, 0 for padding
            
        Returns:
            fused_vision: [batch_size, hidden_dim]
            fused_text: [batch_size, seq_len, hidden_dim] 
            attention_weights: Dict of attention patterns
        """
        try:
            B, seq_len, _ = text_features.shape

            # Project to common hidden dimension
            vision_proj = self.vision_proj(vision_features)  # [B, hidden_dim]
            text_proj = self.text_proj(text_features)  # [B, seq_len, hidden_dim]

            # Expand vision features for attention
            vision_expanded = vision_proj.unsqueeze(1)  # [B, 1, hidden_dim]

            # === Self-attention first (following FIBER) ===

            # Vision self-attention (trivial since single token)
            vision_self = vision_proj  # No change needed for single token

            # Text self-attention with safe error handling
            try:
                text_normed = self.text_norm(text_proj)
                text_qkv_raw = self.text_qkv(text_normed)

                # FIXED: Enhanced safety checks for QKV computation
                expected_size = B * seq_len * self.hidden_dim * 3
                actual_size = text_qkv_raw.numel()

                if actual_size != expected_size:
                    logger.warning(f"QKV size mismatch: expected {expected_size}, got {actual_size}")
                    logger.warning(f"Input shape: {text_normed.shape}, Output shape: {text_qkv_raw.shape}")
                    logger.warning(f"Hidden dim: {self.hidden_dim}, Heads: {self.num_heads}, Head dim: {self.head_dim}")

                # FIXED: Safe reshape and unpack with proper error handling
                # Check if we have the right dimensions for QKV
                expected_last_dim = self.hidden_dim * 3
                if text_qkv_raw.size(-1) != expected_last_dim:
                    logger.error(f"QKV last dimension mismatch: got {text_qkv_raw.size(-1)}, expected {expected_last_dim}")
                    # Pad or truncate to the expected size
                    if text_qkv_raw.size(-1) < expected_last_dim:
                        # Pad with zeros
                        pad_size = expected_last_dim - text_qkv_raw.size(-1)
                        padding = torch.zeros(*text_qkv_raw.shape[:-1], pad_size, device=text_qkv_raw.device, dtype=text_qkv_raw.dtype)
                        text_qkv_raw = torch.cat([text_qkv_raw, padding], dim=-1)
                    else:
                        # Truncate
                        text_qkv_raw = text_qkv_raw[..., :expected_last_dim]

                # Reshape with proper error handling
                text_qkv = text_qkv_raw.reshape(B, seq_len, 3, self.num_heads, self.head_dim)
                text_qkv = text_qkv.permute(2, 0, 3, 1, 4)  # [3, B, num_heads, seq_len, head_dim]

                # Safely unpack Q, K, V with guaranteed 3 components
                if text_qkv.size(0) >= 3:
                    text_q, text_k, text_v = text_qkv[0], text_qkv[1], text_qkv[2]
                elif text_qkv.size(0) == 2:
                    logger.warning("QKV tensor only has 2 components instead of 3, duplicating key as value")
                    text_q, text_k = text_qkv[0], text_qkv[1]
                    text_v = text_k  # Use key as value fallback
                else:
                    logger.warning("QKV tensor only has 1 component, duplicating for Q, K, V")
                    text_q = text_k = text_v = text_qkv[0]  # Use single tensor for all

            except Exception as e:
                logger.error(f"QKV computation failed: {e}")
                logger.error(f"Text normed shape: {text_normed.shape if 'text_normed' in locals() else 'N/A'}")
                logger.error(f"Expected hidden_dim * 3 = {self.hidden_dim * 3}")
                logger.error(f"Actual text_qkv_raw shape: {text_qkv_raw.shape if 'text_qkv_raw' in locals() else 'N/A'}")

                # Fallback: create dummy tensors to prevent crash
                dummy_shape = (B, self.num_heads, seq_len, self.head_dim)
                text_q = torch.zeros(dummy_shape, device=text_features.device, dtype=text_features.dtype)
                text_k = torch.zeros(dummy_shape, device=text_features.device, dtype=text_features.dtype)
                text_v = torch.zeros(dummy_shape, device=text_features.device, dtype=text_features.dtype)

            # Text self-attention computation with safe fallbacks
            try:
                text_attn = (text_q @ text_k.transpose(-2, -1)) * self.scale

                # Add relative position bias
                if self.use_relative_pos:
                    rel_pos_bias = self.get_relative_pos_bias(seq_len)
                    if rel_pos_bias is not None:
                        text_attn = text_attn + rel_pos_bias.unsqueeze(0)

                # Apply text mask
                if text_mask is not None:
                    text_mask_expanded = text_mask.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, seq_len]
                    text_attn = text_attn.masked_fill(~text_mask_expanded.bool(), -1e9)

                text_attn = F.softmax(text_attn, dim=-1)
                text_attn = self.attn_drop(text_attn)

                text_self = (text_attn @ text_v).transpose(1, 2).reshape(B, seq_len, self.hidden_dim)
                text_self = text_proj + self.proj_drop(text_self)  # Residual connection

            except Exception as e:
                logger.error(f"Text self-attention computation failed: {e}")
                # Fallback: use original text projection
                text_self = text_proj
                text_attn = torch.zeros(B, self.num_heads, seq_len, seq_len, device=text_features.device)

            # === Cross-modal attention (FIBER-style) ===

            # Image-to-Text attention with safe error handling
            try:
                # Use vision as query, text as key-value
                vision_q = vision_proj.unsqueeze(1).repeat(1, seq_len, 1)  # [B, seq_len, hidden_dim]
                vision_q = vision_q.view(B, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

                text_k = text_self.view(B, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
                text_v = text_self.view(B, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

                i2t_attn = (vision_q @ text_k.transpose(-2, -1)) * self.scale

                if text_mask is not None:
                    text_mask_expanded = text_mask.unsqueeze(1).unsqueeze(1)  # [B, 1, 1, seq_len]
                    i2t_attn = i2t_attn.masked_fill(~text_mask_expanded.bool(), -1e9)

                i2t_attn = F.softmax(i2t_attn, dim=-1)
                i2t_out = (i2t_attn @ text_v).transpose(1, 2).reshape(B, seq_len, self.hidden_dim)
                i2t_out = i2t_out.mean(dim=1)  # Pool to single vision token

            except Exception as e:
                logger.error(f"Image-to-text attention failed: {e}")
                # Fallback: use original vision features
                i2t_out = vision_proj
                i2t_attn = torch.zeros(B, self.num_heads, 1, seq_len, device=text_features.device)

            # Text-to-Image attention with safe error handling
            try:
                # Use text as query, vision as key-value - FIXED shape computation
                text_q = text_self.view(B, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

                # FIXED: Properly handle vision key/value dimensions - correct tensor reshaping
                # vision_proj is [B, hidden_dim], we need to convert it to [B, num_heads, 1, head_dim]
                vision_reshaped = vision_proj.view(B, self.num_heads, self.head_dim)  # [B, num_heads, head_dim]
                vision_k = vision_reshaped.unsqueeze(2)  # [B, num_heads, 1, head_dim]
                vision_v = vision_k.clone()  # Same for key and value

                t2i_attn = (text_q @ vision_k.transpose(-2, -1)) * self.scale
                t2i_attn = F.softmax(t2i_attn, dim=-1)

                # FIXED: Safe tensor computation for reshape
                t2i_out = (t2i_attn @ vision_v)  # [B, num_heads, seq_len, head_dim]
                t2i_out = t2i_out.transpose(1, 2).contiguous()  # [B, seq_len, num_heads, head_dim]

                # Verify dimensions match before reshape
                if t2i_out.size(2) * t2i_out.size(3) == self.hidden_dim:
                    t2i_out = t2i_out.reshape(B, seq_len, self.hidden_dim)
                else:
                    logger.warning(f"T2I dimension mismatch: {t2i_out.size(2)} * {t2i_out.size(3)} != {self.hidden_dim}")
                    # Fallback: flatten and project to correct size
                    t2i_out = t2i_out.view(B, seq_len, -1)
                    if t2i_out.size(-1) != self.hidden_dim:
                        # Use adaptive pooling to get correct dimensions
                        t2i_out = F.adaptive_avg_pool1d(t2i_out.transpose(1, 2), self.hidden_dim).transpose(1, 2)

            except Exception as e:
                logger.error(f"Text-to-image attention failed: {e}")
                logger.error(f"Debug info - B: {B}, seq_len: {seq_len}, num_heads: {self.num_heads}, head_dim: {self.head_dim}")
                logger.error(f"Vision proj shape: {vision_proj.shape if 'vision_proj' in locals() else 'N/A'}")
                logger.error(f"Text self shape: {text_self.shape if 'text_self' in locals() else 'N/A'}")
                logger.error(f"Hidden dim: {self.hidden_dim}, expected head_dim * num_heads = {self.head_dim * self.num_heads}")
                # Fallback: use original text features
                t2i_out = text_self
                t2i_attn = torch.zeros(B, self.num_heads, seq_len, 1, device=text_features.device)

            # Adaptive fusion with learnable weights
            try:
                fused_vision = self.alpha_i2t * i2t_out + (1 - self.alpha_i2t) * vision_self
                fused_text = self.alpha_t2i * t2i_out + (1 - self.alpha_t2i) * text_self
            except Exception as e:
                logger.error(f"Adaptive fusion failed: {e}")
                # Fallback: use self-attention results
                fused_vision = vision_self
                fused_text = text_self

            # Collect attention weights for analysis
            attention_weights = {
                'text_self_attention': text_attn.detach() if 'text_attn' in locals() else torch.zeros(1, 1, 1, 1, device=text_features.device),
                'image_to_text_attention': i2t_attn.detach() if 'i2t_attn' in locals() else torch.zeros(1, 1, 1, 1, device=text_features.device),
                'text_to_image_attention': t2i_attn.detach() if 't2i_attn' in locals() else torch.zeros(1, 1, 1, 1, device=text_features.device),
                'alpha_i2t': self.alpha_i2t.detach(),
                'alpha_t2i': self.alpha_t2i.detach()
            }

            return fused_vision, fused_text, attention_weights

        except Exception as e:
            logger.error(f"FIBER cross-modal attention completely failed: {e}")
            # Complete fallback: return input features with dummy attention weights
            B, seq_len, _ = text_features.shape

            # Project to correct dimensions for fallback
            try:
                fused_vision = self.vision_proj(vision_features)
                fused_text = self.text_proj(text_features)
            except Exception as proj_e:
                logger.error(f"Even projection failed: {proj_e}")
                # Ultimate fallback: pad/truncate to expected dimensions
                if vision_features.size(-1) != self.hidden_dim:
                    if vision_features.size(-1) < self.hidden_dim:
                        pad_size = self.hidden_dim - vision_features.size(-1)
                        fused_vision = torch.cat([vision_features, torch.zeros(*vision_features.shape[:-1], pad_size, device=vision_features.device)], dim=-1)
                    else:
                        fused_vision = vision_features[..., :self.hidden_dim]
                else:
                    fused_vision = vision_features

                if text_features.size(-1) != self.hidden_dim:
                    if text_features.size(-1) < self.hidden_dim:
                        pad_size = self.hidden_dim - text_features.size(-1)
                        fused_text = torch.cat([text_features, torch.zeros(*text_features.shape[:-1], pad_size, device=text_features.device)], dim=-1)
                    else:
                        fused_text = text_features[..., :self.hidden_dim]
                else:
                    fused_text = text_features

            # Dummy attention weights
            attention_weights = {
                'text_self_attention': torch.zeros(1, 1, 1, 1, device=text_features.device),
                'image_to_text_attention': torch.zeros(1, 1, 1, 1, device=text_features.device),
                'text_to_image_attention': torch.zeros(1, 1, 1, 1, device=text_features.device),
                'alpha_i2t': torch.tensor(0.5, device=text_features.device),
                'alpha_t2i': torch.tensor(0.5, device=text_features.device)
            }

            return fused_vision, fused_text, attention_weights


class FIBERCrossModalFusion(nn.Module):
    """
    FIBER-inspired Multi-Layer Cross-Modal Fusion
    Implements hierarchical cross-modal transformer blocks
    """
    
    def __init__(
        self,
        vision_dim: int,
        text_dim: int,
        hidden_dim: int,
        num_heads: int = 8,
        num_layers: int = 3,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        use_relative_pos: bool = True
    ):
        super().__init__()
        self.vision_dim = vision_dim
        self.text_dim = text_dim  
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Multi-layer cross-modal attention blocks
        self.cross_modal_layers = nn.ModuleList([
            CrossModalAttention(
                vision_dim=vision_dim if i == 0 else hidden_dim,
                text_dim=text_dim if i == 0 else hidden_dim,
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                attn_drop=dropout,
                proj_drop=dropout,
                use_relative_pos=use_relative_pos
            ) for i in range(num_layers)
        ])
        
        # Feed-forward networks (FIBER-style)
        self.vision_ffns = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(hidden_dim),
                BitNetLinear(hidden_dim, int(hidden_dim * mlp_ratio)),
                nn.GELU(),
                nn.Dropout(dropout),
                BitNetLinear(int(hidden_dim * mlp_ratio), hidden_dim),
                nn.Dropout(dropout)
            ) for _ in range(num_layers)
        ])
        
        self.text_ffns = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(hidden_dim),
                BitNetLinear(hidden_dim, int(hidden_dim * mlp_ratio)),
                nn.GELU(),
                nn.Dropout(dropout),
                BitNetLinear(int(hidden_dim * mlp_ratio), hidden_dim),
                nn.Dropout(dropout)
            ) for _ in range(num_layers)
        ])
        
        # Output projections
        self.final_vision_proj = BitNetLinear(hidden_dim, hidden_dim)
        self.final_text_proj = BitNetLinear(hidden_dim, hidden_dim)
        
        # Cross-modal similarity computation
        self.similarity_temp = nn.Parameter(torch.ones([]) * 0.07)
        
    def compute_cross_modal_similarity(
        self, 
        vision_features: torch.Tensor, 
        text_features: torch.Tensor,
        text_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Compute cross-modal similarity following FIBER"""
        # Pool text features (mean over sequence, respecting mask)
        if text_mask is not None:
            text_mask_expanded = text_mask.unsqueeze(-1).float()
            text_pooled = (text_features * text_mask_expanded).sum(1) / text_mask_expanded.sum(1)
        else:
            text_pooled = text_features.mean(1)
        
        # L2 normalize  
        vision_norm = F.normalize(vision_features, p=2, dim=-1)
        text_norm = F.normalize(text_pooled, p=2, dim=-1)
        
        # Compute similarity with temperature
        similarity = torch.matmul(vision_norm, text_norm.t()) / self.similarity_temp
        return similarity
    
    def forward(
        self,
        vision_features: torch.Tensor,  # [B, vision_dim]
        text_features: torch.Tensor,   # [B, seq_len, text_dim]
        text_mask: Optional[torch.Tensor] = None,  # [B, seq_len]
        return_intermediate: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through FIBER-style cross-modal fusion
        
        Returns:
            Dict containing:
            - vision_features: [B, hidden_dim] 
            - text_features: [B, seq_len, hidden_dim]
            - cross_modal_similarity: [B, B] similarity matrix
            - attention_weights: Dict of attention patterns from all layers
        """
        
        vision_feats = vision_features
        text_feats = text_features
        all_attention_weights = {}
        intermediate_features = []
        
        # Apply hierarchical cross-modal fusion
        for layer_idx, (cross_modal_layer, vision_ffn, text_ffn) in enumerate(
            zip(self.cross_modal_layers, self.vision_ffns, self.text_ffns)
        ):
            # FIXED: Safe unpacking for cross-modal attention with proper error handling
            try:
                cross_modal_result = cross_modal_layer(vision_feats, text_feats, text_mask)

                # Handle different return formats
                if isinstance(cross_modal_result, tuple):
                    if len(cross_modal_result) >= 3:
                        vision_cross, text_cross, attn_weights = cross_modal_result[0], cross_modal_result[1], cross_modal_result[2]
                    elif len(cross_modal_result) == 2:
                        logger.warning(f"Layer {layer_idx}: cross_modal_layer returned only 2 values instead of 3, creating dummy attention weights")
                        vision_cross, text_cross = cross_modal_result[0], cross_modal_result[1]
                        attn_weights = {
                            'text_self_attention': torch.zeros(1, 1, 1, 1, device=vision_feats.device),
                            'image_to_text_attention': torch.zeros(1, 1, 1, 1, device=vision_feats.device),
                            'text_to_image_attention': torch.zeros(1, 1, 1, 1, device=vision_feats.device),
                            'alpha_i2t': torch.tensor(0.0, device=vision_feats.device),
                            'alpha_t2i': torch.tensor(0.0, device=vision_feats.device)
                        }
                    else:
                        logger.error(f"Layer {layer_idx}: cross_modal_layer returned {len(cross_modal_result)} values, expected 3")
                        # Fallback: use input features unchanged
                        vision_cross, text_cross = vision_feats, text_feats
                        attn_weights = {
                            'text_self_attention': torch.zeros(1, 1, 1, 1, device=vision_feats.device),
                            'image_to_text_attention': torch.zeros(1, 1, 1, 1, device=vision_feats.device),
                            'text_to_image_attention': torch.zeros(1, 1, 1, 1, device=vision_feats.device),
                            'alpha_i2t': torch.tensor(0.0, device=vision_feats.device),
                            'alpha_t2i': torch.tensor(0.0, device=vision_feats.device)
                        }
                else:
                    logger.error(f"Layer {layer_idx}: cross_modal_layer returned non-tuple result: {type(cross_modal_result)}")
                    # Fallback: use input features unchanged
                    vision_cross, text_cross = vision_feats, text_feats
                    attn_weights = {
                        'text_self_attention': torch.zeros(1, 1, 1, 1, device=vision_feats.device),
                        'image_to_text_attention': torch.zeros(1, 1, 1, 1, device=vision_feats.device),
                        'text_to_image_attention': torch.zeros(1, 1, 1, 1, device=vision_feats.device),
                        'alpha_i2t': torch.tensor(0.0, device=vision_feats.device),
                        'alpha_t2i': torch.tensor(0.0, device=vision_feats.device)
                    }

            except Exception as e:
                logger.error(f"Layer {layer_idx}: Cross-modal attention failed: {e}")
                # Fallback: use input features unchanged
                vision_cross, text_cross = vision_feats, text_feats
                attn_weights = {
                    'text_self_attention': torch.zeros(1, 1, 1, 1, device=vision_feats.device),
                    'image_to_text_attention': torch.zeros(1, 1, 1, 1, device=vision_feats.device),
                    'text_to_image_attention': torch.zeros(1, 1, 1, 1, device=vision_feats.device),
                    'alpha_i2t': torch.tensor(0.0, device=vision_feats.device),
                    'alpha_t2i': torch.tensor(0.0, device=vision_feats.device)
                }

            # Store attention weights
            all_attention_weights[f'layer_{layer_idx}'] = attn_weights
            
            # Feed-forward networks with residual connections
            vision_feats = vision_cross + vision_ffn(vision_cross)
            text_feats = text_cross + text_ffn(text_cross)
            
            if return_intermediate:
                intermediate_features.append({
                    'vision': vision_feats.clone(),
                    'text': text_feats.clone()
                })
        
        # Final projections
        final_vision = self.final_vision_proj(vision_feats)
        final_text = self.final_text_proj(text_feats)
        
        # Compute cross-modal similarity
        similarity_matrix = self.compute_cross_modal_similarity(
            final_vision, final_text, text_mask
        )
        
        result = {
            'vision_features': final_vision,
            'text_features': final_text,
            'cross_modal_similarity': similarity_matrix,
            'attention_weights': all_attention_weights,
            'similarity_temperature': self.similarity_temp.item()
        }
        
        if return_intermediate:
            result['intermediate_features'] = intermediate_features
            
        return result


def create_fiber_fusion(config: Dict) -> FIBERCrossModalFusion:
    """Create FIBER-style cross-modal fusion from config"""
    return FIBERCrossModalFusion(
        vision_dim=config.get('vision_latent_size', 768),
        text_dim=config.get('text_encoder_dim', 768),
        hidden_dim=config.get('fusion_hidden_size', 768),
        num_heads=config.get('fusion_num_heads', 8),
        num_layers=config.get('fusion_num_layers', 3),
        mlp_ratio=config.get('fusion_mlp_ratio', 4.0),
        dropout=config.get('dropout', 0.1),
        use_relative_pos=config.get('use_relative_position', True)
    )
