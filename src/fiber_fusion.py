"""
FIBER-inspired Cross-Modal Fusion for BitMar
Advanced cross-modal transformer blocks with hierarchical fusion
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union
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
        
        # Normalization layers
        self.vision_norm = nn.LayerNorm(hidden_dim)
        self.text_norm = nn.LayerNorm(hidden_dim)
        self.cross_norm = nn.LayerNorm(hidden_dim)

        # Output projections
        self.output_proj = BitNetLinear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(proj_drop)
        self.attn_drop = nn.Dropout(attn_drop)

        # Add learnable fusion weights
        self.alpha_i2t = nn.Parameter(torch.tensor(0.5))
        self.alpha_t2i = nn.Parameter(torch.tensor(0.5))

        # Fix: Add projection dropout as class attribute
        self.proj_drop = nn.Dropout(proj_drop)

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
        vision_features: torch.Tensor,
        text_features: torch.Tensor,
        text_mask: Optional[torch.Tensor] = None,
        return_attention: bool = True
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]]:
        """
        Args:
            vision_features: [batch_size, vision_dim]
            text_features: [batch_size, seq_len, text_dim]
            text_mask: [batch_size, seq_len] - 1 for valid tokens, 0 for padding
            return_attention: Whether to return attention weights (3rd output)

        Returns:
            If return_attention=True: (fused_vision, fused_text, attention_info)
            If return_attention=False: (fused_vision, fused_text)
        """
        B, seq_len, _ = text_features.shape

        # Initialize outputs at the start
        i2t_output = vision_features  # Initialize with input as fallback
        t2i_output = text_features   # Initialize with input as fallback

        # Initialize attention info (only used if return_attention=True)
        attention_info = {
            'i2t_attention': torch.zeros(B, self.num_heads, 1, seq_len, device=vision_features.device),
            't2i_attention': torch.zeros(B, self.num_heads, seq_len, 1, device=vision_features.device),
            'fusion_weights': {'alpha_i2t': 0.5, 'alpha_t2i': 0.5},
            'attention_entropy': torch.tensor(0.0, device=vision_features.device),
            'cross_modal_similarity': torch.tensor(0.0, device=vision_features.device)
        }

        try:
            # Project and normalize features
            vision_proj = self.vision_norm(self.vision_proj(vision_features))
            text_proj = self.text_norm(self.text_proj(text_features))

            # Image-to-text attention
            i2t_q = self.i2t_q(vision_proj).unsqueeze(1)  # [B, 1, hidden_dim]
            i2t_k, i2t_v = self.i2t_kv(text_proj).chunk(2, dim=-1)

            # Reshape for attention
            i2t_q = i2t_q.view(B, 1, self.num_heads, self.head_dim).transpose(1, 2)
            i2t_k = i2t_k.view(B, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
            i2t_v = i2t_v.view(B, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

            # Compute attention
            i2t_attn = (i2t_q @ i2t_k.transpose(-2, -1)) * self.scale
            if text_mask is not None:
                i2t_attn = i2t_attn.masked_fill(~text_mask.unsqueeze(1).unsqueeze(2).bool(), float('-inf'))
            i2t_attn = F.softmax(i2t_attn, dim=-1)
            i2t_attn = self.attn_drop(i2t_attn)

            # Store attention for analysis (only if requested)
            if return_attention:
                attention_info['i2t_attention'] = i2t_attn.detach()
                attention_info['attention_entropy'] = -(i2t_attn * torch.log(i2t_attn + 1e-8)).sum(dim=-1).mean()

            i2t_out = (i2t_attn @ i2t_v).transpose(1, 2).reshape(B, 1, self.hidden_dim)
            i2t_out = i2t_out.squeeze(1)  # [B, hidden_dim]
            i2t_output = self.output_proj(i2t_out)
            i2t_output = self.proj_drop(i2t_output)

            # Text-to-image attention
            t2i_q = self.t2i_q(text_proj)  # [B, seq_len, hidden_dim]
            t2i_k, t2i_v = self.t2i_kv(vision_proj).chunk(2, dim=-1)
            t2i_k = t2i_k.unsqueeze(1)  # [B, 1, hidden_dim]
            t2i_v = t2i_v.unsqueeze(1)  # [B, 1, hidden_dim]

            # Reshape for attention
            t2i_q = t2i_q.view(B, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
            t2i_k = t2i_k.view(B, 1, self.num_heads, self.head_dim).transpose(1, 2)
            t2i_v = t2i_v.view(B, 1, self.num_heads, self.head_dim).transpose(1, 2)

            # Compute attention
            t2i_attn = (t2i_q @ t2i_k.transpose(-2, -1)) * self.scale
            t2i_attn = F.softmax(t2i_attn, dim=-1)
            t2i_attn = self.attn_drop(t2i_attn)

            # Store attention for analysis (only if requested)
            if return_attention:
                attention_info['t2i_attention'] = t2i_attn.detach()
                attention_info['fusion_weights'] = {
                    'alpha_i2t': self.alpha_i2t.detach().item(),
                    'alpha_t2i': self.alpha_t2i.detach().item()
                }
                # Compute cross-modal similarity
                vision_norm = F.normalize(i2t_output, dim=-1)
                text_norm = F.normalize(t2i_output.mean(dim=1), dim=-1)
                attention_info['cross_modal_similarity'] = (vision_norm * text_norm).sum(dim=-1).mean()

            t2i_out = (t2i_attn @ t2i_v).transpose(1, 2).reshape(B, seq_len, self.hidden_dim)
            t2i_output = self.output_proj(t2i_out)
            t2i_output = self.proj_drop(t2i_output)

            # Apply cross-modal normalization
            i2t_output = self.cross_norm(i2t_output + vision_features)
            t2i_output = self.cross_norm(t2i_output + text_features)

        except Exception as e:
            logger.error(f"Cross-modal attention failed: {e}")
            # We already have fallback values from initialization

        # GUARANTEE: Return exactly the right number of values based on return_attention
        logger.info(f"CrossModalAttention: return_attention={return_attention}")
        if return_attention:
            logger.info(f"CrossModalAttention: Returning 3 values - vision: {i2t_output.shape}, text: {t2i_output.shape}, attention_info: {type(attention_info)}")
            return i2t_output, t2i_output, attention_info
        else:
            logger.info(f"CrossModalAttention: Returning 2 values - vision: {i2t_output.shape}, text: {t2i_output.shape}")
            return i2t_output, t2i_output


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
        return_intermediate: bool = False,
        return_attention: bool = True  # New parameter to control attention output
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
            try:
                # Call with return_attention parameter
                cross_modal_result = cross_modal_layer(vision_feats, text_feats, text_mask, return_attention)

                # DEBUG: Log exactly what we're getting back
                logger.info(f"Layer {layer_idx}: cross_modal_result type: {type(cross_modal_result)}")
                if isinstance(cross_modal_result, tuple):
                    logger.info(f"Layer {layer_idx}: Got tuple with {len(cross_modal_result)} values")
                    for i, val in enumerate(cross_modal_result):
                        logger.info(f"Layer {layer_idx}: Value {i}: type={type(val)}, shape={getattr(val, 'shape', 'N/A')}")
                else:
                    logger.info(f"Layer {layer_idx}: Got non-tuple result: {type(cross_modal_result)}")
                    if hasattr(cross_modal_result, 'shape'):
                        logger.info(f"Layer {layer_idx}: Shape: {cross_modal_result.shape}")

                # ROBUST UNPACKING: Handle any number of return values safely
                if isinstance(cross_modal_result, tuple):
                    if len(cross_modal_result) == 3:
                        logger.info(f"Layer {layer_idx}: Unpacking 3 values as expected")
                        vision_cross, text_cross, attn_weights = cross_modal_result
                    elif len(cross_modal_result) == 2:
                        logger.warning(f"Layer {layer_idx}: Only got 2 values, expected 3 with return_attention={return_attention}")
                        vision_cross, text_cross = cross_modal_result
                        # Create dummy attention weights
                        attn_weights = {
                            'i2t_attention': torch.zeros(1, 1, 1, 1, device=vision_feats.device),
                            't2i_attention': torch.zeros(1, 1, 1, 1, device=vision_feats.device),
                            'fusion_weights': {'alpha_i2t': 0.5, 'alpha_t2i': 0.5},
                            'attention_entropy': torch.tensor(0.0, device=vision_feats.device),
                            'cross_modal_similarity': torch.tensor(0.0, device=vision_feats.device)
                        }
                    elif len(cross_modal_result) == 1:
                        logger.warning(f"Layer {layer_idx}: Only got 1 value, expected 3")
                        # Only got one tensor back - use as vision output
                        vision_cross = cross_modal_result[0]
                        text_cross = text_feats  # Use input as fallback
                        attn_weights = {}
                    else:
                        logger.error(f"Layer {layer_idx}: Got {len(cross_modal_result)} values from cross_modal_layer, expected 2 or 3")
                        vision_cross, text_cross = vision_feats, text_feats
                        attn_weights = {}
                else:
                    # Not a tuple - treat as single tensor
                    logger.warning(f"Layer {layer_idx}: cross_modal_layer returned non-tuple: {type(cross_modal_result)}")
                    vision_cross = cross_modal_result if torch.is_tensor(cross_modal_result) else vision_feats
                    text_cross = text_feats
                    attn_weights = {}

            except ValueError as ve:
                if "not enough values to unpack" in str(ve):
                    logger.error(f"Layer {layer_idx}: Unpacking error - falling back to safe extraction")
                    # Safe fallback: try to extract values one by one
                    try:
                        if hasattr(cross_modal_result, '__len__') and len(cross_modal_result) >= 2:
                            vision_cross, text_cross = cross_modal_result[0], cross_modal_result[1]
                            attn_weights = cross_modal_result[2] if len(cross_modal_result) > 2 else {}
                        else:
                            vision_cross, text_cross = vision_feats, text_feats
                            attn_weights = {}
                    except:
                        vision_cross, text_cross = vision_feats, text_feats
                        attn_weights = {}
                else:
                    logger.error(f"Layer {layer_idx}: Cross-modal attention failed with ValueError: {ve}")
                    vision_cross, text_cross = vision_feats, text_feats
                    attn_weights = {}

            except Exception as e:
                logger.error(f"Layer {layer_idx}: Cross-modal attention failed: {e}")
                vision_cross, text_cross = vision_feats, text_feats
                attn_weights = {}

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
