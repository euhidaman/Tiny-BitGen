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
    FIBER-style Cross-Modal Attention following the actual FIBER architecture
    Based on RobertaSelfAttention from the real FIBER repository
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
        use_relative_pos: bool = False  # FIBER doesn't use relative pos by default
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.all_head_size = num_heads * self.head_dim

        assert hidden_dim % num_heads == 0, f"hidden_dim {hidden_dim} not divisible by num_heads {num_heads}"
        
        # Project vision and text to common dimension first
        self.vision_proj = BitNetLinear(vision_dim, hidden_dim, bias=qkv_bias)
        self.text_proj = BitNetLinear(text_dim, hidden_dim, bias=qkv_bias)

        # Standard transformer attention layers (following FIBER)
        self.query = BitNetLinear(hidden_dim, self.all_head_size, bias=qkv_bias)
        self.key = BitNetLinear(hidden_dim, self.all_head_size, bias=qkv_bias)
        self.value = BitNetLinear(hidden_dim, self.all_head_size, bias=qkv_bias)

        # Output projection and normalization
        self.output_proj = BitNetLinear(self.all_head_size, hidden_dim)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(attn_drop)
        self.proj_dropout = nn.Dropout(proj_drop)

    def transpose_for_scores(self, x):
        """Reshape for multi-head attention"""
        new_x_shape = x.size()[:-1] + (self.num_heads, self.head_dim)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        vision_features: torch.Tensor,  # [B, vision_dim]
        text_features: torch.Tensor,   # [B, seq_len, text_dim]
        text_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, Dict]]:
        """
        FIBER-style cross-modal attention

        Returns:
            If output_attentions=False: (vision_output, text_output)
            If output_attentions=True: (vision_output, text_output, attention_info)
        """
        logger.info(f"CrossModalAttention: return_attention={output_attentions}")

        B, seq_len = text_features.shape[:2]

        # Project to common dimension
        vision_proj = self.vision_proj(vision_features)  # [B, hidden_dim]
        text_proj = self.text_proj(text_features)        # [B, seq_len, hidden_dim]

        # Add vision as first token to text sequence (FIBER-style)
        vision_expanded = vision_proj.unsqueeze(1)  # [B, 1, hidden_dim]
        combined_features = torch.cat([vision_expanded, text_proj], dim=1)  # [B, seq_len+1, hidden_dim]

        # Create attention mask for combined sequence
        if text_mask is not None:
            # Add mask for vision token (always attend)
            vision_mask = torch.ones(B, 1, device=text_mask.device, dtype=text_mask.dtype)
            combined_mask = torch.cat([vision_mask, text_mask], dim=1)  # [B, seq_len+1]
            # Convert to attention mask format
            attention_mask = combined_mask.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, seq_len+1]
            attention_mask = (1.0 - attention_mask) * -10000.0
        else:
            attention_mask = None

        # Self-attention on combined sequence
        query_layer = self.transpose_for_scores(self.query(combined_features))
        key_layer = self.transpose_for_scores(self.key(combined_features))
        value_layer = self.transpose_for_scores(self.value(combined_features))

        # Compute attention scores
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.head_dim)

        # Apply attention mask
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask

        # Normalize attention probabilities
        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)

        # Apply attention to values
        context_layer = torch.matmul(attention_probs, value_layer)

        # Reshape output
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        # Project output
        context_layer = self.output_proj(context_layer)
        context_layer = self.proj_dropout(context_layer)

        # Add residual connection and layer norm
        output_features = self.layer_norm(context_layer + combined_features)

        # Split back into vision and text
        vision_output = output_features[:, 0]  # [B, hidden_dim]
        text_output = output_features[:, 1:]   # [B, seq_len, hidden_dim]

        if output_attentions:
            # Create attention info dictionary
            attention_info = {
                'i2t_attention': attention_probs[:, :, 0:1, 1:],  # Vision to text attention
                't2i_attention': attention_probs[:, :, 1:, 0:1],  # Text to vision attention
                'fusion_weights': {'alpha_i2t': 0.5, 'alpha_t2i': 0.5},
                'attention_entropy': -(attention_probs * torch.log(attention_probs + 1e-8)).sum(dim=-1).mean(),
                'cross_modal_similarity': torch.cosine_similarity(vision_output.mean(0), text_output.mean(1).mean(0), dim=0)
            }
            logger.info(f"CrossModalAttention: Returning 3 values - vision: {vision_output.shape}, text: {text_output.shape}, attention_info: {type(attention_info)}")
            return vision_output, text_output, attention_info
        else:
            logger.info(f"CrossModalAttention: Returning 2 values - vision: {vision_output.shape}, text: {text_output.shape}")
            return vision_output, text_output


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
                # Call with output_attentions parameter (fixed parameter name)
                cross_modal_result = cross_modal_layer(vision_feats, text_feats, text_mask, output_attentions=return_attention)

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

                # COMPLETELY SAFE UNPACKING: Use index access instead of unpacking
                if isinstance(cross_modal_result, tuple):
                    if len(cross_modal_result) == 3:
                        logger.info(f"Layer {layer_idx}: Processing 3 values as expected")
                        vision_cross = cross_modal_result[0]
                        text_cross = cross_modal_result[1]
                        attn_weights = cross_modal_result[2]
                    elif len(cross_modal_result) == 2:
                        logger.warning(f"Layer {layer_idx}: Only got 2 values, expected 3 with return_attention={return_attention}")
                        vision_cross = cross_modal_result[0]
                        text_cross = cross_modal_result[1]
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
