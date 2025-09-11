"""
DiNOv2 Feature Reduction for BitMar Edge Deployment
Implements various compression techniques for DiNOv2 features
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional, Dict, Any

class DiNOv2FeatureCompressor(nn.Module):
    """Compress DiNOv2 features for edge deployment"""
    
    def __init__(self, 
                 input_dim: int = 768,
                 target_dim: int = 32,
                 compression_method: str = "top_k_selection",
                 spatial_pooling: bool = True,
                 pool_size: int = 2):
        super().__init__()
        
        self.input_dim = input_dim
        self.target_dim = target_dim
        self.compression_method = compression_method
        self.spatial_pooling = spatial_pooling
        self.pool_size = pool_size
        
        # Initialize compression layers based on method
        self._init_compression_layers()
        
        # Track feature importance for analysis
        self.register_buffer('feature_importance', torch.zeros(input_dim))
        self.register_buffer('usage_count', torch.zeros(1))
        
    def _init_compression_layers(self):
        """Initialize compression layers based on selected method"""
        
        if self.compression_method == "linear_projection":
            self.compressor = nn.Linear(self.input_dim, self.target_dim)
            
        elif self.compression_method == "learned_compression":
            self.compressor = nn.Sequential(
                nn.Linear(self.input_dim, self.input_dim // 2),
                nn.ReLU(inplace=True),
                nn.Dropout(0.1),
                nn.Linear(self.input_dim // 2, self.target_dim)
            )
            
        elif self.compression_method == "autoencoder":
            # More sophisticated compression
            hidden_dim = max(self.target_dim * 2, 64)
            self.compressor = nn.Sequential(
                nn.Linear(self.input_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, self.target_dim),
                nn.ReLU(inplace=True)
            )
            
        elif self.compression_method == "top_k_selection":
            # Will be handled in forward pass
            self.register_buffer('selected_indices', torch.zeros(self.target_dim, dtype=torch.long))
            self._initialize_top_k_indices()
            
        elif self.compression_method == "pca":
            # PCA components will be learned during training
            self.register_buffer('pca_components', torch.zeros(self.target_dim, self.input_dim))
            self.register_buffer('pca_mean', torch.zeros(self.input_dim))
            self._initialize_pca()
            
    def _initialize_top_k_indices(self):
        """Initialize top-K indices with random selection (will be updated during training)"""
        # Start with evenly spaced indices
        indices = torch.linspace(0, self.input_dim - 1, self.target_dim, dtype=torch.long)
        self.selected_indices.copy_(indices)
        
    def _initialize_pca(self):
        """Initialize PCA with identity-like components"""
        # Initialize with random orthogonal vectors
        components = torch.randn(self.target_dim, self.input_dim)
        components = F.normalize(components, dim=1)
        self.pca_components.copy_(components)
        
    def update_feature_importance(self, features: torch.Tensor):
        """Update feature importance statistics during training"""
        
        # Calculate feature variance as importance measure
        if len(features.shape) == 3:
            # [batch, spatial, features] -> variance across batch and spatial
            importance = features.var(dim=(0, 1))
        else:
            # [batch, features] -> variance across batch
            importance = features.var(dim=0)
            
        # Exponential moving average update
        alpha = 0.01
        self.feature_importance.mul_(1 - alpha).add_(importance, alpha=alpha)
        self.usage_count.add_(1)
        
        # Update top-K indices if using that method
        if self.compression_method == "top_k_selection" and self.usage_count % 100 == 0:
            _, top_indices = torch.topk(self.feature_importance, self.target_dim)
            self.selected_indices.copy_(top_indices)
            
    def apply_spatial_pooling(self, features: torch.Tensor) -> torch.Tensor:
        """Apply spatial pooling to reduce spatial dimensions"""
        
        if not self.spatial_pooling or len(features.shape) != 3:
            return features
            
        batch_size, spatial_dim, feature_dim = features.shape
        
        # Assume square spatial layout (e.g., 14x14 = 196)
        spatial_size = int(np.sqrt(spatial_dim))
        if spatial_size * spatial_size != spatial_dim:
            return features  # Can't pool non-square layouts
            
        # Reshape to 2D spatial: [batch, h, w, features]
        features_2d = features.reshape(batch_size, spatial_size, spatial_size, feature_dim)
        
        # Convert to [batch, features, h, w] for pooling
        features_pool_input = features_2d.permute(0, 3, 1, 2)
        
        # Apply average pooling
        pooled = F.avg_pool2d(features_pool_input, kernel_size=self.pool_size, stride=self.pool_size)
        
        # Convert back to [batch, h, w, features]
        pooled_2d = pooled.permute(0, 2, 3, 1)
        
        # Flatten spatial dimensions: [batch, spatial_pooled, features]
        new_spatial_dim = pooled_2d.shape[1] * pooled_2d.shape[2]
        pooled_flat = pooled_2d.reshape(batch_size, new_spatial_dim, feature_dim)
        
        return pooled_flat
        
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Compress DiNOv2 features"""
        
        # Update feature importance during training
        if self.training:
            self.update_feature_importance(features)
            
        # Apply spatial pooling first (if enabled)
        if self.spatial_pooling:
            features = self.apply_spatial_pooling(features)
            
        # Apply feature dimension compression
        if self.compression_method == "linear_projection":
            compressed = self._apply_linear_compression(features)
            
        elif self.compression_method in ["learned_compression", "autoencoder"]:
            compressed = self._apply_learned_compression(features)
            
        elif self.compression_method == "top_k_selection":
            compressed = self._apply_top_k_compression(features)
            
        elif self.compression_method == "pca":
            compressed = self._apply_pca_compression(features)
            
        else:
            raise ValueError(f"Unknown compression method: {self.compression_method}")
            
        return compressed
    
    def _apply_linear_compression(self, features: torch.Tensor) -> torch.Tensor:
        """Apply linear projection compression"""
        
        if len(features.shape) == 3:
            batch_size, spatial_dim, feature_dim = features.shape
            features_flat = features.reshape(-1, feature_dim)
            compressed_flat = self.compressor(features_flat)
            compressed = compressed_flat.reshape(batch_size, spatial_dim, self.target_dim)
        else:
            compressed = self.compressor(features)
            
        return compressed
    
    def _apply_learned_compression(self, features: torch.Tensor) -> torch.Tensor:
        """Apply learned compression (autoencoder-style)"""
        return self._apply_linear_compression(features)  # Same implementation
    
    def _apply_top_k_compression(self, features: torch.Tensor) -> torch.Tensor:
        """Apply top-K feature selection with bounds checking"""

        # Get actual feature dimension
        actual_feature_dim = features.shape[-1]

        # Ensure selected indices are within bounds
        valid_indices = self.selected_indices[self.selected_indices < actual_feature_dim]

        # If we don't have enough valid indices, pad with additional ones
        if len(valid_indices) < self.target_dim:
            # Add more indices up to actual_feature_dim
            additional_needed = min(self.target_dim - len(valid_indices),
                                  actual_feature_dim - len(valid_indices))
            if additional_needed > 0:
                # Find indices not already selected
                all_indices = torch.arange(actual_feature_dim, device=features.device)
                mask = torch.ones(actual_feature_dim, dtype=torch.bool, device=features.device)
                mask[valid_indices] = False
                additional_indices = all_indices[mask][:additional_needed]
                valid_indices = torch.cat([valid_indices, additional_indices])

        # Ensure we don't exceed target_dim
        valid_indices = valid_indices[:self.target_dim]

        # If still not enough, pad with zeros (shouldn't happen with proper config)
        if len(valid_indices) < self.target_dim:
            padding_needed = self.target_dim - len(valid_indices)
            padding_indices = torch.zeros(padding_needed, dtype=torch.long, device=features.device)
            valid_indices = torch.cat([valid_indices, padding_indices])

        # Select features using valid indices
        compressed = features[..., valid_indices]
        return compressed
    
    def _apply_pca_compression(self, features: torch.Tensor) -> torch.Tensor:
        """Apply PCA-based compression"""
        
        # Center features
        features_centered = features - self.pca_mean
        
        # Project onto PCA components
        if len(features.shape) == 3:
            batch_size, spatial_dim, feature_dim = features.shape
            features_flat = features_centered.reshape(-1, feature_dim)
            compressed_flat = torch.matmul(features_flat, self.pca_components.t())
            compressed = compressed_flat.reshape(batch_size, spatial_dim, self.target_dim)
        else:
            compressed = torch.matmul(features_centered, self.pca_components.t())
            
        return compressed
    
    def get_compression_stats(self) -> Dict[str, Any]:
        """Get compression statistics for analysis"""
        
        original_size = self.input_dim
        compressed_size = self.target_dim
        
        spatial_reduction = self.pool_size ** 2 if self.spatial_pooling else 1
        feature_reduction = original_size / compressed_size
        total_reduction = spatial_reduction * feature_reduction
        
        return {
            'method': self.compression_method,
            'original_dim': original_size,
            'compressed_dim': compressed_size,
            'feature_compression_ratio': feature_reduction,
            'spatial_compression_ratio': spatial_reduction,
            'total_compression_ratio': total_reduction,
            'memory_savings_percent': (1 - 1/total_reduction) * 100,
            'spatial_pooling_enabled': self.spatial_pooling,
            'pool_size': self.pool_size if self.spatial_pooling else None
        }
    
    def get_selected_features(self) -> torch.Tensor:
        """Get currently selected feature indices (for top-K method)"""
        if self.compression_method == "top_k_selection":
            return self.selected_indices.clone()
        else:
            return torch.arange(self.target_dim)

class EdgeOptimizedVisionEncoder(nn.Module):
    """Vision encoder optimized for edge deployment with DiNOv2 compression"""
    
    def __init__(self, 
                 input_dim: int = 768,
                 hidden_dim: int = 16,
                 output_dim: int = 32,
                 compression_config: Optional[Dict] = None):
        super().__init__()
        
        # Default compression config
        if compression_config is None:
            compression_config = {
                'method': 'top_k_selection',
                'target_dim': 32,
                'spatial_pooling': True,
                'pool_size': 2
            }
        
        # DiNOv2 feature compressor
        self.feature_compressor = DiNOv2FeatureCompressor(
            input_dim=input_dim,
            target_dim=compression_config.get('target_dim', 32),
            compression_method=compression_config.get('method', 'top_k_selection'),
            spatial_pooling=compression_config.get('spatial_pooling', True),
            pool_size=compression_config.get('pool_size', 2)
        )
        
        # Lightweight processing after compression
        compressed_dim = compression_config.get('target_dim', 32)
        
        self.vision_processor = nn.Sequential(
            nn.Linear(compressed_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim)
        )
        
    def forward(self, dinov2_features: torch.Tensor) -> torch.Tensor:
        """Process DiNOv2 features with compression"""
        
        # Compress features
        compressed_features = self.feature_compressor(dinov2_features)
        
        # Process compressed features
        if len(compressed_features.shape) == 3:
            # [batch, spatial, features] -> process each spatial location
            batch_size, spatial_dim, feature_dim = compressed_features.shape
            compressed_flat = compressed_features.reshape(-1, feature_dim)
            processed_flat = self.vision_processor(compressed_flat)
            processed = processed_flat.reshape(batch_size, spatial_dim, -1)
            
            # Global pooling for final representation
            vision_output = processed.mean(dim=1)  # [batch, output_dim]
        else:
            # Already flattened
            vision_output = self.vision_processor(compressed_features)
            
        return vision_output
    
    def get_compression_info(self) -> Dict[str, Any]:
        """Get information about the compression applied"""
        return self.feature_compressor.get_compression_stats()

def create_edge_vision_encoder(config: Dict) -> EdgeOptimizedVisionEncoder:
    """Factory function to create edge-optimized vision encoder"""
    
    compression_config = {
        'method': config.get('vision_compression_method', 'top_k_selection'),
        'target_dim': config.get('vision_latent_size', 32),
        'spatial_pooling': config.get('vision_spatial_pooling', True),
        'pool_size': config.get('vision_pool_size', 2)
    }
    
    return EdgeOptimizedVisionEncoder(
        input_dim=config.get('vision_encoder_dim', 768),
        hidden_dim=config.get('vision_hidden_size', 16),
        output_dim=config.get('vision_latent_size', 32),
        compression_config=compression_config
    )
