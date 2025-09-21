"""
Attention Head Visualization and Analysis for BitMar
Inspired by lo-fit repository approach to attention head monitoring
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import wandb
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict
import os
from pathlib import Path

class AttentionHeadAnalyzer:
    """Analyze and visualize attention heads during BitMar training"""
    
    def __init__(self, model, tokenizer, save_dir: str = "./attention_analysis", 
                 wandb_logger=None, track_top_k: int = 10):
        self.model = model
        self.tokenizer = tokenizer
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
        self.wandb_logger = wandb_logger
        self.track_top_k = track_top_k
        
        # Get model dimensions
        self.num_encoder_layers = len(model.text_encoder.layers)
        self.num_decoder_layers = len(model.text_decoder.layers)
        self.num_heads = model.text_encoder.layers[0].attn.num_heads
        self.fusion_layers = len(model.fusion.cross_modal_layers)  # Fixed: use correct attribute name

        # Storage for attention patterns over time
        self.attention_history = {
            'encoder': defaultdict(list),
            'decoder': defaultdict(list), 
            'cross_modal': defaultdict(list),
            'memory': []
        }
        
        # Top heads tracking
        self.head_importance_scores = {
            'encoder': np.zeros((self.num_encoder_layers, self.num_heads)),
            'decoder': np.zeros((self.num_decoder_layers, self.num_heads)),
            'cross_modal': np.zeros((self.fusion_layers, self.num_heads))
        }
        
    def analyze_batch_attention(self, model_outputs: Dict[str, torch.Tensor], 
                               input_ids: torch.Tensor, step: int):
        """Analyze attention patterns for a single batch"""
        
        # Save input tokens and captions for visualization
        self._save_input_context(input_ids, step)
        
        # Analyze text encoder attention
        if 'text_attention' in model_outputs:
            self._analyze_text_attention(model_outputs['text_attention'], input_ids, step, 'encoder')
            
        # Analyze text decoder attention  
        if 'decoder_attention' in model_outputs:
            self._analyze_text_attention(model_outputs['decoder_attention'], input_ids, step, 'decoder')
            
        # Analyze cross-modal attention (this is where token-to-pixel happens)
        if 'cross_attention' in model_outputs:
            self._analyze_cross_modal_attention(model_outputs['cross_attention'], step)
            
        # Analyze memory attention
        if 'memory_attention' in model_outputs:
            self._analyze_memory_attention(model_outputs['memory_attention'], step)
    
    def _save_input_context(self, input_ids: torch.Tensor, step: int):
        """Save input tokens and decoded text for visualization context"""
        
        # Create context directory
        context_dir = self.save_dir / "input_context"
        context_dir.mkdir(exist_ok=True)
        
        # Decode first batch item to get caption text
        if input_ids.shape[0] > 0:
            first_sequence = input_ids[0].detach().cpu().numpy()
            
            # Remove padding tokens and decode
            non_pad_tokens = first_sequence[first_sequence != self.tokenizer.pad_token_id]
            caption_text = self.tokenizer.decode(non_pad_tokens, skip_special_tokens=True)
            
            # Save context data
            context_data = {
                'step': step,
                'input_ids': first_sequence,
                'caption': caption_text,
                'tokens': [self.tokenizer.decode([token_id]) for token_id in non_pad_tokens],
                'sequence_length': len(non_pad_tokens)
            }
            
            # Save as JSON for easy reading
            import json
            context_path = context_dir / f"context_step_{step}.json"
            with open(context_path, 'w') as f:
                json.dump({k: v.tolist() if isinstance(v, np.ndarray) else v 
                          for k, v in context_data.items()}, f, indent=2)
            
    def _analyze_text_attention(self, attention_patterns: List[torch.Tensor], 
                               input_ids: torch.Tensor, step: int, attention_type: str):
        """Analyze self-attention patterns in encoder/decoder"""
        
        for layer_idx, attention_weights in enumerate(attention_patterns):
            if attention_weights is None:
                continue
                
            # attention_weights shape: [batch_size, seq_len, seq_len]
            batch_size, seq_len, _ = attention_weights.shape
            
            # Compute attention statistics per head (we average across heads in the model)
            # For head-level analysis, we need to modify the model to return per-head attention
            
            # For now, analyze the averaged attention patterns
            avg_attention = attention_weights.mean(0)  # Average across batch
            
            # Compute attention entropy (measure of focus)
            entropy = self._compute_attention_entropy(avg_attention)
            
            # Compute attention concentration (how much attention goes to top tokens)
            concentration = self._compute_attention_concentration(avg_attention, top_k=5)
            
            # Store metrics (ensure they are Python floats)
            if isinstance(entropy, torch.Tensor):
                entropy = entropy.item()
            if isinstance(concentration, torch.Tensor):
                concentration = concentration.item()
                
            self.attention_history[attention_type][f'layer_{layer_idx}_entropy'].append(entropy)
            self.attention_history[attention_type][f'layer_{layer_idx}_concentration'].append(concentration)
            
            # Update importance scores (using entropy as importance measure)
            # Note: This is a simplified version. For true head-level analysis, 
            # we need per-head attention weights
            layer_importance = 1.0 / (entropy + 1e-8)  # Higher importance for lower entropy
            if attention_type in self.head_importance_scores:
                # Update all heads equally (since we don't have per-head data)
                self.head_importance_scores[attention_type][layer_idx, :] += layer_importance
                
    def _analyze_cross_modal_attention(self, cross_attention: Dict[str, torch.Tensor], step: int):
        """Analyze cross-modal attention patterns"""
        
        for layer_name, attention_weights in cross_attention.items():
            if attention_weights is None:
                continue
                
            # attention_weights shape: [batch_size, seq_len, vision_dim] (text attending to vision)
            batch_size, seq_len, vision_dim = attention_weights.shape
            
            # Save token-to-pixel attention data for visualization
            self._save_token_pixel_attention(attention_weights, step, layer_name)
            
            # Compute statistics
            avg_attention = attention_weights.mean(0)  # [seq_len, vision_dim]
            
            # Attention distribution across text positions with safe variance computation
            try:
                if avg_attention.shape[1] > 1:  # Need at least 2 elements for variance
                    attention_var = torch.var(avg_attention, dim=1, unbiased=False).mean()  # Use population variance
                else:
                    attention_var = torch.tensor(0.0)  # Fallback for single element
            except RuntimeError:
                attention_var = torch.tensor(0.0)  # Fallback on any variance computation error

            attention_max = torch.max(avg_attention)
            attention_mean = torch.mean(avg_attention)

            # Ensure values are Python floats
            if isinstance(attention_var, torch.Tensor):
                attention_var = attention_var.item()
            if isinstance(attention_max, torch.Tensor):
                attention_max = attention_max.item()
            if isinstance(attention_mean, torch.Tensor):
                attention_mean = attention_mean.item()
            
            # Store metrics
            layer_idx = int(layer_name.split('_')[-1]) if '_' in layer_name else 0
            self.attention_history['cross_modal'][f'layer_{layer_idx}_variance'].append(attention_var)
            self.attention_history['cross_modal'][f'layer_{layer_idx}_max'].append(attention_max)
            self.attention_history['cross_modal'][f'layer_{layer_idx}_mean'].append(attention_mean)
            
            # Update importance scores
            importance = attention_var  # Higher variance = more selective attention
            if layer_idx < self.head_importance_scores['cross_modal'].shape[0]:
                self.head_importance_scores['cross_modal'][layer_idx, :] += importance
    
    def _save_token_pixel_attention(self, attention_weights: torch.Tensor, step: int, layer_name: str):
        """Save token-to-pixel attention data for detailed visualization"""
        
        # Create token-pixel attention directory
        token_pixel_dir = self.save_dir / "token_pixel_attention"
        token_pixel_dir.mkdir(exist_ok=True)
        
        # Save attention weights for this step and layer
        save_data = {
            'attention_weights': attention_weights.detach().cpu().numpy(),
            'step': step,
            'layer': layer_name,
            'shape': attention_weights.shape
        }
        
        save_path = token_pixel_dir / f"attention_{layer_name}_step_{step}.npz"
        np.savez_compressed(save_path, **save_data)
        
        # Also save a subset for quick visualization (first batch item only)
        if attention_weights.shape[0] > 0:
            first_batch = attention_weights[0].detach().cpu().numpy()  # [seq_len, vision_dim]
            
            # Save human-readable format for debugging
            viz_save_path = token_pixel_dir / f"viz_data_{layer_name}_step_{step}.npy"
            np.save(viz_save_path, first_batch)
            
    def _analyze_memory_attention(self, memory_attention: torch.Tensor, step: int):
        """Analyze episodic memory attention patterns"""
        
        if memory_attention is None:
            return
            
        # memory_attention shape: [batch_size, memory_size]
        batch_size, memory_size = memory_attention.shape
        
        # Compute statistics
        avg_attention = memory_attention.mean(0)  # [memory_size]
        
        # Memory access patterns
        entropy = self._compute_attention_entropy(avg_attention.unsqueeze(0))
        top_k_access = torch.topk(avg_attention, k=min(10, memory_size))[0].mean()
        sparsity = (avg_attention < 0.01).float().mean()
        
        # Ensure values are Python floats
        if isinstance(entropy, torch.Tensor):
            entropy = entropy.item()
        if isinstance(top_k_access, torch.Tensor):
            top_k_access = top_k_access.item()
        if isinstance(sparsity, torch.Tensor):
            sparsity = sparsity.item()
        
        # Store metrics
        self.attention_history['memory'].append({
            'step': step,
            'entropy': entropy,
            'top_k_access': top_k_access,
            'sparsity': sparsity
        })
        self.attention_history['memory'].append({
            'entropy': entropy,
            'top_k_access': top_k_access,
            'sparsity': sparsity,
            'step': step
        })
        
    def create_attention_head_heatmap(self, step: int, attention_type: str = 'encoder'):
        """Create attention head importance heatmap like in lo-fit"""
        
        if attention_type not in self.head_importance_scores:
            return
            
        importance_matrix = self.head_importance_scores[attention_type]
        
        # Sort heads by importance (like in lo-fit)
        sorted_importance = np.sort(importance_matrix, axis=1)[:, ::-1]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Raw importance heatmap
        im1 = ax1.imshow(importance_matrix, cmap='viridis', aspect='auto')
        ax1.set_title(f'{attention_type.title()} Attention Head Importance')
        ax1.set_xlabel('Head Index')
        ax1.set_ylabel('Layer Index')
        ax1.set_xticks(range(0, self.num_heads, max(1, self.num_heads//10)))
        plt.colorbar(im1, ax=ax1, label='Importance Score')
        
        # Sorted importance heatmap
        im2 = ax2.imshow(sorted_importance, cmap='viridis', aspect='auto')
        ax2.set_title(f'{attention_type.title()} Heads Sorted by Importance')
        ax2.set_xlabel('Head Rank (0=Most Important)')
        ax2.set_ylabel('Layer Index')
        ax2.set_xticks(range(0, self.num_heads, max(1, self.num_heads//10)))
        plt.colorbar(im2, ax=ax2, label='Importance Score')
        
        plt.tight_layout()
        
        # Save locally
        save_path = self.save_dir / f"attention_heads_{attention_type}_step_{step}.png"
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        # Log to wandb
        if self.wandb_logger:
            wandb.log({f"Attention_Heads/{attention_type.title()}_Importance": wandb.Image(fig)}, step=step)
            
        plt.close(fig)
        
    def create_attention_timeline_plot(self, step: int):
        """Create timeline plots of attention patterns"""
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Encoder attention entropy over time
        ax = axes[0, 0]
        for layer_idx in range(min(3, self.num_encoder_layers)):  # Show first 3 layers
            if f'layer_{layer_idx}_entropy' in self.attention_history['encoder']:
                entropy_values = self.attention_history['encoder'][f'layer_{layer_idx}_entropy']
                steps = range(len(entropy_values))
                ax.plot(steps, entropy_values, label=f'Layer {layer_idx}', marker='o', markersize=2)
        ax.set_title('Encoder Attention Entropy Over Time')
        ax.set_xlabel('Training Step')
        ax.set_ylabel('Attention Entropy')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Cross-modal attention variance over time
        ax = axes[0, 1]
        for layer_idx in range(min(3, self.fusion_layers)):
            if f'layer_{layer_idx}_variance' in self.attention_history['cross_modal']:
                var_values = self.attention_history['cross_modal'][f'layer_{layer_idx}_variance']
                steps = range(len(var_values))
                ax.plot(steps, var_values, label=f'Fusion Layer {layer_idx}', marker='s', markersize=2)
        ax.set_title('Cross-Modal Attention Variance Over Time')
        ax.set_xlabel('Training Step')
        ax.set_ylabel('Attention Variance')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Memory attention patterns
        ax = axes[1, 0]
        if self.attention_history['memory']:
            memory_data = self.attention_history['memory']
            steps = [d['step'] for d in memory_data]
            entropies = [d['entropy'] for d in memory_data]
            sparsities = [d['sparsity'] for d in memory_data]
            
            ax.plot(steps, entropies, label='Memory Entropy', marker='o', markersize=2)
            ax2 = ax.twinx()
            ax2.plot(steps, sparsities, label='Memory Sparsity', color='red', marker='s', markersize=2)
            ax.set_title('Memory Attention Patterns Over Time')
            ax.set_xlabel('Training Step')
            ax.set_ylabel('Entropy', color='blue')
            ax2.set_ylabel('Sparsity', color='red')
            ax.grid(True, alpha=0.3)
        
        # Top attention heads evolution
        ax = axes[1, 1]
        if 'encoder' in self.head_importance_scores:
            # Get top 5 heads overall
            flat_importance = self.head_importance_scores['encoder'].flatten()
            top_indices = np.argsort(flat_importance)[-5:]
            
            for i, idx in enumerate(top_indices):
                layer = idx // self.num_heads
                head = idx % self.num_heads
                ax.bar(i, flat_importance[idx], label=f'L{layer}H{head}')
                
            ax.set_title('Top 5 Most Important Encoder Heads')
            ax.set_xlabel('Head Rank')
            ax.set_ylabel('Importance Score')
            ax.legend()
        
        plt.tight_layout()
        
        # Save and log
        save_path = self.save_dir / f"attention_timeline_step_{step}.png"
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        if self.wandb_logger:
            wandb.log({f"Attention_Timeline/Training_Progress": wandb.Image(fig)}, step=step)
            
        plt.close(fig)
        
    def get_top_attention_heads(self, attention_type: str = 'encoder', k: int = None) -> List[Tuple[int, int]]:
        """Get top-k attention heads like in lo-fit"""
        
        if k is None:
            k = self.track_top_k
            
        if attention_type not in self.head_importance_scores:
            return []
            
        importance_matrix = self.head_importance_scores[attention_type]
        flat_importance = importance_matrix.flatten()
        
        # Get top-k indices
        top_indices = np.argsort(flat_importance)[-k:][::-1]  # Descending order
        
        # Convert to (layer, head) tuples
        top_heads = []
        num_heads = importance_matrix.shape[1]
        for idx in top_indices:
            layer = idx // num_heads
            head = idx % num_heads
            top_heads.append((layer, head))
            
        return top_heads
        
    def save_top_heads(self, step: int, attention_type: str = 'encoder', k: int = None):
        """Save top attention heads to file like in lo-fit"""
        
        top_heads = self.get_top_attention_heads(attention_type, k)
        
        save_path = self.save_dir / f"top_heads_{attention_type}_step_{step}.npy"
        np.save(save_path, np.array(top_heads))
        
        print(f"Saved top {len(top_heads)} {attention_type} attention heads to {save_path}")
        
    def create_head_attention_visualization(self, input_ids: torch.Tensor, 
                                          attention_weights: torch.Tensor,
                                          layer_idx: int, head_idx: int, 
                                          step: int, max_seq_len: int = 50):
        """Create detailed visualization of specific attention head"""
        
        if attention_weights is None or input_ids is None:
            return
            
        # Take first batch item and truncate for visualization
        seq_len = min(input_ids.size(1), max_seq_len)
        tokens = input_ids[0, :seq_len].cpu().numpy()
        attn = attention_weights[0, :seq_len, :seq_len].cpu().numpy()
        
        # Convert token IDs to text
        token_texts = [self.tokenizer.decode([token]) for token in tokens]
        
        # Create attention heatmap
        fig, ax = plt.subplots(figsize=(12, 10))
        
        im = ax.imshow(attn, cmap='Blues', aspect='auto')
        
        # Set ticks and labels
        ax.set_xticks(range(len(token_texts)))
        ax.set_yticks(range(len(token_texts)))
        ax.set_xticklabels(token_texts, rotation=45, ha='right')
        ax.set_yticklabels(token_texts)
        
        ax.set_title(f'Attention Pattern - Layer {layer_idx}, Head {head_idx}\\nStep {step}')
        ax.set_xlabel('Key Tokens')
        ax.set_ylabel('Query Tokens')
        
        # Add colorbar
        plt.colorbar(im, ax=ax, label='Attention Weight')
        
        # Add attention values as text (for smaller matrices)
        if seq_len <= 20:
            for i in range(seq_len):
                for j in range(seq_len):
                    text = ax.text(j, i, f'{attn[i, j]:.2f}',
                                 ha="center", va="center", color="red", fontsize=8)
        
        plt.tight_layout()
        
        # Save and log
        save_path = self.save_dir / f"head_attention_L{layer_idx}H{head_idx}_step_{step}.png"
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        if self.wandb_logger:
            wandb.log({f"Head_Attention/L{layer_idx}H{head_idx}": wandb.Image(fig)}, step=step)
            
        plt.close(fig)
        
    def _compute_attention_entropy(self, attention_weights: torch.Tensor) -> float:
        """Compute entropy of attention weights"""
        # Add small epsilon to avoid log(0)
        entropy = -(attention_weights * torch.log(attention_weights + 1e-8)).sum(dim=-1).mean()
        return entropy.item()
        
    def _compute_attention_concentration(self, attention_weights: torch.Tensor, top_k: int = 5) -> float:
        """Compute attention concentration (sum of top-k attention weights)"""
        top_k_attn = torch.topk(attention_weights, k=min(top_k, attention_weights.size(-1)), dim=-1)[0]
        concentration = top_k_attn.sum(dim=-1).mean()
        return concentration.item()
        
    def generate_attention_report(self, step: int) -> Dict[str, Any]:
        """Generate comprehensive attention analysis report"""
        
        report = {
            'step': step,
            'top_encoder_heads': self.get_top_attention_heads('encoder', 10),
            'top_decoder_heads': self.get_top_attention_heads('decoder', 10),
            'top_cross_modal_heads': self.get_top_attention_heads('cross_modal', 10),
        }
        
        # Add attention statistics
        if 'encoder' in self.head_importance_scores:
            encoder_importance = self.head_importance_scores['encoder']
            report['encoder_avg_importance'] = float(encoder_importance.mean())
            report['encoder_max_importance'] = float(encoder_importance.max())
            report['encoder_importance_std'] = float(encoder_importance.std())
            
        if self.attention_history['memory']:
            recent_memory = self.attention_history['memory'][-10:]  # Last 10 steps
            report['memory_avg_entropy'] = float(np.mean([d['entropy'] for d in recent_memory]))
            report['memory_avg_sparsity'] = float(np.mean([d['sparsity'] for d in recent_memory]))
            
        return report
