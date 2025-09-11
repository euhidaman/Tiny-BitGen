"""
FLOPS Tracking System for BitMar Training
Monitors computational cost and efficiency during training
"""

import torch
import torch.nn as nn
import time
import logging
from typing import Dict, Optional, Tuple, Any
from collections import defaultdict
import numpy as np
from pathlib import Path
import json

logger = logging.getLogger(__name__)


class FLOPsCounter:
    """
    FLOPS counter for neural network operations
    Tracks forward and backward pass FLOPS
    """

    def __init__(self):
        self.flops_dict = defaultdict(int)
        self.hooks = []
        self.current_step_flops = 0
        self.total_flops = 0

    def add_flops(self, flops: int):
        """Add FLOPS to current step and total"""
        self.current_step_flops += flops
        self.total_flops += flops

    def reset_step_flops(self):
        """Reset current step FLOPS counter"""
        self.current_step_flops = 0

    def get_step_flops(self) -> int:
        """Get FLOPS for current step"""
        return self.current_step_flops

    def get_total_flops(self) -> int:
        """Get total FLOPS since start"""
        return self.total_flops


def flops_linear_layer(input_tensor: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor] = None) -> int:
    """Calculate FLOPS for linear layer"""
    batch_size = input_tensor.shape[0]
    input_dim = weight.shape[1]
    output_dim = weight.shape[0]

    # Matrix multiplication: batch_size * input_dim * output_dim
    flops = batch_size * input_dim * output_dim

    # Add bias if present
    if bias is not None:
        flops += batch_size * output_dim

    return flops


def flops_attention(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> int:
    """Calculate FLOPS for attention mechanism"""
    batch_size, seq_len, d_model = query.shape

    # Q @ K^T
    qk_flops = batch_size * seq_len * seq_len * d_model

    # Softmax (approximated)
    softmax_flops = batch_size * seq_len * seq_len * 5  # exp, sum, div

    # Attention @ V
    av_flops = batch_size * seq_len * seq_len * d_model

    return qk_flops + softmax_flops + av_flops


def flops_layer_norm(input_tensor: torch.Tensor) -> int:
    """Calculate FLOPS for layer normalization"""
    numel = input_tensor.numel()
    # Mean calculation + variance + normalization + scale + shift
    return numel * 5


def flops_activation(input_tensor: torch.Tensor, activation_type: str = "gelu") -> int:
    """Calculate FLOPS for activation functions"""
    numel = input_tensor.numel()

    if activation_type.lower() == "gelu":
        return numel * 8  # GELU is expensive
    elif activation_type.lower() == "relu":
        return numel * 1
    elif activation_type.lower() == "silu":
        return numel * 4
    else:
        return numel * 2  # Default estimate


class FLOPsTracker:
    """
    Main FLOPS tracking system for BitMar training
    """

    def __init__(self, model: nn.Module, log_frequency: int = 100, save_dir: Optional[str] = None):
        self.model = model
        self.log_frequency = log_frequency
        self.save_dir = Path(save_dir) if save_dir else Path("./flops_logs")
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # FLOPS tracking
        self.flops_counter = FLOPsCounter()
        self.step_count = 0
        self.start_time = time.time()

        # History tracking
        self.flops_history = []
        self.timing_history = []
        self.throughput_history = []

        # Hook storage
        self.hooks = []

        # Register hooks for automatic FLOPS counting
        self._register_hooks()

        logger.info(f"🔢 FLOPS Tracker initialized")
        logger.info(f"  • Log frequency: {log_frequency} steps")
        logger.info(f"  • Save directory: {self.save_dir}")

    def _register_hooks(self):
        """Register forward hooks for automatic FLOPS counting"""

        def linear_hook(module, input, output):
            if len(input) > 0 and hasattr(module, 'weight'):
                flops = flops_linear_layer(input[0], module.weight, module.bias)
                self.flops_counter.add_flops(flops)

        def attention_hook(module, input, output):
            # This is a simplified hook for attention - might need customization
            if hasattr(module, 'num_heads') and len(input) > 0:
                batch_size, seq_len, d_model = input[0].shape
                # Estimate attention FLOPS
                flops = batch_size * module.num_heads * seq_len * seq_len * (d_model // module.num_heads) * 4
                self.flops_counter.add_flops(flops)

        def layernorm_hook(module, input, output):
            if len(input) > 0:
                flops = flops_layer_norm(input[0])
                self.flops_counter.add_flops(flops)

        def activation_hook(module, input, output):
            if len(input) > 0:
                activation_name = module.__class__.__name__.lower()
                flops = flops_activation(input[0], activation_name)
                self.flops_counter.add_flops(flops)

        # Register hooks for different layer types
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                hook = module.register_forward_hook(linear_hook)
                self.hooks.append(hook)
            elif isinstance(module, nn.LayerNorm):
                hook = module.register_forward_hook(layernorm_hook)
                self.hooks.append(hook)
            elif isinstance(module, (nn.GELU, nn.ReLU, nn.SiLU)):
                hook = module.register_forward_hook(activation_hook)
                self.hooks.append(hook)
            elif hasattr(module, 'num_heads'):  # Attention modules
                hook = module.register_forward_hook(attention_hook)
                self.hooks.append(hook)

        logger.info(f"Registered {len(self.hooks)} FLOPS tracking hooks")

    def start_step(self):
        """Start tracking a new training step"""
        self.flops_counter.reset_step_flops()
        self.step_start_time = time.time()

    def end_step(self, batch_size: int = 1, sequence_length: int = 1) -> Dict[str, float]:
        """End tracking current step and return metrics"""
        step_time = time.time() - self.step_start_time
        step_flops = self.flops_counter.get_step_flops()
        total_flops = self.flops_counter.get_total_flops()

        # Calculate throughput metrics
        flops_per_second = step_flops / step_time if step_time > 0 else 0
        tokens_per_second = (batch_size * sequence_length) / step_time if step_time > 0 else 0

        # Store history
        self.flops_history.append(step_flops)
        self.timing_history.append(step_time)
        self.throughput_history.append(flops_per_second)

        self.step_count += 1

        metrics = {
            'step_flops': step_flops,
            'total_flops': total_flops,
            'step_time': step_time,
            'flops_per_second': flops_per_second,
            'tokens_per_second': tokens_per_second,
            'avg_flops_per_step': total_flops / self.step_count if self.step_count > 0 else 0,
            'total_time': time.time() - self.start_time
        }

        return metrics

    def should_log(self) -> bool:
        """Check if we should log FLOPS at this step"""
        return self.step_count % self.log_frequency == 0

    def log_flops(self, metrics: Dict[str, float], logger_func=None, wandb_logger=None, step: int = None):
        """Log FLOPS metrics"""
        if logger_func is None:
            logger_func = logger.info

        # Format FLOPS values for readability
        step_flops_str = self._format_flops(metrics['step_flops'])
        total_flops_str = self._format_flops(metrics['total_flops'])
        flops_per_sec_str = self._format_flops(metrics['flops_per_second'])

        logger_func(f"🔢 FLOPS Metrics (Step {self.step_count}):")
        logger_func(f"  • Step FLOPS: {step_flops_str}")
        logger_func(f"  • Total FLOPS: {total_flops_str}")
        logger_func(f"  • FLOPS/sec: {flops_per_sec_str}")
        logger_func(f"  • Tokens/sec: {metrics['tokens_per_second']:.1f}")
        logger_func(f"  • Step time: {metrics['step_time']:.3f}s")
        logger_func(f"  • Total time: {metrics['total_time']:.1f}s")

        # Log to wandb if available
        if wandb_logger and hasattr(wandb_logger, 'log'):
            wandb_metrics = {
                'flops/step_flops': metrics['step_flops'],
                'flops/total_flops': metrics['total_flops'],
                'flops/flops_per_second': metrics['flops_per_second'],
                'flops/tokens_per_second': metrics['tokens_per_second'],
                'flops/avg_flops_per_step': metrics['avg_flops_per_step'],
                'flops/step_time': metrics['step_time'],
                'flops/total_time': metrics['total_time']
            }
            try:
                if step is not None:
                    wandb_logger.log(wandb_metrics, step=step)
                else:
                    wandb_logger.log(wandb_metrics)
            except Exception as e:
                logger.warning(f"Failed to log FLOPS to wandb: {e}")

    def _format_flops(self, flops: float) -> str:
        """Format FLOPS with appropriate units"""
        if flops >= 1e12:
            return f"{flops/1e12:.2f} TFLOPS"
        elif flops >= 1e9:
            return f"{flops/1e9:.2f} GFLOPS"
        elif flops >= 1e6:
            return f"{flops/1e6:.2f} MFLOPS"
        elif flops >= 1e3:
            return f"{flops/1e3:.2f} KFLOPS"
        else:
            return f"{flops:.0f} FLOPS"

    def get_summary_stats(self) -> Dict[str, Any]:
        """Get summary statistics"""
        if not self.flops_history:
            return {}

        total_flops = self.flops_counter.get_total_flops()
        total_time = time.time() - self.start_time

        return {
            'total_flops': total_flops,
            'total_time': total_time,
            'total_steps': self.step_count,
            'avg_flops_per_step': np.mean(self.flops_history),
            'std_flops_per_step': np.std(self.flops_history),
            'avg_step_time': np.mean(self.timing_history),
            'avg_throughput': np.mean(self.throughput_history),
            'peak_throughput': np.max(self.throughput_history) if self.throughput_history else 0,
            'flops_formatted': self._format_flops(total_flops),
            'avg_throughput_formatted': self._format_flops(np.mean(self.throughput_history)) + "/s" if self.throughput_history else "0 FLOPS/s"
        }

    def save_statistics(self, filename: str = "flops_statistics.json"):
        """Save FLOPS statistics to file"""
        stats = self.get_summary_stats()
        stats['flops_history'] = self.flops_history
        stats['timing_history'] = self.timing_history
        stats['throughput_history'] = self.throughput_history

        save_path = self.save_dir / filename
        try:
            with open(save_path, 'w') as f:
                json.dump(stats, f, indent=2, default=str)
            logger.info(f"💾 FLOPS statistics saved to {save_path}")
        except Exception as e:
            logger.error(f"Failed to save FLOPS statistics: {e}")

    def cleanup(self):
        """Remove hooks and cleanup"""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
        logger.info("🔢 FLOPS Tracker cleaned up")

    def log_model_complexity(self):
        """Log model computational complexity information"""
        try:
            # Count parameters
            total_params = sum(p.numel() for p in self.model.parameters())
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

            logger.info(f"🏗️  Model Complexity:")
            logger.info(f"  • Total parameters: {total_params:,}")
            logger.info(f"  • Trainable parameters: {trainable_params:,}")
            logger.info(f"  • Non-trainable parameters: {total_params - trainable_params:,}")

            # Estimate memory usage
            param_memory = total_params * 4 / (1024**2)  # 4 bytes per float32 parameter
            logger.info(f"  • Parameter memory: {param_memory:.1f} MB")

        except Exception as e:
            logger.warning(f"Failed to log model complexity: {e}")


class FLOPsEstimator:
    """
    Estimate FLOPS for BitMar model components
    """

    @staticmethod
    def estimate_transformer_flops(batch_size: int, seq_length: int, d_model: int,
                                 num_layers: int, num_heads: int, vocab_size: int) -> Dict[str, int]:
        """Estimate FLOPS for transformer model"""

        # Embedding layer
        embedding_flops = batch_size * seq_length * d_model

        # Transformer layers
        layer_flops = 0
        for _ in range(num_layers):
            # Self-attention
            attention_flops = flops_attention(
                torch.zeros(batch_size, seq_length, d_model),
                torch.zeros(batch_size, seq_length, d_model),
                torch.zeros(batch_size, seq_length, d_model)
            )

            # Feed-forward (assuming 4x expansion)
            ff_flops = batch_size * seq_length * d_model * (4 * d_model) * 2  # Two linear layers

            # Layer norms
            ln_flops = flops_layer_norm(torch.zeros(batch_size, seq_length, d_model)) * 2

            layer_flops += attention_flops + ff_flops + ln_flops

        # Output projection
        output_flops = batch_size * seq_length * d_model * vocab_size

        total_flops = embedding_flops + layer_flops + output_flops

        return {
            'embedding_flops': embedding_flops,
            'transformer_flops': layer_flops,
            'output_flops': output_flops,
            'total_flops': total_flops
        }

    @staticmethod
    def estimate_vision_encoder_flops(batch_size: int, vision_dim: int,
                                    latent_dim: int) -> Dict[str, int]:
        """Estimate FLOPS for vision encoder"""

        # Vision projection
        projection_flops = batch_size * vision_dim * latent_dim

        # Activation
        activation_flops = batch_size * latent_dim

        total_flops = projection_flops + activation_flops

        return {
            'projection_flops': projection_flops,
            'activation_flops': activation_flops,
            'total_flops': total_flops
        }
