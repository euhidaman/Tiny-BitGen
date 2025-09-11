"""
BitMar Evaluation Compatibility Adapter
Ensures seamless compatibility between the hybrid episodic memory system
and the existing evaluation pipeline (evaluate_bitmar_2025.py)
"""

import torch
import logging
from pathlib import Path
from typing import Dict, Optional

logger = logging.getLogger(__name__)


def load_bitmar_checkpoint_with_memory_compatibility(
    checkpoint_path: str,
    config: Dict,
    device: str = 'cuda:0',
    memory_mode: str = "auto"
) -> tuple:
    """
    Load BitMar model checkpoint with full backward compatibility for evaluation

    This function ensures that:
    1. Existing evaluation scripts work without modification
    2. Both integrated and external memory modes are supported
    3. Memory state is properly loaded regardless of storage format

    Args:
        checkpoint_path: Path to model checkpoint
        config: Model configuration
        device: Device to load model on
        memory_mode: "auto", "integrated", or "external"

    Returns:
        (model, config) tuple compatible with evaluation scripts
    """
    try:
        from model import create_bitmar_model
        from memory_utils import load_checkpoint_with_memory_options

        logger.info(f"🔄 Loading BitMar checkpoint with memory compatibility...")
        logger.info(f"   Checkpoint: {checkpoint_path}")
        logger.info(f"   Memory mode: {memory_mode}")

        # Load checkpoint to inspect structure
        checkpoint = torch.load(checkpoint_path, map_location='cpu')

        # Extract model config (maintain compatibility)
        if 'config' in checkpoint:
            model_config = checkpoint['config']
            if 'model' in model_config:
                model_config = model_config['model']
        else:
            model_config = config.get('model', config)

        # Create model with memory configuration
        logger.info("🏗️ Creating BitMar model...")
        model = create_bitmar_model(model_config)

        # Determine memory loading strategy
        separate_memory = checkpoint.get('separate_memory', False)
        memory_path = None

        # Check for separate memory file
        checkpoint_path_obj = Path(checkpoint_path)
        potential_memory_path = checkpoint_path_obj.with_suffix('.memory.pt')

        if separate_memory or potential_memory_path.exists():
            memory_path = str(potential_memory_path)
            logger.info(f"📁 Detected separate memory file: {memory_path}")

        # Handle different memory modes
        if memory_mode == "external" and memory_path:
            logger.info("🔧 Configuring external memory mode...")
            # Enable external storage but load memory for evaluation
            model.memory.enable_external_storage(
                storage_path=memory_path,
                compress=True,
                lazy=False  # Load immediately for evaluation
            )
            # Load the external memory
            success = model.memory.load_external_memory(memory_path, device=device)
            if not success:
                logger.warning("⚠️ External memory loading failed, falling back to integrated")
                memory_mode = "integrated"

        if memory_mode != "external":
            # Use integrated mode (default for evaluation compatibility)
            model.memory.disable_external_storage()

            # Load model state dict (this will include memory if integrated)
            try:
                model.load_state_dict(checkpoint['model_state_dict'], strict=False)
                logger.info("✅ Model state loaded successfully")
            except Exception as e:
                logger.warning(f"⚠️ Partial model loading: {e}")
                # Try loading with strict=False to handle missing keys
                model.load_state_dict(checkpoint['model_state_dict'], strict=False)

            # If there's separate memory, load it in integrated mode
            if memory_path and Path(memory_path).exists():
                logger.info("🔄 Loading separate memory in integrated mode...")
                success = model.memory.load_external_memory(memory_path, device=device)
                if success:
                    logger.info("✅ Separate memory loaded successfully")
                else:
                    logger.warning("⚠️ Separate memory loading failed, using checkpoint memory")

        # Move model to device
        model = model.to(device)
        model.eval()

        # Verify memory state
        memory_info = model.memory.get_memory_info()
        logger.info(f"📊 Memory verification:")
        logger.info(f"   Memory loaded: {memory_info['memory_loaded']}")
        logger.info(f"   External storage: {memory_info['external_storage']}")
        logger.info(f"   Memory utilization: {memory_info.get('memory_utilization', 0):.2%}")

        # Return in format expected by evaluation scripts
        return model, config

    except Exception as e:
        logger.error(f"❌ Failed to load BitMar checkpoint: {e}")
        raise


def save_hf_compatible_model_with_memory(
    checkpoint_path: str,
    output_dir: str,
    memory_strategy: str = "embed"
) -> str:
    """
    Save BitMar model in HuggingFace compatible format while handling episodic memory

    Args:
        checkpoint_path: Path to BitMar checkpoint
        output_dir: Output directory for HF model
        memory_strategy: "embed" (include in model), "separate" (save separately), or "snapshot" (create snapshot)

    Returns:
        Path to HF compatible model directory
    """
    try:
        import json
        import shutil
        from pathlib import Path

        logger.info(f"🔄 Converting BitMar to HuggingFace format...")
        logger.info(f"   Memory strategy: {memory_strategy}")

        # Load BitMar model
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        config = checkpoint.get('config', {})

        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Handle memory based on strategy
        model_state = checkpoint['model_state_dict'].copy()

        if memory_strategy == "separate":
            # Remove memory from model state and save separately
            memory_keys = [k for k in model_state.keys() if k.startswith('memory.')]
            memory_state = {k: model_state.pop(k) for k in memory_keys}

            # Save memory separately
            memory_path = output_path / "episodic_memory.pt"
            torch.save(memory_state, memory_path)
            logger.info(f"💾 Memory saved separately: {memory_path}")

        elif memory_strategy == "snapshot":
            # Create a memory snapshot for later loading
            try:
                # Try to create model to extract memory
                from model import create_bitmar_model
                temp_model = create_bitmar_model(config.get('model', config))
                temp_model.load_state_dict(model_state, strict=False)

                # Create memory snapshot
                snapshot_path = output_path / "memory_snapshot.pt"
                temp_model.memory.save_external_memory(str(snapshot_path), compress=True)
                logger.info(f"📸 Memory snapshot created: {snapshot_path}")

                # Remove memory from model state
                memory_keys = [k for k in model_state.keys() if k.startswith('memory.')]
                for k in memory_keys:
                    model_state.pop(k, None)

            except Exception as e:
                logger.warning(f"⚠️ Snapshot creation failed: {e}, keeping memory embedded")
                memory_strategy = "embed"

        # If embed or snapshot failed, keep memory in model
        if memory_strategy == "embed":
            logger.info("📦 Keeping memory embedded in model")

        # Save model state dict in HF format
        model_path = output_path / "pytorch_model.bin"
        torch.save(model_state, model_path)

        # Create config.json
        model_config = config.get('model', config)
        hf_config = {
            "architectures": ["BitMarModel"],
            "model_type": "bitmar",
            "vocab_size": model_config.get('vocab_size', 50257),
            "hidden_size": model_config.get('text_encoder_dim', 128),
            "num_hidden_layers": model_config.get('text_encoder_layers', 4),
            "num_attention_heads": model_config.get('text_encoder_heads', 4),
            "intermediate_size": model_config.get('text_encoder_dim', 128) * 4,
            "max_position_embeddings": model_config.get('max_seq_len', 256),
            "torch_dtype": "float32",
            "transformers_version": "4.0.0",
            "episodic_memory": {
                "memory_size": model_config.get('memory_size', 32),
                "episode_dim": model_config.get('episode_dim', 128),
                "memory_strategy": memory_strategy,
                "external_files": {
                    "separate": "episodic_memory.pt" if memory_strategy == "separate" else None,
                    "snapshot": "memory_snapshot.pt" if memory_strategy == "snapshot" else None
                }
            }
        }

        config_path = output_path / "config.json"
        with open(config_path, 'w') as f:
            json.dump(hf_config, f, indent=2)

        # Create tokenizer files (basic GPT-2 compatible)
        tokenizer_config = {
            "tokenizer_class": "GPT2Tokenizer",
            "model_max_length": model_config.get('max_seq_len', 256)
        }

        tokenizer_config_path = output_path / "tokenizer_config.json"
        with open(tokenizer_config_path, 'w') as f:
            json.dump(tokenizer_config, f, indent=2)

        # Create README with memory information
        readme_content = f"""# BitMar Model - HuggingFace Compatible

This is a BitMar model converted to HuggingFace format.

## Model Details
- Architecture: BitNet-quantized Vision-Language Episodic Memory Transformer  
- Text Encoder Layers: {model_config.get('text_encoder_layers', 4)}
- Hidden Size: {model_config.get('text_encoder_dim', 128)}
- Episodic Memory: {model_config.get('memory_size', 32)} slots

## Episodic Memory
- Strategy: {memory_strategy}
- Memory Size: {model_config.get('memory_size', 32)} slots
- Episode Dimension: {model_config.get('episode_dim', 128)}

{"- External Memory File: episodic_memory.pt" if memory_strategy == "separate" else ""}
{"- Memory Snapshot: memory_snapshot.pt" if memory_strategy == "snapshot" else ""}
{"- Memory is embedded in the model weights" if memory_strategy == "embed" else ""}

## Usage
```python
from transformers import AutoModel, AutoTokenizer

model = AutoModel.from_pretrained("{output_dir}")
tokenizer = AutoTokenizer.from_pretrained("{output_dir}")
```

Note: This model requires the BitMar architecture implementation to function properly.
"""

        readme_path = output_path / "README.md"
        with open(readme_path, 'w') as f:
            f.write(readme_content)

        logger.info(f"✅ HuggingFace model saved: {output_path}")
        logger.info(f"   Strategy: {memory_strategy}")
        logger.info(f"   Files: pytorch_model.bin, config.json, tokenizer_config.json")

        return str(output_path)

    except Exception as e:
        logger.error(f"❌ HF conversion failed: {e}")
        raise


def patch_evaluation_script_for_memory_compatibility():
    """
    Runtime patches for evaluation script compatibility
    This ensures existing evaluation scripts work without modification
    """

    # Monkey patch the load_model_checkpoint function if needed
    try:
        import sys
        if 'evaluate_bitmar_2025' in sys.modules:
            eval_module = sys.modules['evaluate_bitmar_2025']

            # Store original function
            original_load_model = getattr(eval_module, 'load_model_checkpoint', None)

            if original_load_model:
                def patched_load_model_checkpoint(checkpoint_path, device='cuda:0'):
                    """Patched version that handles memory compatibility"""
                    try:
                        # Try our compatible loader first
                        return load_bitmar_checkpoint_with_memory_compatibility(
                            checkpoint_path, {}, device, memory_mode="auto"
                        )
                    except Exception as e:
                        logger.warning(f"⚠️ Compatible loader failed: {e}")
                        # Fall back to original loader
                        return original_load_model(checkpoint_path, device)

                # Replace the function
                setattr(eval_module, 'load_model_checkpoint', patched_load_model_checkpoint)
                logger.info("✅ Evaluation script patched for memory compatibility")

    except Exception as e:
        logger.warning(f"⚠️ Could not patch evaluation script: {e}")


# Auto-patch when module is imported
patch_evaluation_script_for_memory_compatibility()


class MemoryCompatibilityWrapper:
    """
    Wrapper that ensures memory compatibility across different deployment scenarios
    """

    def __init__(self, model):
        self.model = model
        self._original_state_dict = None

    def __enter__(self):
        """Context manager entry - prepare for evaluation"""
        # Store original memory state
        self._original_state_dict = self.model.state_dict()

        # Ensure memory is loaded and accessible
        if hasattr(self.model, 'memory'):
            self.model.memory._ensure_memory_loaded()

        return self.model

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - cleanup if needed"""
        # Could restore state if needed, but usually not necessary for evaluation
        pass

    def get_memory_info(self) -> Dict:
        """Get comprehensive memory information for debugging"""
        if hasattr(self.model, 'memory'):
            return self.model.memory.get_memory_info()
        return {'error': 'No episodic memory found'}

    def ensure_evaluation_compatibility(self):
        """Ensure model is ready for evaluation"""
        if hasattr(self.model, 'memory'):
            # Make sure memory is in integrated mode for evaluation
            if self.model.memory.external_storage:
                logger.info("🔄 Switching to integrated mode for evaluation...")
                self.model.memory._ensure_memory_loaded()
                self.model.memory.disable_external_storage()

            # Verify memory state
            memory_info = self.model.memory.get_memory_info()
            logger.info(f"📊 Evaluation readiness check:")
            logger.info(f"   Memory loaded: {memory_info['memory_loaded']}")
            logger.info(f"   Update count: {memory_info.get('update_count', 0)}")

        return True


def prepare_model_for_evaluation_pipeline(model, evaluation_type: str = "2025"):
    """
    Prepare BitMar model for specific evaluation pipeline

    Args:
        model: BitMar model
        evaluation_type: "2025" or "2024" pipeline

    Returns:
        Prepared model ready for evaluation
    """
    logger.info(f"🎯 Preparing model for {evaluation_type} evaluation pipeline...")

    # Use compatibility wrapper
    wrapper = MemoryCompatibilityWrapper(model)
    wrapper.ensure_evaluation_compatibility()

    # Pipeline-specific preparations
    if evaluation_type == "2025":
        # 2025 pipeline focuses on text tasks, ensure memory is optimized
        if hasattr(model, 'memory'):
            # Consolidate memory for better text performance
            model.memory.consolidate_memory()
            logger.info("🧠 Memory consolidated for text evaluation")

    elif evaluation_type == "2024":
        # 2024 pipeline focuses on multimodal tasks
        if hasattr(model, 'memory'):
            # Memory is already optimized for multimodal tasks
            logger.info("🖼️ Memory ready for multimodal evaluation")

    logger.info(f"✅ Model prepared for {evaluation_type} evaluation")
    return model
