"""
Memory Visualization Integration for BitMar Training
Seamlessly integrates memory visualization with the training loop
"""

import torch
import logging
from typing import Dict, List, Optional
import wandb
from .memory_visualizer import EpisodicMemoryVisualizer

logger = logging.getLogger(__name__)


class MemoryVisualizationIntegration:
    """Integration layer for memory visualization in BitMar training"""
    
    def __init__(self, config: Dict, model):
        self.config = config
        self.model = model
        self.enabled = config['wandb'].get('log_memory_evolution', False)
        
        if self.enabled:
            self.visualizer = EpisodicMemoryVisualizer(
                memory_size=config['model']['memory_size'],
                episode_dim=config['model']['episode_dim'],
                snapshot_frequency=config['wandb'].get('memory_snapshot_frequency', 100),
                visualization_frequency=config['wandb'].get('memory_visualization_frequency', 500),
                save_dir=config['output'].get('memory_dir', 'memory_visualizations')
            )
            logger.info("🎯 Memory visualization integration enabled")
        else:
            self.visualizer = None
            logger.info("⚪ Memory visualization integration disabled")
    
    def log_training_step(self, 
                         batch: Dict,
                         epoch: int,
                         step: int,
                         model_outputs: Dict,
                         episode_types: List[str] = None):
        """
        Log memory state during training step
        
        Args:
            batch: Training batch data
            epoch: Current epoch
            step: Current training step  
            model_outputs: Model outputs including memory states
            episode_types: List of episode types for this batch
        """
        if not self.enabled or self.visualizer is None:
            return
        
        try:
            # Extract memory slots from model
            memory_slots = self._extract_memory_slots()
            if memory_slots is None:
                return
            
            # Extract episode types if not provided
            if episode_types is None:
                episode_types = self._infer_episode_types(batch)
            
            # Extract access information if available
            slot_access_counts = self._extract_access_counts(model_outputs)
            
            # Get pending data from previous updates
            pending_access, pending_types = self.visualizer.get_pending_data_and_clear()
            
            # Combine with current data
            if slot_access_counts is not None:
                total_access = pending_access + slot_access_counts
            else:
                total_access = pending_access
            
            all_episode_types = pending_types + (episode_types or [])
            
            # Log snapshot
            self.visualizer.log_memory_snapshot(
                memory_slots=memory_slots,
                epoch=epoch,
                step=step,
                episode_types=all_episode_types,
                slot_access_counts=total_access if torch.sum(total_access) > 0 else None
            )
            
        except Exception as e:
            logger.error(f"❌ Error in memory visualization logging: {e}")
    
    def log_memory_update(self,
                         slot_indices: torch.Tensor,
                         similarity_scores: torch.Tensor, 
                         episode_types: List[str]):
        """
        Log memory update event
        
        Args:
            slot_indices: Indices of updated memory slots
            similarity_scores: Similarity scores for slot selection
            episode_types: Types of episodes being stored
        """
        if not self.enabled or self.visualizer is None:
            return
            
        try:
            self.visualizer.log_memory_update(
                slot_indices=slot_indices,
                similarity_scores=similarity_scores,
                episode_types=episode_types
            )
        except Exception as e:
            logger.error(f"❌ Error logging memory update: {e}")
    
    def generate_final_report(self):
        """Generate final memory analysis report"""
        if not self.enabled or self.visualizer is None:
            return
            
        try:
            self.visualizer.generate_final_report()
            logger.info("📋 Generated final memory visualization report")
        except Exception as e:
            logger.error(f"❌ Error generating final memory report: {e}")
    
    def _extract_memory_slots(self) -> Optional[torch.Tensor]:
        """Extract memory slots from the model"""
        try:
            # Try different possible attribute names for memory slots based on actual BitMar structure
            possible_attrs = [
                'memory.memory',  # BitMar structure: self.memory (EpisodicMemory) -> .memory (buffer)
                'memory.memory_slots', 
                'episodic_memory.memory',
                'episodic_memory.memory_slots', 
                'episodic_memory.slots',
                'memory_slots',
                'memory.slots',
            ]
            
            for attr_path in possible_attrs:
                try:
                    # Navigate nested attributes
                    obj = self.model
                    for attr in attr_path.split('.'):
                        obj = getattr(obj, attr)
                    
                    if isinstance(obj, torch.Tensor):
                        logger.debug(f"Found memory slots at: {attr_path}, shape: {obj.shape}")
                        # Ensure tensor is on CPU for visualization to avoid device conflicts
                        return obj.detach().cpu()
                    
                except AttributeError:
                    continue
            
            # Debug: Print available attributes
            if hasattr(self.model, 'memory'):
                memory_attrs = [attr for attr in dir(self.model.memory) if not attr.startswith('_')]
                logger.debug(f"Available memory attributes: {memory_attrs}")
            
            model_attrs = [attr for attr in dir(self.model) if not attr.startswith('_') and 'memory' in attr.lower()]
            logger.debug(f"Available model attributes with 'memory': {model_attrs}")
            
            logger.warning("⚠️  Could not find memory slots in model")
            return None
            
        except Exception as e:
            logger.error(f"❌ Error extracting memory slots: {e}")
            return None
    
    def _infer_episode_types(self, batch: Dict) -> List[str]:
        """Infer episode types from batch data"""
        episode_types = []
        batch_size = batch.get('input_ids', torch.tensor([])).size(0)
        
        # Check if we have vision information
        has_vision = batch.get('has_vision', None)
        if has_vision is not None:
            for i in range(batch_size):
                if has_vision[i]:
                    episode_types.append('multimodal')
                else:
                    episode_types.append('text_only')
        else:
            # Fallback: check if vision features are present and non-zero
            vision_features = batch.get('vision_features', None)
            if vision_features is not None:
                for i in range(batch_size):
                    if torch.sum(torch.abs(vision_features[i])) > 0.1:
                        episode_types.append('multimodal')
                    else:
                        episode_types.append('text_only')
            else:
                # Default to text-only if no vision info available
                episode_types = ['text_only'] * batch_size
        
        return episode_types
    
    def _extract_access_counts(self, model_outputs: Dict) -> Optional[torch.Tensor]:
        """Extract memory slot access counts from model outputs"""
        try:
            # Check for access counts in model outputs (BitMar specific)
            access_keys = [
                'memory_usage',  # BitMar outputs this directly
                'memory_access_counts',
                'slot_access_counts', 
                'slot_usage'
            ]
            
            for key in access_keys:
                if key in model_outputs:
                    tensor = model_outputs[key]
                    if isinstance(tensor, torch.Tensor):
                        # Ensure tensor is on CPU to avoid device conflicts
                        return tensor.detach().cpu()
            
            # If not directly available, try to infer from memory attention weights
            if 'memory_attention' in model_outputs:
                attention_weights = model_outputs['memory_attention']  # [batch_size, memory_size]
                if isinstance(attention_weights, torch.Tensor) and attention_weights.dim() == 2:
                    # Convert attention to access counts (sum over batch)
                    access_counts = torch.sum(attention_weights, dim=0)  # [memory_size]
                    return access_counts.detach().cpu()
            
            # Try alternative attention keys
            attention_keys = ['cross_attention', 'memory_attn', 'episodic_attention']
            for key in attention_keys:
                if key in model_outputs:
                    attention = model_outputs[key]
                    if isinstance(attention, torch.Tensor) and attention.dim() >= 2:
                        # Assume last dimension is memory_size
                        if attention.dim() == 2:  # [batch_size, memory_size]
                            access_counts = torch.sum(attention, dim=0)
                            return access_counts.detach().cpu()
                        elif attention.dim() == 3:  # [batch_size, seq_len, memory_size] or similar
                            access_counts = torch.sum(attention, dim=(0, 1))
                            return access_counts.detach().cpu()
            
            return None
            
        except Exception as e:
            logger.error(f"❌ Error extracting access counts: {e}")
            return None


def setup_memory_visualization(config: Dict, model) -> MemoryVisualizationIntegration:
    """
    Setup memory visualization integration
    
    Args:
        config: Training configuration
        model: BitMar model instance
    
    Returns:
        MemoryVisualizationIntegration instance
    """
    return MemoryVisualizationIntegration(config, model)


# Example usage in training loop:
"""
# In your training script (train_100M_tokens.py):

# Setup visualization
memory_viz = setup_memory_visualization(config, model)

# In training loop:
for epoch in range(max_epochs):
    for step, batch in enumerate(dataloader):
        # Forward pass
        outputs = model(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            vision_features=batch['vision_features'],
            labels=batch['labels'],
            has_vision=batch['has_vision']
        )
        
        # Log memory visualization
        memory_viz.log_training_step(
            batch=batch,
            epoch=epoch,
            step=step,
            model_outputs=outputs
        )
        
        # If you have explicit memory update info:
        # memory_viz.log_memory_update(
        #     slot_indices=updated_slots,
        #     similarity_scores=similarities,
        #     episode_types=episode_types
        # )
        
        # ... rest of training step

# At the end of training:
memory_viz.generate_final_report()
"""
