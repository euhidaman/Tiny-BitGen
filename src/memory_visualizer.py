"""
Enhanced Episodic Memory Visualizer for BitMar
Tracks memory evolution, diversity, specialization, and access patterns over training
Integrates with WandB for comprehensive logging
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import wandb
from pathlib import Path
from collections import defaultdict, deque
from typing import Dict, List, Optional, Tuple
import time
import logging
from sklearn.decomposition import PCA
from scipy.stats import entropy

logger = logging.getLogger(__name__)


class EpisodicMemoryVisualizer:
    """Comprehensive memory visualization and tracking for BitMar training"""
    
    def __init__(self, 
                 memory_size: int = 32, 
                 episode_dim: int = 128,
                 snapshot_frequency: int = 100,
                 visualization_frequency: int = 500,
                 save_dir: str = "memory_visualizations"):
        
        self.memory_size = memory_size
        self.episode_dim = episode_dim
        self.snapshot_frequency = snapshot_frequency
        self.visualization_frequency = visualization_frequency
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True, parents=True)
        
        # Memory tracking data structures
        self.memory_snapshots = []  # Store memory states over time
        self.access_counts = defaultdict(lambda: defaultdict(int))  # {epoch: {slot_id: count}}
        self.modal_usage = defaultdict(lambda: {"text_only": 0, "multimodal": 0})  # Modal specialization
        self.diversity_history = []  # Track diversity over time
        self.specialization_history = []  # Track specialization over time
        self.utilization_history = []  # Track slot utilization over time
        self.update_frequency_history = []  # Track update frequencies
        
        # Circular buffer for recent similarity tracking
        self.similarity_buffer = deque(maxlen=1000)
        
        # Step and epoch tracking
        self.current_step = 0
        self.current_epoch = 0
        self.last_visualization_step = 0
        self.last_snapshot_step = 0
        
        # PCA for trajectory visualization (fitted once, reused)
        self.pca = None
        self.trajectory_points = []
        
        logger.info(f"🔍 Memory visualizer initialized: {memory_size} slots, {episode_dim}D episodes")
    
    def log_memory_snapshot(self, 
                           memory_slots: torch.Tensor, 
                           epoch: int, 
                           step: int, 
                           episode_types: Optional[List[str]] = None,
                           slot_access_counts: Optional[torch.Tensor] = None):
        """
        Log a snapshot of current memory state
        
        Args:
            memory_slots: Current memory slot values [memory_size, episode_dim]
            epoch: Current training epoch
            step: Current training step
            episode_types: List of episode types for this batch ("text_only", "multimodal")
            slot_access_counts: Access counts for each slot in this batch
        """
        self.current_step = step
        self.current_epoch = epoch
        
        # Store snapshot if frequency conditions met
        if step - self.last_snapshot_step >= self.snapshot_frequency:
            snapshot = {
                'epoch': epoch,
                'step': step,
                'memory_slots': memory_slots.detach().cpu().numpy(),
                'timestamp': time.time()
            }
            self.memory_snapshots.append(snapshot)
            self.last_snapshot_step = step
            
            # Track trajectory points
            flattened_memory = memory_slots.detach().cpu().numpy().flatten()
            self.trajectory_points.append({
                'epoch': epoch,
                'step': step,
                'memory_flat': flattened_memory
            })
        
        # Update access counts and modal usage
        if slot_access_counts is not None:
            for slot_idx, count in enumerate(slot_access_counts):
                self.access_counts[epoch][slot_idx] += count.item()
        
        if episode_types is not None:
            for ep_type in episode_types:
                if ep_type == "text_only":
                    self.modal_usage[epoch]["text_only"] += 1
                elif ep_type == "multimodal":
                    self.modal_usage[epoch]["multimodal"] += 1
        
        # Compute and store metrics
        self._compute_memory_metrics(memory_slots, epoch, step)
        
        # Generate visualizations if frequency conditions met
        if step - self.last_visualization_step >= self.visualization_frequency:
            self._generate_and_log_visualizations(epoch, step)
            self.last_visualization_step = step
    
    def _compute_memory_metrics(self, memory_slots: torch.Tensor, epoch: int, step: int):
        """Compute memory diversity, specialization, and utilization metrics"""
        memory_np = memory_slots.detach().cpu().numpy()
        
        # 1. Memory Diversity (how different are slots from each other)
        pairwise_similarities = np.corrcoef(memory_np)
        mask = ~np.eye(pairwise_similarities.shape[0], dtype=bool)
        avg_similarity = np.mean(pairwise_similarities[mask])
        diversity_score = 1 - avg_similarity
        self.diversity_history.append({'epoch': epoch, 'step': step, 'diversity': diversity_score})
        
        # 2. Memory Specialization (how concentrated each slot's activation is)
        slot_specializations = []
        for slot in memory_np:
            # Measure concentration using normalized entropy
            normalized_slot = np.abs(slot) / (np.sum(np.abs(slot)) + 1e-8)
            slot_entropy = entropy(normalized_slot + 1e-8)
            specialization = 1 - (slot_entropy / np.log(len(slot)))  # Normalized
            slot_specializations.append(specialization)
        
        avg_specialization = np.mean(slot_specializations)
        self.specialization_history.append({
            'epoch': epoch, 
            'step': step, 
            'specialization': avg_specialization,
            'slot_specializations': slot_specializations
        })
        
        # 3. Memory Utilization (how many slots are actively used)
        slot_norms = np.linalg.norm(memory_np, axis=1)
        active_slots = np.sum(slot_norms > 0.1)  # Threshold for "active" slots
        utilization = active_slots / self.memory_size
        self.utilization_history.append({'epoch': epoch, 'step': step, 'utilization': utilization})
        
        # 4. Log metrics to WandB
        try:
            wandb.log({
                "memory/diversity_score": diversity_score,
                "memory/specialization_score": avg_specialization,
                "memory/slot_utilization": utilization,
                "memory/active_slots": active_slots,
                "memory/max_slot_norm": np.max(slot_norms),
                "memory/min_slot_norm": np.min(slot_norms),
                "memory/avg_slot_norm": np.mean(slot_norms),
            }, step=step)
        except Exception as e:
            logger.warning(f"Failed to log memory metrics to wandb: {e}")
    
    def _generate_and_log_visualizations(self, epoch: int, step: int):
        """Generate and log all visualization plots to WandB"""
        try:
            # 1. Memory Evolution Heatmap
            self._plot_memory_evolution_heatmap(epoch, step)
            
            # 2. Memory Diversity & Specialization
            self._plot_diversity_specialization(epoch, step)
            
            # 3. Memory Access Patterns
            self._plot_access_patterns(epoch, step)
            
            # 4. Cross-Modal Memory Distribution
            self._plot_modal_distribution(epoch, step)
            
            # 5. Memory Learning Trajectory
            self._plot_memory_trajectory(epoch, step)
            
            logger.info(f"📊 Generated memory visualizations at epoch {epoch}, step {step}")
            
        except Exception as e:
            logger.error(f"❌ Error generating memory visualizations: {e}")
    
    def _plot_memory_evolution_heatmap(self, epoch: int, step: int):
        """Plot memory slot evolution heatmap"""
        if len(self.memory_snapshots) < 2:
            return
        
        # Get recent snapshots (last 5 epochs worth)
        recent_snapshots = [s for s in self.memory_snapshots 
                          if s['epoch'] >= max(0, epoch - 4)]
        
        if len(recent_snapshots) < 2:
            return
        
        fig, axes = plt.subplots(1, min(len(recent_snapshots), 5), 
                                figsize=(20, 6))
        if not isinstance(axes, np.ndarray):
            axes = [axes]
        
        for i, snap in enumerate(recent_snapshots[-5:]):  # Last 5 snapshots
            if i >= len(axes):
                break
                
            memory_slots = snap['memory_slots']  # [32, 128]
            ax = axes[i]
            
            # Create heatmap
            sns.heatmap(memory_slots, ax=ax, cmap='RdYlBu_r', 
                       cbar=True if i == 0 else False,
                       xticklabels=False, yticklabels=True if i == 0 else False)
            ax.set_title(f'Epoch {snap["epoch"]}\nStep {snap["step"]}')
            if i == 0:
                ax.set_ylabel('Memory Slots')
        
        plt.suptitle('Memory Slot Evolution Heatmap', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        # Log to WandB
        try:
            wandb.log({f"memory/evolution_heatmap": wandb.Image(plt)}, step=step)
        except Exception as e:
            logger.warning(f"Failed to log evolution heatmap to wandb: {e}")
        plt.close()
    
    def _plot_diversity_specialization(self, epoch: int, step: int):
        """Plot memory diversity and specialization over time"""
        if len(self.diversity_history) < 2:
            return
        
        # Extract data
        steps = [h['step'] for h in self.diversity_history]
        diversity_scores = [h['diversity'] for h in self.diversity_history]
        specialization_scores = [h['specialization'] for h in self.specialization_history]
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # Diversity plot
        ax1.plot(steps, diversity_scores, 'b-', linewidth=2, label='Memory Diversity')
        ax1.set_ylabel('Diversity Score')
        ax1.set_title('Memory Slot Diversity Over Training')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Specialization plot
        ax2.plot(steps, specialization_scores, 'r-', linewidth=2, label='Memory Specialization')
        ax2.set_xlabel('Training Step')
        ax2.set_ylabel('Specialization Score')
        ax2.set_title('Memory Slot Specialization Over Training')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        plt.tight_layout()
        
        # Log to WandB
        try:
            wandb.log({f"memory/diversity_specialization": wandb.Image(plt)}, step=step)
        except Exception as e:
            logger.warning(f"Failed to log diversity/specialization to wandb: {e}")
        plt.close()
    
    def _plot_access_patterns(self, epoch: int, step: int):
        """Plot memory access patterns"""
        if len(self.access_counts) < 2:
            return
        
        # Convert access counts to matrix
        epochs = sorted(self.access_counts.keys())
        access_matrix = np.zeros((len(epochs), self.memory_size))
        
        for i, ep in enumerate(epochs):
            for slot_idx in range(self.memory_size):
                access_matrix[i, slot_idx] = self.access_counts[ep].get(slot_idx, 0)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Access pattern heatmap
        sns.heatmap(access_matrix.T, ax=ax1, cmap='YlOrRd', 
                    xticklabels=epochs[::max(1, len(epochs)//10)],
                    yticklabels=list(range(self.memory_size)))
        ax1.set_title('Memory Slot Access Frequency Over Epochs')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Memory Slot Index')
        
        # Usage distribution
        slot_totals = np.sum(access_matrix, axis=0)
        most_used = np.argmax(slot_totals)
        least_used = np.argmin(slot_totals[slot_totals > 0]) if np.any(slot_totals > 0) else 0
        
        ax2.plot(epochs, access_matrix[:, most_used], 'g-', 
                linewidth=2, label=f'Most Used Slot #{most_used}')
        ax2.plot(epochs, access_matrix[:, least_used], 'r-', 
                linewidth=2, label=f'Least Used Slot #{least_used}')
        ax2.plot(epochs, np.mean(access_matrix, axis=1), 'b--', 
                linewidth=2, label='Average Usage')
        
        ax2.set_title('Memory Slot Usage Patterns')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Access Count')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Log to WandB
        try:
            wandb.log({f"memory/access_patterns": wandb.Image(plt)}, step=step)
        except Exception as e:
            logger.warning(f"Failed to log access patterns to wandb: {e}")
        
        # Also log usage entropy
        if len(slot_totals) > 0 and np.sum(slot_totals) > 0:
            normalized_usage = slot_totals / np.sum(slot_totals)
            usage_entropy = entropy(normalized_usage + 1e-8)
            try:
                wandb.log({"memory/usage_entropy": usage_entropy}, step=step)
            except Exception as e:
                logger.warning(f"Failed to log usage entropy to wandb: {e}")
        
        plt.close()
    
    def _plot_modal_distribution(self, epoch: int, step: int):
        """Plot cross-modal memory distribution"""
        if len(self.modal_usage) < 2:
            return
        
        epochs = sorted(self.modal_usage.keys())
        text_only_counts = [self.modal_usage[ep]["text_only"] for ep in epochs]
        multimodal_counts = [self.modal_usage[ep]["multimodal"] for ep in epochs]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Stacked bar chart
        width = 0.8
        ax1.bar(epochs, text_only_counts, width, label='Text-Only Episodes', color='skyblue')
        ax1.bar(epochs, multimodal_counts, width, bottom=text_only_counts, 
                label='Multimodal Episodes', color='lightcoral')
        
        ax1.set_title('Memory Episode Modal Distribution Over Epochs')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Episode Count')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Modal ratio over time
        total_counts = np.array(text_only_counts) + np.array(multimodal_counts)
        text_ratio = np.array(text_only_counts) / (total_counts + 1e-8)
        multimodal_ratio = np.array(multimodal_counts) / (total_counts + 1e-8)
        
        ax2.plot(epochs, text_ratio, 'b-', linewidth=2, label='Text-Only Ratio')
        ax2.plot(epochs, multimodal_ratio, 'r-', linewidth=2, label='Multimodal Ratio')
        ax2.axhline(y=0.5, color='gray', linestyle='--', alpha=0.7, label='50% Line')
        
        ax2.set_title('Modal Episode Ratio Over Training')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Episode Ratio')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Log to WandB
        try:
            wandb.log({f"memory/modal_distribution": wandb.Image(plt)}, step=step)
        except Exception as e:
            logger.warning(f"Failed to log modal distribution to wandb: {e}")
        
        # Log current ratios
        if len(total_counts) > 0 and total_counts[-1] > 0:
            try:
                wandb.log({
                    "memory/text_only_ratio": text_ratio[-1],
                    "memory/multimodal_ratio": multimodal_ratio[-1]
                }, step=step)
            except Exception as e:
                logger.warning(f"Failed to log modal ratios to wandb: {e}")
        
        plt.close()
    
    def _plot_memory_trajectory(self, epoch: int, step: int):
        """Plot memory learning trajectory in 2D space"""
        if len(self.trajectory_points) < 10:
            return
        
        # Prepare data for PCA
        memory_matrices = [point['memory_flat'] for point in self.trajectory_points]
        epochs = [point['epoch'] for point in self.trajectory_points]
        
        # Fit PCA if not already fitted
        if self.pca is None:
            self.pca = PCA(n_components=2)
            self.pca.fit(memory_matrices)
        
        # Transform to 2D
        memory_2d = self.pca.transform(memory_matrices)
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Plot trajectory with color coding for epochs
        scatter = ax.scatter(memory_2d[:, 0], memory_2d[:, 1], 
                           c=epochs, cmap='viridis', s=60, alpha=0.7)
        
        # Draw arrows showing learning direction
        for i in range(0, len(memory_2d) - 1, max(1, len(memory_2d)//20)):
            ax.annotate('', xy=memory_2d[i+1], xytext=memory_2d[i],
                       arrowprops=dict(arrowstyle='->', color='gray', alpha=0.5))
        
        # Mark start and current points
        ax.scatter(memory_2d[0, 0], memory_2d[0, 1], 
                  color='red', s=100, marker='s', label='Start', zorder=5)
        ax.scatter(memory_2d[-1, 0], memory_2d[-1, 1], 
                  color='green', s=100, marker='*', label='Current', zorder=5)
        
        plt.colorbar(scatter, ax=ax, label='Epoch')
        ax.set_title('Episodic Memory Learning Trajectory')
        ax.set_xlabel(f'PC1 ({self.pca.explained_variance_ratio_[0]:.2%} variance)')
        ax.set_ylabel(f'PC2 ({self.pca.explained_variance_ratio_[1]:.2%} variance)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Log to WandB
        try:
            wandb.log({f"memory/learning_trajectory": wandb.Image(plt)}, step=step)
        except Exception as e:
            logger.warning(f"Failed to log learning trajectory to wandb: {e}")
        plt.close()
    
    def log_memory_update(self, 
                         slot_indices: torch.Tensor, 
                         similarity_scores: torch.Tensor,
                         episode_types: List[str]):
        """
        Log memory update information
        
        Args:
            slot_indices: Which slots were updated [batch_size]
            similarity_scores: Similarity scores used for slot selection [batch_size, memory_size]
            episode_types: Types of episodes in this batch
        """
        # Track slot access frequency
        slot_access_counts = torch.bincount(slot_indices, minlength=self.memory_size)
        
        # Store for next snapshot
        if not hasattr(self, '_pending_access_counts'):
            self._pending_access_counts = slot_access_counts
        else:
            self._pending_access_counts += slot_access_counts
        
        if not hasattr(self, '_pending_episode_types'):
            self._pending_episode_types = episode_types.copy()
        else:
            self._pending_episode_types.extend(episode_types)
    
    def get_pending_data_and_clear(self) -> Tuple[torch.Tensor, List[str]]:
        """Get and clear pending access counts and episode types"""
        access_counts = getattr(self, '_pending_access_counts', 
                               torch.zeros(self.memory_size))
        episode_types = getattr(self, '_pending_episode_types', [])
        
        # Clear pending data
        self._pending_access_counts = torch.zeros(self.memory_size)
        self._pending_episode_types = []
        
        return access_counts, episode_types
    
    def generate_final_report(self, save_path: Optional[str] = None):
        """Generate comprehensive final memory analysis report"""
        if save_path is None:
            save_path = self.save_dir / "final_memory_report.png"
        
        fig = plt.figure(figsize=(20, 12))
        gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
        
        # Plot all metrics in a comprehensive dashboard
        self._plot_final_diversity_metrics(fig.add_subplot(gs[0, :2]))
        self._plot_final_access_patterns(fig.add_subplot(gs[0, 2:]))
        self._plot_final_utilization(fig.add_subplot(gs[1, :2]))
        self._plot_final_modal_distribution(fig.add_subplot(gs[1, 2:]))
        self._plot_final_trajectory(fig.add_subplot(gs[2, :]))
        
        plt.suptitle('Final Episodic Memory Analysis Report', 
                    fontsize=16, fontweight='bold')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        # Log final report to WandB
        try:
            wandb.log({"memory/final_report": wandb.Image(str(save_path))})
        except Exception as e:
            logger.warning(f"Failed to log final report to wandb: {e}")
        plt.close()
        
        logger.info(f"📋 Generated final memory report: {save_path}")
    
    def _plot_final_diversity_metrics(self, ax):
        """Plot final diversity and specialization metrics"""
        if not self.diversity_history:
            return
            
        steps = [h['step'] for h in self.diversity_history]
        diversity = [h['diversity'] for h in self.diversity_history]
        specialization = [h['specialization'] for h in self.specialization_history]
        
        ax2 = ax.twinx()
        line1 = ax.plot(steps, diversity, 'b-', linewidth=2, label='Diversity')
        line2 = ax2.plot(steps, specialization, 'r-', linewidth=2, label='Specialization')
        
        ax.set_xlabel('Training Step')
        ax.set_ylabel('Diversity Score', color='b')
        ax2.set_ylabel('Specialization Score', color='r')
        ax.set_title('Memory Evolution: Diversity & Specialization')
        
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax.legend(lines, labels, loc='upper left')
        ax.grid(True, alpha=0.3)
    
    def _plot_final_access_patterns(self, ax):
        """Plot final access pattern summary"""
        if not self.access_counts:
            return
            
        # Aggregate total access counts
        total_access = defaultdict(int)
        for epoch_data in self.access_counts.values():
            for slot_idx, count in epoch_data.items():
                total_access[slot_idx] += count
        
        slots = list(range(self.memory_size))
        counts = [total_access[i] for i in slots]
        
        bars = ax.bar(slots, counts, color='skyblue', alpha=0.7)
        ax.set_xlabel('Memory Slot Index')
        ax.set_ylabel('Total Access Count')
        ax.set_title('Total Memory Slot Usage')
        ax.grid(True, alpha=0.3)
        
        # Highlight most and least used slots
        if counts:
            max_idx = np.argmax(counts)
            min_idx = np.argmin([c for c in counts if c > 0]) if any(c > 0 for c in counts) else 0
            bars[max_idx].set_color('green')
            bars[min_idx].set_color('red')
    
    def _plot_final_utilization(self, ax):
        """Plot final utilization trends"""
        if not self.utilization_history:
            return
            
        steps = [h['step'] for h in self.utilization_history]
        utilization = [h['utilization'] for h in self.utilization_history]
        
        ax.plot(steps, utilization, 'g-', linewidth=2)
        ax.axhline(y=0.8, color='orange', linestyle='--', alpha=0.7, label='80% Target')
        ax.set_xlabel('Training Step')
        ax.set_ylabel('Slot Utilization Ratio')
        ax.set_title('Memory Slot Utilization Over Time')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_final_modal_distribution(self, ax):
        """Plot final modal distribution"""
        if not self.modal_usage:
            return
            
        epochs = sorted(self.modal_usage.keys())
        text_counts = [self.modal_usage[ep]["text_only"] for ep in epochs]
        multi_counts = [self.modal_usage[ep]["multimodal"] for ep in epochs]
        
        total_counts = np.array(text_counts) + np.array(multi_counts)
        text_ratio = np.array(text_counts) / (total_counts + 1e-8)
        
        ax.plot(epochs, text_ratio, 'b-', linewidth=2, label='Text-Only Ratio')
        ax.plot(epochs, 1 - text_ratio, 'r-', linewidth=2, label='Multimodal Ratio')
        ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.7, label='50% Balance')
        
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Episode Ratio')
        ax.set_title('Cross-Modal Episode Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_final_trajectory(self, ax):
        """Plot final learning trajectory"""
        if len(self.trajectory_points) < 10 or self.pca is None:
            return
            
        memory_matrices = [point['memory_flat'] for point in self.trajectory_points]
        epochs = [point['epoch'] for point in self.trajectory_points]
        
        memory_2d = self.pca.transform(memory_matrices)
        
        scatter = ax.scatter(memory_2d[:, 0], memory_2d[:, 1], 
                           c=epochs, cmap='viridis', s=40, alpha=0.7)
        
        # Draw path
        ax.plot(memory_2d[:, 0], memory_2d[:, 1], 'gray', alpha=0.5, linewidth=1)
        
        # Mark key points
        ax.scatter(memory_2d[0, 0], memory_2d[0, 1], 
                  color='red', s=100, marker='s', label='Start', zorder=5)
        ax.scatter(memory_2d[-1, 0], memory_2d[-1, 1], 
                  color='green', s=100, marker='*', label='End', zorder=5)
        
        ax.set_xlabel(f'PC1 ({self.pca.explained_variance_ratio_[0]:.2%} variance)')
        ax.set_ylabel(f'PC2 ({self.pca.explained_variance_ratio_[1]:.2%} variance)')
        ax.set_title('Memory Learning Trajectory')
        ax.legend()
        ax.grid(True, alpha=0.3)


def create_memory_visualizer(config: Dict) -> EpisodicMemoryVisualizer:
    """Factory function to create memory visualizer from config"""
    return EpisodicMemoryVisualizer(
        memory_size=config['model']['memory_size'],
        episode_dim=config['model']['episode_dim'],
        snapshot_frequency=config['wandb'].get('memory_snapshot_frequency', 100),
        visualization_frequency=config['wandb'].get('memory_visualization_frequency', 500),
        save_dir=config['output'].get('memory_dir', 'memory_visualizations')
    )
