"""
Robot Selection Dataset Loader
Handles loading and processing of robot selection datasets for text-only reasoning
"""

import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from typing import Dict, List, Optional, Tuple
import logging
import random
from pathlib import Path

logger = logging.getLogger(__name__)


class RobotSelectionDataset(Dataset):
    """Dataset for robot selection tasks with text-only inputs"""

    def __init__(
        self,
        single_robot_path: str,
        multi_robot_path: str,
        tokenizer_name: str = "gpt2",
        max_seq_length: int = 512,
        max_samples: Optional[int] = None,
        include_multi_robot: bool = True
    ):
        self.max_seq_length = max_seq_length
        self.include_multi_robot = include_multi_robot

        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Robot type to index mapping
        self.robot_to_idx = {
            "drone": 0,
            "underwater robot": 1,
            "humanoid": 2,
            "robot with wheels": 3,
            "robot with legs": 4,
            "no robot": 5
        }

        # Load datasets
        self.samples = []
        self._load_single_robot_data(single_robot_path)
        if include_multi_robot:
            self._load_multi_robot_data(multi_robot_path)

        # Limit samples if specified
        if max_samples is not None and len(self.samples) > max_samples:
            random.seed(42)
            self.samples = random.sample(self.samples, max_samples)

        logger.info(f"Loaded {len(self.samples)} robot selection samples")

    def _load_single_robot_data(self, dataset_path: str):
        """Load single robot selection dataset"""
        logger.info(f"Loading single robot selection data from {dataset_path}")

        with open(dataset_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        for item in data:
            # Extract task description from input
            task_description = item['input'].replace('Task: ', '')

            # Parse robot output
            robot_output = item['output'].lower()
            robot_labels = self._parse_robot_output(robot_output)

            sample = {
                'task_description': task_description,
                'robot_labels': robot_labels,
                'dataset_type': 'single',
                'instruction': item['instruction']
            }
            self.samples.append(sample)

    def _load_multi_robot_data(self, dataset_path: str):
        """Load multi-robot selection dataset"""
        logger.info(f"Loading multi-robot selection data from {dataset_path}")

        with open(dataset_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        for item in data:
            # Extract task description from input
            task_description = item['input'].replace('Task: ', '')

            # Parse robot assignments from subtasks
            robot_labels = self._parse_multi_robot_output(item['subtasks'])

            sample = {
                'task_description': task_description,
                'robot_labels': robot_labels,
                'dataset_type': 'multi',
                'instruction': item['instruction'],
                'subtasks': item['subtasks']
            }
            self.samples.append(sample)

    def _parse_robot_output(self, robot_output: str) -> torch.Tensor:
        """Parse single robot output into one-hot tensor"""
        # Create zero tensor for all robots
        labels = torch.zeros(6)

        # Split by comma and clean up robot names
        robots = [r.strip().lower() for r in robot_output.split(',')]

        for robot in robots:
            if robot in self.robot_to_idx:
                labels[self.robot_to_idx[robot]] = 1.0

        return labels

    def _parse_multi_robot_output(self, subtasks: List[Dict]) -> torch.Tensor:
        """Parse multi-robot subtasks into aggregated robot labels"""
        # Create zero tensor for all robots
        labels = torch.zeros(6)

        # Count robot assignments across subtasks
        robot_counts = {}
        for subtask in subtasks:
            robot = subtask['assigned_robot'].lower()
            if robot in self.robot_to_idx:
                robot_counts[robot] = robot_counts.get(robot, 0) + 1

        # Convert counts to probabilities
        total_assignments = sum(robot_counts.values())
        if total_assignments > 0:
            for robot, count in robot_counts.items():
                if robot in self.robot_to_idx:
                    labels[self.robot_to_idx[robot]] = count / total_assignments

        return labels

    def _create_robot_reasoning_context(self, task_description: str) -> str:
        """Create context for robot reasoning including task and capabilities"""
        context = f"""Task: {task_description}

Available Robots and Capabilities:
- Drone: aerial navigation, surveillance, lightweight transport, aerial inspection
- Underwater Robot: underwater navigation, deep sea exploration, marine inspection
- Humanoid: manipulation, walking, human interaction, complex tasks, tool use
- Robot with Wheels: fast movement, good payload, stable platform, efficient
- Robot with Legs: rough terrain navigation, stability, load carrying, inspection

Based on the task requirements and environment, select the most suitable robot(s)."""
        return context

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # Create reasoning context
        context_text = self._create_robot_reasoning_context(sample['task_description'])

        # Tokenize the context
        inputs = self.tokenizer(
            context_text,
            max_length=self.max_seq_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'input_ids': inputs['input_ids'].squeeze(0),
            'attention_mask': inputs['attention_mask'].squeeze(0),
            'robot_labels': sample['robot_labels'],
            'task_description': sample['task_description'],
            'dataset_type': sample['dataset_type'],
            'is_robot_selection': True,  # Flag to identify robot selection samples
            'has_vision': False  # No vision features for robot selection
        }


def create_robot_selection_data_module(
    single_robot_path: str,
    multi_robot_path: str,
    tokenizer_name: str = "gpt2",
    batch_size: int = 8,
    max_seq_length: int = 512,
    max_samples: Optional[int] = None,
    include_multi_robot: bool = True,
    train_split: float = 0.8,
    num_workers: int = 2
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and validation data loaders for robot selection

    Returns:
        Tuple of (train_loader, val_loader)
    """

    # Create full dataset
    full_dataset = RobotSelectionDataset(
        single_robot_path=single_robot_path,
        multi_robot_path=multi_robot_path,
        tokenizer_name=tokenizer_name,
        max_seq_length=max_seq_length,
        max_samples=max_samples,
        include_multi_robot=include_multi_robot
    )

    # Split into train and validation
    total_samples = len(full_dataset)
    train_size = int(train_split * total_samples)
    val_size = total_samples - train_size

    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    def robot_selection_collate_fn(batch):
        """Custom collate function for robot selection data"""
        if not batch:
            return {}

        # Stack tensors
        input_ids = torch.stack([item['input_ids'] for item in batch])
        attention_mask = torch.stack([item['attention_mask'] for item in batch])
        robot_labels = torch.stack([item['robot_labels'] for item in batch])

        # Collect text data
        task_descriptions = [item['task_description'] for item in batch]
        dataset_types = [item['dataset_type'] for item in batch]

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'robot_labels': robot_labels,
            'task_descriptions': task_descriptions,
            'dataset_types': dataset_types,
            'is_robot_selection': True,
            'has_vision': False,
            'vision_features': None  # No vision features
        }

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=robot_selection_collate_fn,
        pin_memory=True,
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=robot_selection_collate_fn,
        pin_memory=True,
        drop_last=False
    )

    logger.info(f"Created robot selection data loaders:")
    logger.info(f"  Train samples: {len(train_dataset)}")
    logger.info(f"  Val samples: {len(val_dataset)}")
    logger.info(f"  Batch size: {batch_size}")

    return train_loader, val_loader


def compute_robot_selection_accuracy(predictions: torch.Tensor, labels: torch.Tensor) -> float:
    """
    Compute accuracy for robot selection predictions

    Args:
        predictions: [batch_size, 6] probability distributions
        labels: [batch_size, 6] ground truth labels (can be multi-hot)

    Returns:
        Accuracy score
    """
    # Get predicted robots (threshold-based for multi-robot support)
    threshold = 0.3
    pred_robots = (predictions > threshold).float()

    # If no robot meets threshold, select the highest probability one
    for i in range(predictions.size(0)):
        if pred_robots[i].sum() == 0:
            max_idx = torch.argmax(predictions[i])
            pred_robots[i, max_idx] = 1.0

    # Compute accuracy (exact match for multi-hot labels)
    correct = (pred_robots == labels).all(dim=1).float()
    accuracy = correct.mean().item()

    return accuracy


def compute_robot_selection_rewards(
    predictions: torch.Tensor,
    labels: torch.Tensor,
    task_descriptions: List[str]
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute rewards for robot selection and reasoning quality

    Args:
        predictions: [batch_size, 6] robot selection probabilities
        labels: [batch_size, 6] ground truth robot labels
        task_descriptions: List of task descriptions

    Returns:
        Tuple of (robot_rewards, reasoning_rewards)
    """
    batch_size = predictions.size(0)

    # Robot selection rewards based on accuracy
    robot_accuracy = compute_robot_selection_accuracy(predictions, labels)
    robot_rewards = torch.full((batch_size,), robot_accuracy)

    # Reasoning quality rewards based on confidence and task complexity
    reasoning_rewards = torch.zeros(batch_size)

    for i, task_desc in enumerate(task_descriptions):
        # Base reward from prediction confidence
        max_confidence = torch.max(predictions[i]).item()
        confidence_reward = max_confidence * 0.5

        # Bonus for task complexity handling
        task_lower = task_desc.lower()
        complexity_bonus = 0.0

        # Multi-environment tasks get complexity bonus
        if any(word in task_lower for word in ['multiple', 'complex', 'coordination']):
            complexity_bonus += 0.2

        # Specific environment matching bonus
        if any(word in task_lower for word in ['underwater', 'marine']) and predictions[i, 1] > 0.5:
            complexity_bonus += 0.3  # Correct underwater task identification
        elif any(word in task_lower for word in ['aerial', 'above']) and predictions[i, 0] > 0.5:
            complexity_bonus += 0.3  # Correct aerial task identification

        reasoning_rewards[i] = confidence_reward + complexity_bonus

    return robot_rewards, reasoning_rewards


if __name__ == "__main__":
    # Test the robot selection dataset
    single_path = "D:/BabyLM/robot_selection_data/data/Single-Robot-Selection/single_robot_selection_dataset.json"
    multi_path = "D:/BabyLM/robot_selection_data/data/Multi-Robot-Selection/multi_robot_selection_dataset.json"

    try:
        train_loader, val_loader = create_robot_selection_data_module(
            single_robot_path=single_path,
            multi_robot_path=multi_path,
            batch_size=4,
            max_samples=20
        )

        # Test a batch
        for batch in train_loader:
            print("Robot Selection Dataset Test:")
            print(f"Input IDs shape: {batch['input_ids'].shape}")
            print(f"Robot labels shape: {batch['robot_labels'].shape}")
            print(f"Sample task: {batch['task_descriptions'][0]}")
            print(f"Sample robot labels: {batch['robot_labels'][0]}")
            break

    except FileNotFoundError as e:
        print(f"Dataset files not found: {e}")
        print("Please ensure the robot selection dataset files exist at the specified paths")
