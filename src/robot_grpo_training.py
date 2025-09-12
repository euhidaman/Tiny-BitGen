"""
Robot Selection GRPO Training Module
Implements GRPO reasoning for robot selection grounding based on task requirements
"""

import json
import re
from typing import Dict, List, Tuple, Optional
from datasets import Dataset
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from trl import GRPOConfig, GRPOTrainer, ModelConfig
import logging

logger = logging.getLogger(__name__)

# Robot Selection System Prompt
ROBOT_SELECTION_SYSTEM_PROMPT = """
You are an expert robot selection assistant. Given a task description and available robots with their capabilities, limitations, and environments, you must reason about which robot(s) are most suitable.

Respond in the following format:
<reasoning>
Analyze the task requirements, consider each robot's capabilities and limitations, evaluate environment compatibility, and explain your selection logic.
</reasoning>
<answer>
Selected robot(s): [Robot Name(s)]
</answer>
"""

class RobotSelectionRewardFunctions:
    """Reward functions for robot selection reasoning tasks"""
    
    @staticmethod
    def extract_robot_answer(text: str) -> str:
        """Extract robot selection from XML answer tags"""
        try:
            answer = text.split("<answer>")[-1]
            answer = answer.split("</answer>")[0]
            # Extract just the robot names, handle various formats
            answer = answer.replace("Selected robot(s):", "").strip()
            return answer.strip()
        except:
            return ""
    
    @staticmethod
    def correctness_reward_func(prompts, completions, ground_truth_robots, **kwargs) -> List[float]:
        """
        Reward function that checks if the selected robot(s) match ground truth
        """
        responses = [completion[0]["content"] for completion in completions]
        extracted_responses = [RobotSelectionRewardFunctions.extract_robot_answer(r) for r in responses]
        
        rewards = []
        for pred, gt in zip(extracted_responses, ground_truth_robots):
            # Normalize both predictions and ground truth
            pred_robots = set([r.strip() for r in pred.split(",")])
            gt_robots = set([r.strip() for r in gt.split(",")])
            
            # Calculate IoU-style reward
            if len(gt_robots) == 0:
                reward = 0.0
            else:
                intersection = len(pred_robots.intersection(gt_robots))
                union = len(pred_robots.union(gt_robots))
                reward = 2.0 * intersection / len(gt_robots) if intersection == len(gt_robots) else 0.0
            
            rewards.append(reward)
        
        return rewards
    
    @staticmethod
    def reasoning_quality_reward_func(completions, **kwargs) -> List[float]:
        """
        Reward function that evaluates reasoning quality
        """
        responses = [completion[0]["content"] for completion in completions]
        rewards = []
        
        for response in responses:
            reward = 0.0
            
            # Check for proper reasoning structure
            if "<reasoning>" in response and "</reasoning>" in response:
                reasoning_text = response.split("<reasoning>")[1].split("</reasoning>")[0]
                
                # Reward for mentioning key concepts
                key_concepts = [
                    "capabilities", "limitations", "environment", "task", "suitable",
                    "requirement", "analyze", "consider", "evaluate"
                ]
                
                concept_score = sum(1 for concept in key_concepts if concept.lower() in reasoning_text.lower())
                reward += min(concept_score * 0.1, 0.8)  # Max 0.8 for concepts
                
                # Reward for reasoning length (within bounds)
                reasoning_length = len(reasoning_text.split())
                if 20 <= reasoning_length <= 200:
                    reward += 0.2
                elif reasoning_length > 10:
                    reward += 0.1
            
            rewards.append(reward)
        
        return rewards
    
    @staticmethod
    def format_reward_func(completions, **kwargs) -> List[float]:
        """
        Reward function that checks for proper XML formatting
        """
        pattern = r"<reasoning>.*?</reasoning>\s*<answer>.*?</answer>"
        responses = [completion[0]["content"] for completion in completions]
        matches = [bool(re.search(pattern, r, re.DOTALL)) for r in responses]
        return [0.5 if match else 0.0 for match in matches]
    
    @staticmethod
    def robot_validity_reward_func(completions, **kwargs) -> List[float]:
        """
        Reward function that checks if selected robots are valid
        """
        valid_robots = {
            "drone", "underwater robot", "humanoid", "robot with wheels", "robot with legs"
        }
        
        responses = [completion[0]["content"] for completion in completions]
        rewards = []
        
        for response in responses:
            extracted = RobotSelectionRewardFunctions.extract_robot_answer(response)
            selected_robots = [r.strip().lower() for r in extracted.split(",")]
            
            # Check if all selected robots are valid
            valid_count = sum(1 for robot in selected_robots if robot in valid_robots)
            total_count = len(selected_robots)
            
            if total_count == 0:
                reward = 0.0
            else:
                reward = 0.3 * (valid_count / total_count)
            
            rewards.append(reward)
        
        return rewards


class RobotSelectionDataPrep:
    """Data preparation for robot selection tasks"""
    
    @staticmethod
    def load_robot_selection_data(
        single_robot_path: str, 
        multi_robot_path: str,
        use_multi_robot: bool = True
    ) -> Dataset:
        """
        Load and prepare robot selection dataset for GRPO training
        """
        # Load single robot data
        with open(single_robot_path, 'r') as f:
            single_data = json.load(f)
        
        dataset_entries = []
        
        if use_multi_robot:
            # Load multi robot data for more complex reasoning
            with open(multi_robot_path, 'r') as f:
                multi_data = json.load(f)
            
            for entry in multi_data:
                # Use the original single robot output as ground truth
                ground_truth = entry["original_single_robot_output"]
                
                dataset_entries.append({
                    "prompt": [
                        {"role": "system", "content": ROBOT_SELECTION_SYSTEM_PROMPT},
                        {"role": "user", "content": f"{entry['instruction']}\n\nTask: {entry['input']}"}
                    ],
                    "ground_truth_robots": ground_truth,
                    "task_complexity": "multi" if len(entry.get("subtasks", [])) > 1 else "single"
                })
        else:
            # Use single robot data
            for entry in single_data:
                dataset_entries.append({
                    "prompt": [
                        {"role": "system", "content": ROBOT_SELECTION_SYSTEM_PROMPT},
                        {"role": "user", "content": f"{entry['instruction']}\n\nTask: {entry['input']}"}
                    ],
                    "ground_truth_robots": entry["output"],
                    "task_complexity": "single"
                })
        
        return Dataset.from_list(dataset_entries)


class BitMarGRPOTrainer:
    """
    GRPO Trainer specifically for BitMar robot selection reasoning
    """
    
    def __init__(
        self,
        bitmar_model_path: str,
        robot_data_single_path: str,
        robot_data_multi_path: str,
        output_dir: str = "./bitmar_robot_grpo",
        use_multi_robot: bool = True
    ):
        self.bitmar_model_path = bitmar_model_path
        self.robot_data_single_path = robot_data_single_path
        self.robot_data_multi_path = robot_data_multi_path
        self.output_dir = output_dir
        self.use_multi_robot = use_multi_robot
        
        # Initialize reward functions
        self.reward_funcs = [
            RobotSelectionRewardFunctions.correctness_reward_func,
            RobotSelectionRewardFunctions.reasoning_quality_reward_func,
            RobotSelectionRewardFunctions.format_reward_func,
            RobotSelectionRewardFunctions.robot_validity_reward_func,
        ]
        
        logger.info(f"Initialized BitMar GRPO trainer with {len(self.reward_funcs)} reward functions")
    
    def prepare_data(self) -> Dataset:
        """Prepare robot selection dataset for training"""
        return RobotSelectionDataPrep.load_robot_selection_data(
            self.robot_data_single_path,
            self.robot_data_multi_path,
            self.use_multi_robot
        )
    
    def create_grpo_config(
        self,
        num_train_epochs: int = 3,
        per_device_train_batch_size: int = 4,
        learning_rate: float = 1e-6,
        max_length: int = 1024,
        **kwargs
    ) -> GRPOConfig:
        """Create GRPO configuration for robot selection training"""
        return GRPOConfig(
            output_dir=self.output_dir,
            num_train_epochs=num_train_epochs,
            per_device_train_batch_size=per_device_train_batch_size,
            learning_rate=learning_rate,
            max_length=max_length,
            logging_steps=10,
            save_strategy="epoch",
            evaluation_strategy="epoch",
            warmup_ratio=0.1,
            lr_scheduler_type="cosine",
            gradient_accumulation_steps=2,
            dataloader_drop_last=True,
            remove_unused_columns=False,
            **kwargs
        )
    
    def train(self, grpo_config: Optional[GRPOConfig] = None) -> GRPOTrainer:
        """
        Train BitMar model with GRPO for robot selection reasoning
        """
        if grpo_config is None:
            grpo_config = self.create_grpo_config()
        
        # Prepare data
        train_dataset = self.prepare_data()
        logger.info(f"Loaded {len(train_dataset)} robot selection examples")
        
        # Initialize tokenizer
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        tokenizer.pad_token = tokenizer.eos_token
        
        # Create GRPO trainer
        trainer = GRPOTrainer(
            model=self.bitmar_model_path,
            processing_class=tokenizer,
            reward_funcs=self.reward_funcs,
            args=grpo_config,
            train_dataset=train_dataset,
        )
        
        logger.info("Starting GRPO training for robot selection reasoning...")
        trainer.train()
        
        # Save the trained model
        trainer.save_model(grpo_config.output_dir)
        logger.info(f"GRPO-trained BitMar model saved to {grpo_config.output_dir}")
        
        return trainer


def create_robot_grpo_trainer(
    bitmar_checkpoint_path: str,
    single_robot_data_path: str = "../robot_selection_data/data/Single-Robot-Selection/single_robot_selection_dataset.json",
    multi_robot_data_path: str = "../robot_selection_data/data/Multi-Robot-Selection/multi_robot_selection_dataset.json",
    output_dir: str = "./bitmar_robot_reasoning",
    **grpo_kwargs
) -> BitMarGRPOTrainer:
    """
    Factory function to create BitMar GRPO trainer for robot selection
    """
    trainer = BitMarGRPOTrainer(
        bitmar_model_path=bitmar_checkpoint_path,
        robot_data_single_path=single_robot_data_path,
        robot_data_multi_path=multi_robot_data_path,
        output_dir=output_dir,
        use_multi_robot=True
    )
    
    return trainer


if __name__ == "__main__":
    # Example usage
    trainer = create_robot_grpo_trainer(
        bitmar_checkpoint_path="./checkpoints_coco/best_model",
        output_dir="./bitmar_robot_reasoning"
    )
    
    # Train with GRPO
    grpo_trainer = trainer.train()
    print("Robot selection reasoning training completed!")