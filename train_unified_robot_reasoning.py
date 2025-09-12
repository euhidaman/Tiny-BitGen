"""
Unified Training Script for BitMar with Robot Reasoning
Combines COCO vision-language training with GRPO robot selection reasoning
"""

import argparse
import yaml
import torch
import logging
from pathlib import Path
from typing import Dict, Optional
import os
import sys

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.robot_grpo_training import create_robot_grpo_trainer, BitMarGRPOTrainer
from src.robot_reasoning_integration import create_robot_reasoner, BitMarRobotReasoner
from train import main as bitmar_main  # Import existing BitMar training

logger = logging.getLogger(__name__)


class UnifiedBitMarTrainer:
    """
    Unified trainer for BitMar: Vision-Language + Robot Reasoning
    """
    
    def __init__(
        self,
        config_path: str,
        robot_data_dir: str = "../robot_selection_data/data",
        output_dir: str = "./unified_bitmar_outputs"
    ):
        self.config_path = config_path
        self.robot_data_dir = Path(robot_data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Set up paths
        self.coco_checkpoint_dir = self.output_dir / "coco_checkpoints"
        self.robot_reasoning_dir = self.output_dir / "robot_reasoning" 
        self.final_model_dir = self.output_dir / "final_unified_model"
        
        # Robot data paths
        self.single_robot_path = self.robot_data_dir / "Single-Robot-Selection" / "single_robot_selection_dataset.json"
        self.multi_robot_path = self.robot_data_dir / "Multi-Robot-Selection" / "multi_robot_selection_dataset.json"
        
        logger.info(f"Unified BitMar trainer initialized")
        logger.info(f"COCO checkpoints: {self.coco_checkpoint_dir}")
        logger.info(f"Robot reasoning: {self.robot_reasoning_dir}")
        logger.info(f"Final model: {self.final_model_dir}")
    
    def stage_1_coco_training(self, resume_from_checkpoint: Optional[str] = None) -> str:
        """
        Stage 1: Train BitMar on COCO dataset for vision-language understanding
        """
        logger.info("🚀 Stage 1: Starting COCO vision-language training...")
        
        # Update config for COCO training
        self.config['output']['checkpoint_dir'] = str(self.coco_checkpoint_dir)
        
        # Save updated config
        temp_config_path = self.output_dir / "stage1_coco_config.yaml"
        with open(temp_config_path, 'w') as f:
            yaml.dump(self.config, f)
        
        # Run COCO training using existing train.py
        try:
            import subprocess
            cmd = [
                "python", "train.py",
                "--config", str(temp_config_path),
                "--mode", "coco"
            ]
            
            if resume_from_checkpoint:
                cmd.extend(["--resume", resume_from_checkpoint])
            
            logger.info(f"Running command: {' '.join(cmd)}")
            result = subprocess.run(cmd, cwd=".", capture_output=True, text=True)
            
            if result.returncode != 0:
                logger.error(f"COCO training failed: {result.stderr}")
                raise RuntimeError("COCO training failed")
            
            logger.info("✅ Stage 1: COCO training completed successfully")
            
            # Find best checkpoint
            best_checkpoint = self._find_best_checkpoint(self.coco_checkpoint_dir)
            logger.info(f"Best COCO checkpoint: {best_checkpoint}")
            
            return str(best_checkpoint)
            
        except Exception as e:
            logger.error(f"Stage 1 COCO training failed: {e}")
            raise
    
    def stage_2_robot_grpo_training(self, coco_checkpoint_path: str) -> str:
        """
        Stage 2: Train robot selection reasoning with GRPO
        """
        logger.info("🤖 Stage 2: Starting robot selection GRPO training...")
        
        # Verify robot data exists
        if not self.single_robot_path.exists():
            raise FileNotFoundError(f"Single robot data not found: {self.single_robot_path}")
        if not self.multi_robot_path.exists():
            raise FileNotFoundError(f"Multi robot data not found: {self.multi_robot_path}")
        
        # Create GRPO trainer
        grpo_trainer = create_robot_grpo_trainer(
            bitmar_checkpoint_path=coco_checkpoint_path,
            single_robot_data_path=str(self.single_robot_path),
            multi_robot_data_path=str(self.multi_robot_path),
            output_dir=str(self.robot_reasoning_dir)
        )
        
        # Configure GRPO training
        grpo_config = grpo_trainer.create_grpo_config(
            num_train_epochs=3,
            per_device_train_batch_size=2,  # Small batch for reasoning tasks
            learning_rate=5e-7,  # Lower LR for fine-tuning
            max_length=1024,
            gradient_accumulation_steps=4,
            warmup_ratio=0.1,
            logging_steps=5,
            save_strategy="epoch",
            evaluation_strategy="no"  # No eval set for robot data
        )
        
        # Train with GRPO
        try:
            trained_grpo = grpo_trainer.train(grpo_config)
            logger.info("✅ Stage 2: Robot GRPO training completed successfully")
            
            return str(self.robot_reasoning_dir)
            
        except Exception as e:
            logger.error(f"Stage 2 GRPO training failed: {e}")
            raise
    
    def stage_3_unified_model_creation(
        self, 
        coco_checkpoint_path: str, 
        grpo_model_path: str
    ) -> BitMarRobotReasoner:
        """
        Stage 3: Create unified model combining vision-language and robot reasoning
        """
        logger.info("🔧 Stage 3: Creating unified BitMar robot reasoner...")
        
        try:
            # Create unified model
            unified_model = create_robot_reasoner(
                base_model_config_path=self.config_path,
                grpo_model_path=grpo_model_path,
                bitmar_checkpoint_path=coco_checkpoint_path
            )
            
            # Save unified model
            self.final_model_dir.mkdir(exist_ok=True)
            torch.save(unified_model.state_dict(), self.final_model_dir / "unified_model.pt")
            
            # Save model configuration
            unified_config = {
                "base_model_config": self.config,
                "coco_checkpoint": coco_checkpoint_path,
                "grpo_model_path": grpo_model_path,
                "robot_reasoning_config": unified_model.robot_reasoning_config,
                "model_type": "BitMarRobotReasoner"
            }
            
            with open(self.final_model_dir / "unified_config.yaml", 'w') as f:
                yaml.dump(unified_config, f)
            
            logger.info("✅ Stage 3: Unified model creation completed successfully")
            logger.info(f"Unified model saved to: {self.final_model_dir}")
            
            return unified_model
            
        except Exception as e:
            logger.error(f"Stage 3 unified model creation failed: {e}")
            raise
    
    def train_full_pipeline(
        self, 
        resume_coco_checkpoint: Optional[str] = None,
        skip_coco: bool = False,
        skip_grpo: bool = False
    ) -> BitMarRobotReasoner:
        """
        Run the complete training pipeline: COCO + GRPO + Unification
        """
        logger.info("🎯 Starting unified BitMar training pipeline...")
        logger.info("Pipeline: COCO Vision-Language → Robot GRPO → Unified Model")
        
        try:
            # Stage 1: COCO Training
            if skip_coco and resume_coco_checkpoint:
                logger.info("⏭️  Skipping COCO training, using provided checkpoint")
                coco_checkpoint = resume_coco_checkpoint
            else:
                coco_checkpoint = self.stage_1_coco_training(resume_coco_checkpoint)
            
            # Stage 2: Robot GRPO Training  
            if skip_grpo and self.robot_reasoning_dir.exists():
                logger.info("⏭️  Skipping GRPO training, using existing model")
                grpo_model_path = str(self.robot_reasoning_dir)
            else:
                grpo_model_path = self.stage_2_robot_grpo_training(coco_checkpoint)
            
            # Stage 3: Unified Model Creation
            unified_model = self.stage_3_unified_model_creation(coco_checkpoint, grpo_model_path)
            
            logger.info("🎉 Complete training pipeline finished successfully!")
            logger.info(f"Unified BitMar model with robot reasoning available at: {self.final_model_dir}")
            
            return unified_model
            
        except Exception as e:
            logger.error(f"Training pipeline failed: {e}")
            raise
    
    def _find_best_checkpoint(self, checkpoint_dir: Path) -> Path:
        """Find the best checkpoint in directory"""
        checkpoint_files = list(checkpoint_dir.glob("*.pt"))
        if not checkpoint_files:
            checkpoint_files = list(checkpoint_dir.glob("**/pytorch_model.bin"))
        
        if not checkpoint_files:
            raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}")
        
        # Return the most recent checkpoint
        return max(checkpoint_files, key=lambda p: p.stat().st_mtime)
    
    def test_unified_model(self, model: Optional[BitMarRobotReasoner] = None) -> Dict:
        """
        Test the unified model with sample robot selection tasks
        """
        if model is None:
            # Load saved model
            model = create_robot_reasoner(
                base_model_config_path=self.config_path,
                grpo_model_path=str(self.robot_reasoning_dir),
                bitmar_checkpoint_path=str(self._find_best_checkpoint(self.coco_checkpoint_dir))
            )
        
        test_tasks = [
            "Inspect underwater pipelines for damage and leaks",
            "Survey a large outdoor construction site from above",
            "Navigate through a crowded indoor shopping mall",
            "Transport heavy equipment across rough mountain terrain",
            "Perform delicate manipulation tasks in a laboratory"
        ]
        
        results = {}
        logger.info("🧪 Testing unified model with robot selection tasks...")
        
        for i, task in enumerate(test_tasks):
            try:
                reasoning_result = model.generate_robot_reasoning(
                    task_description=task,
                    return_reasoning_steps=True
                )
                
                results[f"test_{i+1}"] = {
                    "task": task,
                    "selected_robots": reasoning_result["selected_robots"],
                    "reasoning": reasoning_result["reasoning"][:200] + "..." if len(reasoning_result["reasoning"]) > 200 else reasoning_result["reasoning"],
                    "confidence": model._compute_reasoning_confidence(reasoning_result)
                }
                
                logger.info(f"✅ Test {i+1}: {task}")
                logger.info(f"   Selected: {reasoning_result['selected_robots']}")
                
            except Exception as e:
                logger.error(f"❌ Test {i+1} failed: {e}")
                results[f"test_{i+1}"] = {"error": str(e)}
        
        # Save test results
        with open(self.final_model_dir / "test_results.yaml", 'w') as f:
            yaml.dump(results, f)
        
        logger.info(f"Test results saved to: {self.final_model_dir / 'test_results.yaml'}")
        return results


def main():
    """Main training script for unified BitMar"""
    parser = argparse.ArgumentParser(description="Unified BitMar Training: COCO + Robot Reasoning")
    parser.add_argument("--config", type=str, default="configs/bitmar_coco.yaml", 
                       help="Path to BitMar configuration file")
    parser.add_argument("--robot-data-dir", type=str, default="../robot_selection_data/data",
                       help="Directory containing robot selection datasets")
    parser.add_argument("--output-dir", type=str, default="./unified_bitmar_outputs",
                       help="Output directory for all training artifacts")
    parser.add_argument("--resume-coco", type=str, default=None,
                       help="Resume COCO training from checkpoint")
    parser.add_argument("--skip-coco", action="store_true",
                       help="Skip COCO training (requires --resume-coco)")
    parser.add_argument("--skip-grpo", action="store_true", 
                       help="Skip GRPO training (use existing)")
    parser.add_argument("--test-only", action="store_true",
                       help="Only run testing on existing model")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f"{args.output_dir}/training.log"),
            logging.StreamHandler()
        ]
    )
    
    # Create trainer
    trainer = UnifiedBitMarTrainer(
        config_path=args.config,
        robot_data_dir=args.robot_data_dir,
        output_dir=args.output_dir
    )
    
    try:
        if args.test_only:
            # Only run testing
            results = trainer.test_unified_model()
            print("Testing completed. Results saved.")
        else:
            # Run full training pipeline
            unified_model = trainer.train_full_pipeline(
                resume_coco_checkpoint=args.resume_coco,
                skip_coco=args.skip_coco,
                skip_grpo=args.skip_grpo
            )
            
            # Test the trained model
            test_results = trainer.test_unified_model(unified_model)
            
            print("\n🎉 Unified BitMar training completed successfully!")
            print(f"Model saved to: {trainer.final_model_dir}")
            print("\nModel capabilities:")
            print("✅ Vision-language understanding (COCO)")
            print("✅ Robot selection reasoning (GRPO)")
            print("✅ Multimodal grounded reasoning")
            
    except Exception as e:
        logger.error(f"Training failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()