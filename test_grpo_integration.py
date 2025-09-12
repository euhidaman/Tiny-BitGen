"""
Test GRPO Reasoning Integration with BitMar Model
Verifies that the GRPO reasoning module works correctly in the BitMar architecture
"""

import torch
import torch.nn.functional as F
import yaml
from pathlib import Path
import sys
import logging

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_grpo_integration():
    """Test GRPO reasoning integration with BitMar model"""
    
    # Load configuration
    config_path = "configs/bitmar_coco.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Ensure GRPO is enabled
    config['grpo_reasoning']['enabled'] = True
    
    logger.info("🧪 Testing GRPO reasoning integration...")
    
    try:
        # Import BitMar model
        from src.model import create_bitmar_model
        
        # Create model with GRPO reasoning
        model = create_bitmar_model(config)
        logger.info("✅ BitMar model created successfully")
        
        # Check if GRPO reasoning is enabled
        if hasattr(model, 'grpo_reasoning') and model.grpo_reasoning is not None:
            logger.info("✅ GRPO reasoning module integrated")
        else:
            logger.error("❌ GRPO reasoning module not integrated")
            return False
        
        # Create test inputs
        batch_size = 2
        seq_len = 64
        vocab_size = config['vocab_size']
        vision_dim = config['vision_encoder_dim']
        
        # Generate test data
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len)
        vision_features = torch.randn(batch_size, vision_dim)
        labels = torch.randint(0, vocab_size, (batch_size, seq_len))
        
        logger.info(f"📊 Test data shapes:")
        logger.info(f"   input_ids: {input_ids.shape}")
        logger.info(f"   vision_features: {vision_features.shape}")
        
        # Test forward pass with GRPO reasoning
        model.eval()
        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                vision_features=vision_features,
                mode="inference"
            )
        
        logger.info("✅ Forward pass completed")
        
        # Check GRPO reasoning outputs
        if 'grpo_reasoning_output' in outputs and outputs['grpo_reasoning_output'] is not None:
            grpo_output = outputs['grpo_reasoning_output']
            logger.info("✅ GRPO reasoning outputs found")
            
            # Check robot selection
            if 'robot_selection' in grpo_output:
                robot_probs = grpo_output['robot_selection']
                logger.info(f"   Robot selection shape: {robot_probs.shape}")
                logger.info(f"   Robot probabilities: {robot_probs[0].tolist()}")
                
                # Extract robot selections
                robot_selections = model.extract_robot_selections(outputs)
                logger.info(f"   Robot selections: {robot_selections}")
                
            # Check reasoning quality
            if 'reasoning_quality' in grpo_output:
                quality = grpo_output['reasoning_quality']
                logger.info(f"   Reasoning quality shape: {quality.shape}")
                
            # Check thought scores
            if 'thought_scores' in grpo_output:
                scores = grpo_output['thought_scores']
                logger.info(f"   Thought scores shape: {scores.shape}")
                
        else:
            logger.error("❌ GRPO reasoning outputs not found")
            return False
        
        # Test training mode with loss computation
        model.train()
        train_outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            vision_features=vision_features,
            labels=labels,
            mode="train"
        )
        
        logger.info("✅ Training forward pass completed")
        
        # Check loss computation
        if 'loss' in train_outputs and train_outputs['loss'] is not None:
            total_loss = train_outputs['loss']
            logger.info(f"   Total loss: {total_loss.item():.4f}")
            
            # Check GRPO reasoning loss
            if 'grpo_reasoning_loss' in train_outputs:
                grpo_loss = train_outputs['grpo_reasoning_loss']
                if grpo_loss is not None:
                    logger.info(f"   GRPO reasoning loss: {grpo_loss.item():.4f}")
                else:
                    logger.info("   GRPO reasoning loss: None (may be expected)")
            
        else:
            logger.error("❌ Training loss not computed")
            return False
        
        # Test reasoning analysis
        analysis = model.get_reasoning_analysis(outputs)
        logger.info("📊 Reasoning Analysis:")
        logger.info(f"   Reasoning available: {analysis.get('reasoning_available', False)}")
        
        if 'robot_probabilities' in analysis:
            logger.info("   Robot probabilities:")
            for robot, prob in analysis['robot_probabilities'].items():
                logger.info(f"     {robot}: {prob:.4f}")
        
        if 'reasoning_quality' in analysis:
            logger.info("   Reasoning quality:")
            for aspect, score in analysis['reasoning_quality'].items():
                logger.info(f"     {aspect}: {score:.4f}")
        
        logger.info("🎉 GRPO integration test completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"❌ GRPO integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_robot_task_reasoning():
    """Test GRPO reasoning with specific robot selection tasks"""
    
    logger.info("🤖 Testing robot task reasoning...")
    
    # Sample robot selection tasks
    test_tasks = [
        "Inspect underwater pipelines for damage and leaks",
        "Survey a large outdoor construction site from above", 
        "Navigate through a crowded indoor shopping mall",
        "Transport heavy equipment across rough mountain terrain",
        "Perform delicate manipulation tasks in a laboratory"
    ]
    
    expected_robots = [
        "Underwater Robot",
        "Drone", 
        "Humanoid",
        "Robot with Legs",
        "Humanoid"
    ]
    
    try:
        # Load configuration
        config_path = "configs/bitmar_coco.yaml"
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        config['grpo_reasoning']['enabled'] = True
        
        # Create model
        from src.model import create_bitmar_model
        model = create_bitmar_model(config)
        
        # Create tokenizer for encoding tasks
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained('gpt2')
        tokenizer.pad_token = tokenizer.eos_token
        
        model.eval()
        
        correct_predictions = 0
        
        for i, (task, expected) in enumerate(zip(test_tasks, expected_robots)):
            logger.info(f"\n📋 Task {i+1}: {task}")
            logger.info(f"   Expected robot: {expected}")
            
            # Encode task
            encoded = tokenizer(task, return_tensors='pt', padding='max_length', 
                              max_length=64, truncation=True)
            
            # Create dummy vision features
            vision_features = torch.randn(1, config['vision_encoder_dim'])
            
            # Forward pass
            with torch.no_grad():
                outputs = model(
                    input_ids=encoded['input_ids'],
                    attention_mask=encoded['attention_mask'],
                    vision_features=vision_features,
                    mode="inference"
                )
            
            # Extract robot selection
            robot_selections = model.extract_robot_selections(outputs)
            predicted = robot_selections[0] if robot_selections else "No prediction"
            
            logger.info(f"   Predicted robot: {predicted}")
            
            # Check if prediction is reasonable (contains expected robot)
            if expected.lower() in predicted.lower():
                correct_predictions += 1
                logger.info("   ✅ Prediction contains expected robot")
            else:
                logger.info("   ⚠️  Prediction differs from expected")
        
        accuracy = correct_predictions / len(test_tasks)
        logger.info(f"\n📊 Robot selection accuracy: {accuracy:.2%} ({correct_predictions}/{len(test_tasks)})")
        
        if accuracy >= 0.4:  # 40% threshold for basic functionality
            logger.info("🎉 Robot task reasoning test passed!")
            return True
        else:
            logger.warning("⚠️  Robot task reasoning accuracy below threshold")
            return False
            
    except Exception as e:
        logger.error(f"❌ Robot task reasoning test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all GRPO integration tests"""
    logger.info("🚀 Starting GRPO integration tests...")
    
    test_results = []
    
    # Test 1: Basic GRPO integration
    logger.info("\n" + "="*50)
    logger.info("Test 1: Basic GRPO Integration")
    logger.info("="*50)
    test_results.append(test_grpo_integration())
    
    # Test 2: Robot task reasoning (only if basic integration works)
    if test_results[0]:
        logger.info("\n" + "="*50)
        logger.info("Test 2: Robot Task Reasoning")
        logger.info("="*50)
        test_results.append(test_robot_task_reasoning())
    else:
        logger.warning("⚠️  Skipping robot task reasoning test due to basic integration failure")
        test_results.append(False)
    
    # Summary
    logger.info("\n" + "="*50)
    logger.info("Test Summary")
    logger.info("="*50)
    
    test_names = ["Basic GRPO Integration", "Robot Task Reasoning"]
    
    for i, (name, result) in enumerate(zip(test_names, test_results)):
        status = "✅ PASSED" if result else "❌ FAILED"
        logger.info(f"Test {i+1}: {name} - {status}")
    
    passed_tests = sum(test_results)
    total_tests = len(test_results)
    
    logger.info(f"\nOverall: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        logger.info("🎉 All GRPO integration tests passed!")
        return True
    else:
        logger.warning("⚠️  Some GRPO integration tests failed")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)