#!/usr/bin/env python3
"""
Test script for FIBER-inspired cross-modal fusion integration
"""

import torch
import yaml
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from model import BitMarModel
from fiber_fusion import FIBERCrossModalFusion

def test_fiber_fusion_standalone():
    """Test the FIBER fusion module standalone"""
    print("Testing FIBER fusion module standalone...")
    
    # Create test configuration
    text_dim = 128
    vision_dim = 128
    num_heads = 4
    num_layers = 2
    batch_size = 2
    text_seq_len = 10
    vision_seq_len = 1  # Single vision token for now
    
    # Initialize fusion module
    fusion = FIBERCrossModalFusion(
        text_dim=text_dim,
        vision_dim=vision_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        hidden_dim=128,
        dropout=0.1
    )
    
    # Create test inputs
    text_features = torch.randn(batch_size, text_seq_len, text_dim)
    vision_features = torch.randn(batch_size, vision_dim)  # Single token
    text_mask = torch.ones(batch_size, text_seq_len, dtype=torch.bool)
    vision_mask = torch.ones(batch_size, vision_dim, dtype=torch.bool)
    
    # Forward pass
    try:
        result = fusion(vision_features, text_features, text_mask)
        print(f"✅ FIBER fusion forward pass successful!")
        print(f"   Vision features shape: {result['vision_features'].shape}")
        print(f"   Text features shape: {result['text_features'].shape}")
        print(f"   Cross-modal similarity shape: {result['cross_modal_similarity'].shape}")
        print(f"   Attention weights available: {len(result['attention_weights'])} layers")
        return True
    except Exception as e:
        print(f"❌ FIBER fusion failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_bitmar_model_integration():
    """Test the full BitMar model with FIBER fusion"""
    print("\nTesting BitMar model with FIBER integration...")
    
    # Load configuration
    config_path = "configs/bitmar_coco.yaml"
    if not os.path.exists(config_path):
        print(f"❌ Config file not found: {config_path}")
        return False
        
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)['model']
    
    # Initialize model
    try:
        model = BitMarModel(config)
        print(f"✅ BitMar model initialized successfully!")
        
        # Create test inputs
        batch_size = 2
        seq_len = 20
        vocab_size = config['vocab_size']
        vision_dim = config['vision_encoder_dim']
        
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
        vision_features = torch.randn(batch_size, vision_dim)
        labels = torch.randint(0, vocab_size, (batch_size, seq_len))
        
        # Forward pass
        model.eval()
        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                vision_features=vision_features,
                labels=labels,
                mode="eval"
            )
        
        print(f"✅ BitMar model forward pass successful!")
        print(f"   Output keys: {list(outputs.keys())}")
        print(f"   Logits shape: {outputs['logits'].shape}")
        print(f"   Fused features shape: {outputs['fused_features'].shape}")
        print(f"   Cross attention keys: {list(outputs['cross_attention'].keys())}")
        print(f"   Attention weights available: {'attention_weights' in outputs['cross_attention']}")
        print(f"   Cross-modal similarity available: {'cross_modal_similarity' in outputs['cross_attention']}")
        
        return True
        
    except Exception as e:
        print(f"❌ BitMar model integration failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("🧪 Testing FIBER Integration in Tiny-BitGen")
    print("=" * 50)
    
    # Test standalone fusion
    fusion_success = test_fiber_fusion_standalone()
    
    # Test full model integration
    model_success = test_bitmar_model_integration()
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 Test Summary:")
    print(f"   FIBER Fusion Module: {'✅ PASS' if fusion_success else '❌ FAIL'}")
    print(f"   BitMar Integration: {'✅ PASS' if model_success else '❌ FAIL'}")
    
    if fusion_success and model_success:
        print("\n🎉 All tests passed! FIBER integration successful!")
        print("   You can now train with the advanced cross-modal fusion:")
        print("   python train.py --config configs/bitmar_coco.yaml --mode coco")
    else:
        print("\n⚠️  Some tests failed. Please check the errors above.")
        
    return fusion_success and model_success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)