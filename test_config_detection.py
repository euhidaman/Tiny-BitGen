"""
Simple test script to validate training mode detection
"""

import yaml
import sys
from pathlib import Path

def test_config_detection():
    """Test automatic training mode detection"""
    
    # Test COCO config
    try:
        with open('configs/bitmar_coco.yaml', 'r') as f:
            coco_config = yaml.safe_load(f)
        
        print("✅ COCO Config loaded successfully")
        print(f"   - Has token_constraints: {'token_constraints' in coco_config}")
        if 'token_constraints' in coco_config:
            print(f"   - Token constraints enabled: {coco_config['token_constraints'].get('enabled', False)}")
        
        print(f"   - Dataset: {coco_config.get('data', {}).get('dataset_name', 'Unknown')}")
        
    except Exception as e:
        print(f"❌ Failed to load COCO config: {e}")
        return False
    
    # Test BabyLM config
    try:
        with open('configs/bitmar_100M_tokens.yaml', 'r') as f:
            babylm_config = yaml.safe_load(f)
        
        print("\n✅ BabyLM Config loaded successfully")
        print(f"   - Has token_constraints: {'token_constraints' in babylm_config}")
        if 'token_constraints' in babylm_config:
            print(f"   - Token constraints enabled: {babylm_config['token_constraints'].get('enabled', False)}")
            print(f"   - Target tokens: {babylm_config['token_constraints'].get('total_tokens', 'Unknown'):,}")
        
        print(f"   - Dataset: {babylm_config.get('data', {}).get('dataset_name', 'Unknown')}")
        
    except Exception as e:
        print(f"❌ Failed to load BabyLM config: {e}")
        return False
    
    # Test detection logic
    def detect_training_mode(config):
        """Replicate the detection logic from train.py"""
        token_constraints = config.get('token_constraints', {})
        return 'babylm' if token_constraints.get('enabled', False) else 'coco'
    
    coco_mode = detect_training_mode(coco_config)
    babylm_mode = detect_training_mode(babylm_config)
    
    print(f"\n🎯 Mode Detection Results:")
    print(f"   - COCO config detected as: {coco_mode}")
    print(f"   - BabyLM config detected as: {babylm_mode}")
    
    # Validate results
    if coco_mode == 'coco' and babylm_mode == 'babylm':
        print("\n✅ All tests passed! Training mode detection works correctly.")
        return True
    else:
        print("\n❌ Training mode detection failed!")
        return False

if __name__ == "__main__":
    success = test_config_detection()
    sys.exit(0 if success else 1)