"""
COCO Dataset processing for BitMar
Handles COCO dataset with captions and supports multiple image formats
"""

import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from typing import Dict, List, Tuple, Optional
import logging
import random
import os
from pathlib import Path
from PIL import Image
import torchvision.transforms as transforms

logger = logging.getLogger(__name__)


class COCODataset(Dataset):
    """COCO Dataset with Vision Feature Extraction"""

    def __init__(
        self,
        dataset_dir: str,
        tokenizer_name: str = "gpt2",
        vision_model_name: str = "facebook/dinov2-base",
        max_seq_length: int = 512,
        split: str = "train",
        max_samples: Optional[int] = None,
        use_dummy_vision: bool = False,  # This parameter is now ignored - always False
        extract_vision_features: bool = True
    ):
        self.dataset_dir = Path(dataset_dir)
        self.max_seq_length = max_seq_length
        self.split = split
        # Force real vision features only - no dummy mode allowed
        self.use_dummy_vision = False
        self.extract_vision_features = extract_vision_features

        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Initialize vision model - MUST succeed or crash
        if not extract_vision_features:
            raise ValueError("❌ CRITICAL: extract_vision_features=False is not allowed. Real vision features are mandatory.")

        try:
            logger.info(f"🔥 Loading vision model (REAL FEATURES ONLY): {vision_model_name}")
            self.vision_model = AutoModel.from_pretrained(vision_model_name)
            self.vision_model.eval()

            # Image preprocessing
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                                     0.229, 0.224, 0.225])
            ])
            logger.info(f"✅ Vision model loaded successfully: {vision_model_name}")
        except Exception as e:
            raise RuntimeError(f"❌ CRITICAL FAILURE: Failed to load vision model '{vision_model_name}'. Real vision features are mandatory. Error: {e}")

        # Load COCO data
        self._load_coco_data()

        # Create indices for this split
        if split == "train":
            # Use full dataset for training
            self.indices = list(range(len(self.image_caption_pairs)))
        else:
            # Use a small subset for validation (10% of data)
            total_samples = len(self.image_caption_pairs)
            val_size = max(100, total_samples // 10)  # At least 100 samples
            random.seed(42)
            self.indices = random.sample(
                range(total_samples), min(val_size, total_samples))

        # Limit samples if specified
        if max_samples is not None and len(self.indices) > max_samples:
            random.seed(42)
            self.indices = random.sample(self.indices, max_samples)

        logger.info(f"Loaded {len(self.indices)} samples for {split} split")

    def _load_coco_data(self):
        """Load COCO image-caption pairs"""
        logger.info("Loading COCO dataset...")

        # Load aligned pairs
        aligned_pairs_file = self.dataset_dir / "aligned_pairs.json"
        coco_pairs_file = self.dataset_dir / "coco_aligned_pairs.json"

        # Try to load from aligned_pairs.json first, then coco_aligned_pairs.json
        pairs_file = aligned_pairs_file if aligned_pairs_file.exists() else coco_pairs_file

        if not pairs_file.exists():
            raise FileNotFoundError(
                f"No COCO pairs file found. Please run download_coco_supplement.py first.")

        with open(pairs_file, 'r', encoding='utf-8') as f:
            self.image_caption_pairs = json.load(f)

        logger.info(
            f"Loaded {len(self.image_caption_pairs)} COCO image-caption pairs")

    def _load_image(self, image_path: str) -> torch.Tensor:
        """Load and preprocess image - MUST succeed or crash"""
        # Support multiple image formats
        img_path = Path(image_path)

        # Check if file exists with different extensions
        if not img_path.exists():
            # Try different extensions
            base_path = img_path.parent / img_path.stem
            for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG', '.webp', '.WEBP']:
                candidate_path = base_path.with_suffix(ext)
                if candidate_path.exists():
                    img_path = candidate_path
                    break
            else:
                raise FileNotFoundError(f"❌ CRITICAL: Image not found at any path: {image_path}")

        # Load image - MUST succeed
        try:
            with Image.open(img_path) as img:
                # Convert to RGB to handle different formats
                img = img.convert('RGB')

                # Apply transforms for vision model - always required
                img_tensor = self.transform(img)
                logger.debug(f"✅ Successfully loaded and preprocessed image: {img_path}")
                return img_tensor
        except Exception as e:
            raise RuntimeError(f"❌ CRITICAL: Failed to load/preprocess image {img_path}. Error: {e}")

    def _extract_vision_features(self, image_tensor: torch.Tensor) -> torch.Tensor:
        """Extract vision features using DiNOv2 - MUST succeed or crash"""
        if self.vision_model is None:
            raise RuntimeError(f"❌ CRITICAL: Vision model is None. This should never happen with mandatory real features.")

        try:
            with torch.no_grad():
                # Add batch dimension
                image_batch = image_tensor.unsqueeze(0)

                # Extract features using DiNOv2
                outputs = self.vision_model(image_batch)
                features = outputs.last_hidden_state

                # Global average pooling to get single feature vector
                pooled_features = features.mean(dim=1).squeeze(0)

                logger.debug(f"✅ Successfully extracted vision features: shape {pooled_features.shape}")
                return pooled_features
        except Exception as e:
            raise RuntimeError(f"❌ CRITICAL: Failed to extract vision features from DiNOv2. Error: {e}")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        pair = self.image_caption_pairs[real_idx]

        # Load and process image
        image_tensor = self._load_image(pair['image_path'])

        # Extract vision features
        vision_features = self._extract_vision_features(image_tensor)

        # Process caption
        caption = pair['caption']

        # Tokenize caption
        inputs = self.tokenizer(
            caption,
            max_length=self.max_seq_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'input_ids': inputs['input_ids'].squeeze(0),
            'attention_mask': inputs['attention_mask'].squeeze(0),
            'vision_features': vision_features,
            'caption': caption,
            'image_id': pair.get('image_id', f'coco_{real_idx}'),
            'dataset': pair.get('dataset', 'coco')
        }


def create_coco_data_module(
    dataset_dir: str,
    tokenizer_name: str = "gpt2",
    vision_model_name: str = "facebook/dinov2-base",
    batch_size: int = 16,
    max_seq_length: int = 512,
    max_samples: Optional[int] = None,
    use_dummy_vision: bool = False,  # This parameter is now ignored - always False
    extract_vision_features: bool = True,  # This parameter is now ignored - always True
    num_workers: int = 4
) -> Tuple[DataLoader, DataLoader]:
    """Create COCO data loaders - REAL VISION FEATURES ONLY"""

    # Force real vision features only
    use_dummy_vision = False
    extract_vision_features = True

    logger.info("Creating COCO data module with MANDATORY REAL VISION FEATURES...")

    # Create datasets - will crash if vision model fails to load
    train_dataset = COCODataset(
        dataset_dir=dataset_dir,
        tokenizer_name=tokenizer_name,
        vision_model_name=vision_model_name,
        max_seq_length=max_seq_length,
        split="train",
        max_samples=max_samples,
        use_dummy_vision=use_dummy_vision,  # Always False
        extract_vision_features=extract_vision_features  # Always True
    )

    val_dataset = COCODataset(
        dataset_dir=dataset_dir,
        tokenizer_name=tokenizer_name,
        vision_model_name=vision_model_name,
        max_seq_length=max_seq_length,
        split="val",
        max_samples=max_samples // 10 if max_samples else None,
        use_dummy_vision=use_dummy_vision,  # Always False
        extract_vision_features=extract_vision_features  # Always True
    )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False
    )

    logger.info(f"✅ COCO data module created with REAL VISION FEATURES ONLY")
    logger.info(f"   📊 Training samples: {len(train_dataset)}")
    logger.info(f"   📊 Validation samples: {len(val_dataset)}")
    logger.info(f"   🔧 Batch size: {batch_size}")
    logger.info(f"   🎯 Max sequence length: {max_seq_length}")
    logger.info(f"   👁️ Vision features: REAL DiNOv2 extraction (no fallbacks)")

    return train_loader, val_loader


# Compatibility wrapper for existing code
def create_data_module(*args, **kwargs):
    """Compatibility wrapper for existing code"""
    return create_coco_data_module(*args, **kwargs)
