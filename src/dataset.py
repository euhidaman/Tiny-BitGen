"""
Dataset processing for BitMar
Handles complete BabyLM multimodal dataset (Conceptual Captions + Localized Narratives)
"""

import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from datasets import load_dataset
from typing import Dict, List, Tuple, Optional
import logging
import random
import os
from pathlib import Path

# Import token-constrained dataset
try:
    from .token_constrained_dataset import create_token_constrained_data_module
    TOKEN_CONSTRAINED_AVAILABLE = True
except ImportError:
    try:
        # Try importing without relative import
        from token_constrained_dataset import create_token_constrained_data_module
        TOKEN_CONSTRAINED_AVAILABLE = True
    except ImportError:
        TOKEN_CONSTRAINED_AVAILABLE = False

logger = logging.getLogger(__name__)


class CompleteBabyLMDataset(Dataset):
    """Complete BabyLM Multimodal Dataset (CC3M + Localized Narratives)"""

    def __init__(
        self,
        dataset_dir: str,
        tokenizer_name: str = "gpt2",
        max_seq_length: int = 512,
        split: str = "train",
        max_samples: Optional[int] = None
    ):
        self.dataset_dir = Path(dataset_dir)
        self.max_seq_length = max_seq_length
        self.split = split

        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load all data sources
        self._load_all_data()
        
        # Create indices for this split (train uses full dataset, no validation split)
        if split == "train":
            self.indices = list(range(len(self.all_captions)))
        else:
            # For validation, we'll use HuggingFace datasets
            self.indices = []

        # Limit samples if specified
        if max_samples is not None and len(self.indices) > max_samples:
            random.seed(42)
            self.indices = random.sample(self.indices, max_samples)

        logger.info(f"Loaded {len(self.indices)} samples for {split} split")

    def _load_all_data(self):
        """Load all multimodal data sources"""
        logger.info("Loading complete BabyLM multimodal dataset...")

        # Load Conceptual Captions 3M
        cc_captions_file = self.dataset_dir / "cc_3M_captions.json"
        cc_feat1_file = self.dataset_dir / "cc_3M_dino_v2_states_1of2.npy"
        cc_feat2_file = self.dataset_dir / "cc_3M_dino_v2_states_2of2.npy"

        with open(cc_captions_file, 'r', encoding='utf-8') as f:
            cc_captions = json.load(f)

        cc_feat1 = np.load(cc_feat1_file, mmap_mode='r')
        cc_feat2 = np.load(cc_feat2_file, mmap_mode='r')
        cc_features = VisionFeaturesConcatenated(cc_feat1, cc_feat2)

        logger.info(f"Loaded Conceptual Captions: {len(cc_captions)} samples")

        # Load Localized Narratives  
        ln_captions_file = self.dataset_dir / "local_narr_captions.json"
        ln_feat_file = self.dataset_dir / "local_narr_dino_v2_states.npy"

        with open(ln_captions_file, 'r', encoding='utf-8') as f:
            ln_captions = json.load(f)

        ln_features = np.load(ln_feat_file, mmap_mode='r')

        logger.info(f"Loaded Localized Narratives: {len(ln_captions)} samples")

        # Combine all data
        self.all_captions = cc_captions + ln_captions
        self.all_features = CombinedVisionFeatures(cc_features, ln_features)

        logger.info(f"Total multimodal samples: {len(self.all_captions)}")

        # Verify alignment
        if len(self.all_captions) != len(self.all_features):
            raise ValueError(f"Data alignment error: {len(self.all_captions)} captions vs {len(self.all_features)} features")

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single data sample"""
        actual_idx = self.indices[idx]

        # Get caption and vision features
        caption = self.all_captions[actual_idx]
        vision_feature = self.all_features[actual_idx]

        # Tokenize caption
        encoded = self.tokenizer(
            caption,
            max_length=self.max_seq_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        input_ids = encoded['input_ids'].squeeze(0)
        attention_mask = encoded['attention_mask'].squeeze(0)

        # Create labels for text generation (shifted input_ids)
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100  # Ignore padding tokens in loss

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
            'vision_features': torch.tensor(vision_feature.copy(), dtype=torch.float32),
            'caption': caption,
            'index': actual_idx,
            'vision_index': actual_idx,  # Add for compatibility
            'has_vision': True  # Add for compatibility
        }


class HuggingFaceValidationDataset(Dataset):
    """Validation dataset using HuggingFace datasets"""

    def __init__(
        self,
        dataset_name: str,
        hf_token: str,
        tokenizer_name: str = "gpt2",
        max_seq_length: int = 512,
        max_samples: Optional[int] = None
    ):
        self.dataset_name = dataset_name
        self.max_seq_length = max_seq_length

        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load dataset from HuggingFace
        logger.info(f"Loading validation dataset: {dataset_name}")
        try:
            if dataset_name == "ewok-core/ewok-core-1.0":
                self.dataset = load_dataset(dataset_name, token=hf_token, split="test")
                self.text_field = "text"  # Adjust based on actual schema
            elif dataset_name == "facebook/winoground":
                self.dataset = load_dataset(dataset_name, token=hf_token, split="test")
                self.text_field = "caption_0"  # Winoground has multiple captions
            elif dataset_name == "squad":
                self.dataset = load_dataset(dataset_name, split="validation")  # No token needed
                self.text_field = "question"  # Use questions for text generation
            elif dataset_name == "glue/sst2":
                self.dataset = load_dataset("glue", "sst2", split="validation")  # No token needed
                self.text_field = "sentence"  # Use sentences for text generation
            else:
                # Try to load as a generic public dataset
                self.dataset = load_dataset(dataset_name, split="validation")
                # Try to guess the text field
                sample = self.dataset[0]
                text_fields = ["text", "sentence", "question", "input", "content"]
                self.text_field = next((field for field in text_fields if field in sample), None)
                if self.text_field is None:
                    raise ValueError(f"Could not find text field in dataset: {dataset_name}")

            if max_samples is not None:
                self.dataset = self.dataset.select(range(min(max_samples, len(self.dataset))))

            logger.info(f"Loaded {len(self.dataset)} validation samples")

        except Exception as e:
            logger.error(f"Failed to load {dataset_name}: {e}")
            # Create dummy dataset for testing
            self.dataset = [{"text": f"Validation sample {i}"} for i in range(100)]
            self.text_field = "text"
            logger.warning(f"Using dummy validation data with {len(self.dataset)} samples")

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single validation sample"""
        item = self.dataset[idx]
        
        # Extract text based on dataset structure
        if isinstance(item, dict) and self.text_field in item:
            text = item[self.text_field]
        else:
            text = str(item)

        # Tokenize text
        encoded = self.tokenizer(
            text,
            max_length=self.max_seq_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        input_ids = encoded['input_ids'].squeeze(0)
        attention_mask = encoded['attention_mask'].squeeze(0)
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
            'vision_features': torch.zeros(768, dtype=torch.float32),  # Dummy vision features
            'caption': text,
            'index': idx,
            'vision_index': idx,  # Add for compatibility
            'has_vision': False  # Add for compatibility - text-only validation
        }


class VisionFeaturesConcatenated:
    """Memory-efficient concatenation of two vision feature arrays"""
    
    def __init__(self, features_1, features_2):
        self.features_1 = features_1
        self.features_2 = features_2
        self.len_1 = len(features_1)
        self.len_2 = len(features_2)
        self.total_len = self.len_1 + self.len_2
        
    def __len__(self):
        return self.total_len
        
    def __getitem__(self, idx):
        if idx < self.len_1:
            return self.features_1[idx]
        else:
            return self.features_2[idx - self.len_1]
            
    @property
    def shape(self):
        return (self.total_len, self.features_1.shape[1])


class CombinedVisionFeatures:
    """Combine Conceptual Captions and Localized Narratives features"""
    
    def __init__(self, cc_features, ln_features):
        self.cc_features = cc_features
        self.ln_features = ln_features
        self.cc_len = len(cc_features)
        self.ln_len = len(ln_features)
        self.total_len = self.cc_len + self.ln_len
        
    def __len__(self):
        return self.total_len
        
    def __getitem__(self, idx):
        if idx < self.cc_len:
            return self.cc_features[idx]
        else:
            return self.ln_features[idx - self.cc_len]
            
    @property
    def shape(self):
        return (self.total_len, 768)  # DiNOv2 features are 768D


class BabyLMDataModule:
    """Data module for BitMar training with complete BabyLM dataset"""

    def __init__(self, config: Dict):
        self.config = config
        self.tokenizer_name = config.get('text_encoder_name', 'gpt2')

        # Dataset parameters
        self.dataset_dir = config['dataset_dir']
        self.max_seq_length = config['max_seq_length']
        self.hf_token = config.get('hf_token') or os.getenv('HF_TOKEN', '')

        # DataLoader parameters
        self.batch_size = config['batch_size']
        self.num_workers = config['num_workers']
        self.pin_memory = config['pin_memory']
        self.persistent_workers = config.get('persistent_workers', True)

        # Validation datasets
        self.validation_datasets = config.get('validation_datasets', [
            'ewok-core/ewok-core-1.0',
            'facebook/winoground'
        ])

        # Datasets
        self.train_dataset = None
        self.val_datasets = {}

    def setup(self, max_samples: Optional[int] = None):
        """Setup train and validation datasets"""
        logger.info("Setting up complete BabyLM dataset...")

        # Create training dataset (uses complete BabyLM data)
        self.train_dataset = CompleteBabyLMDataset(
            dataset_dir=self.dataset_dir,
            tokenizer_name=self.tokenizer_name,
            max_seq_length=self.max_seq_length,
            split="train",
            max_samples=max_samples
        )

        # Create validation datasets from HuggingFace
        for dataset_name in self.validation_datasets:
            try:
                logger.info(f"Setting up validation dataset: {dataset_name}")
                self.val_datasets[dataset_name] = HuggingFaceValidationDataset(
                    dataset_name=dataset_name,
                    hf_token=self.hf_token,
                    tokenizer_name=self.tokenizer_name,
                    max_seq_length=self.max_seq_length,
                    max_samples=100 if max_samples else 500  # Smaller validation sets
                )
            except Exception as e:
                logger.warning(f"Failed to load {dataset_name}: {e}")

        logger.info(f"Train dataset: {len(self.train_dataset)} samples")
        logger.info(f"Validation datasets: {list(self.val_datasets.keys())}")

    def train_dataloader(self) -> DataLoader:
        """Create training dataloader"""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers if self.num_workers > 0 else False,
            drop_last=True
        )

    def val_dataloader(self) -> List[DataLoader]:
        """Create validation dataloaders"""
        val_loaders = []
        for name, dataset in self.val_datasets.items():
            loader = DataLoader(
                dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=0,  # Use 0 for validation to avoid memory issues
                pin_memory=self.pin_memory,
                drop_last=False
            )
            val_loaders.append(loader)
        
        return val_loaders if val_loaders else [self._create_dummy_val_loader()]

    def _create_dummy_val_loader(self):
        """Create dummy validation loader if HF datasets fail"""
        logger.warning("Creating dummy validation dataset")
        dummy_dataset = HuggingFaceValidationDataset(
            dataset_name="dummy",
            hf_token="",
            tokenizer_name=self.tokenizer_name,
            max_seq_length=self.max_seq_length,
            max_samples=50
        )
        return DataLoader(dummy_dataset, batch_size=self.batch_size, shuffle=False)

    def get_sample_batch(self, split: str = "train", num_samples: int = 4) -> Dict[str, torch.Tensor]:
        """Get a sample batch for testing"""
        if split == "train":
            dataset = self.train_dataset
        else:
            dataset = list(self.val_datasets.values())[0] if self.val_datasets else None

        if dataset is None:
            raise ValueError("Dataset not setup. Call setup() first.")

        # Get random samples
        indices = random.sample(range(len(dataset)), min(num_samples, len(dataset)))
        samples = [dataset[i] for i in indices]

        # Collate samples
        batch = {}
        for key in samples[0].keys():
            if key in ['caption']:
                batch[key] = [sample[key] for sample in samples]
            else:
                batch[key] = torch.stack([sample[key] for sample in samples])

        return batch


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """Custom collate function for batching"""
    keys = batch[0].keys()
    collated = {}

    for key in keys:
        if key == 'caption':
            collated[key] = [item[key] for item in batch]
        else:
            collated[key] = torch.stack([item[key] for item in batch])

    return collated


def create_data_module(config: Dict) -> 'BabyLMDataModule':
    """Create appropriate data module based on configuration"""
    
    # Check if token constraints are specified
    if config.get('token_constraints') and TOKEN_CONSTRAINED_AVAILABLE:
        logger.info("🎯 Creating token-constrained data module for 100M tokens")
        return create_token_constrained_data_module(config)
    else:
        logger.info("📊 Creating standard BabyLM data module")
        return BabyLMDataModule(config)


def test_dataset(config: Dict, max_samples: int = 10):
    """Test dataset loading and processing"""
    logger.info("Testing complete BabyLM dataset...")

    # Create data module
    data_module = create_data_module(config)
    data_module.setup(max_samples=max_samples)

    # Test sample
    sample = data_module.train_dataset[0]
    logger.info(f"Sample keys: {sample.keys()}")
    logger.info(f"Input IDs shape: {sample['input_ids'].shape}")
    logger.info(f"Vision features shape: {sample['vision_features'].shape}")
    logger.info(f"Caption: {sample['caption'][:100]}...")

    # Test batch
    batch = data_module.get_sample_batch(num_samples=4)
    logger.info(f"Batch input IDs shape: {batch['input_ids'].shape}")
    logger.info(f"Batch vision features shape: {batch['vision_features'].shape}")
    logger.info(f"Number of captions in batch: {len(batch['caption'])}")

    logger.info("Dataset test completed successfully!")

    return data_module


if __name__ == "__main__":
    # Test configuration
    test_config = {
        'dataset_dir': "../babylm_dataset",
        'text_encoder_name': "gpt2",
        'max_seq_length': 512,
        'batch_size': 4,
        'num_workers': 0,
        'pin_memory': False,
        'hf_token': os.getenv('HF_TOKEN', 'your_hf_token_here'),
        'validation_datasets': ['ewok-core/ewok-core-1.0', 'facebook/winoground']
    }

    # Test dataset
    test_dataset(test_config)
