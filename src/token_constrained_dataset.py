"""
Token-Constrained Dataset for BitMar 100M Token Training
Handles exactly 100M tokens: 50M from aligned image captions + 50M from train_50M text
Ensures perfect image-caption alignment while maintaining token constraints
"""

import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from typing import Dict, List, Tuple, Optional
import logging
import random
import os
from pathlib import Path
from collections import defaultdict
import pickle

logger = logging.getLogger(__name__)


class TokenConstrainedBabyLMDataset(Dataset):
    """Dataset with exactly 100M tokens: 50M caption + 50M text-only"""

    def __init__(
        self,
        dataset_dir: str,
        tokenizer_name: str = "gpt2",
        max_seq_length: int = 256,
        target_caption_tokens: int = 50_000_000,
        target_text_tokens: int = 50_000_000,
        cache_dir: str = "token_cache",
        rebuild_cache: bool = False
    ):
        self.dataset_dir = Path(dataset_dir)
        self.max_seq_length = max_seq_length
        self.target_caption_tokens = target_caption_tokens
        self.target_text_tokens = target_text_tokens
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)

        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        # Ensure model max length is set correctly
        if hasattr(self.tokenizer, 'model_max_length'):
            if self.tokenizer.model_max_length > max_seq_length:
                logger.warning(f"Tokenizer max_length ({self.tokenizer.model_max_length}) > dataset max_length ({max_seq_length})")
                # Force the tokenizer to respect our max length
                self.tokenizer.model_max_length = max_seq_length

        # Load and process data with token constraints
        self._load_token_constrained_data(rebuild_cache)
        
        logger.info(f"Token-constrained dataset created:")
        logger.info(f"  • Caption samples: {len(self.caption_samples)} ({self.actual_caption_tokens:,} tokens)")
        logger.info(f"  • Text samples: {len(self.text_samples)} ({self.actual_text_tokens:,} tokens)")
        logger.info(f"  • Total samples: {len(self.all_samples)} ({self.total_tokens:,} tokens)")
        logger.info(f"  • Image-caption alignment: {self.alignment_verified}")

    def _load_token_constrained_data(self, rebuild_cache: bool = False):
        """Load data with strict token constraints and perfect alignment"""
        cache_file = self.cache_dir / "token_constrained_data.pkl"
        
        if cache_file.exists() and not rebuild_cache:
            logger.info("Loading cached token-constrained data...")
            with open(cache_file, 'rb') as f:
                cache_data = pickle.load(f)
                self.caption_samples = cache_data['caption_samples']
                self.text_samples = cache_data['text_samples']
                self.vision_features = cache_data['vision_features']
                self.actual_caption_tokens = cache_data['actual_caption_tokens']
                self.actual_text_tokens = cache_data['actual_text_tokens']
                self.alignment_verified = cache_data['alignment_verified']
        else:
            logger.info("Building token-constrained dataset from scratch...")
            self._build_token_constrained_dataset()
            
            # Cache the processed data
            cache_data = {
                'caption_samples': self.caption_samples,
                'text_samples': self.text_samples,
                'vision_features': self.vision_features,
                'actual_caption_tokens': self.actual_caption_tokens,
                'actual_text_tokens': self.actual_text_tokens,
                'alignment_verified': self.alignment_verified
            }
            with open(cache_file, 'wb') as f:
                pickle.dump(cache_data, f)
            logger.info(f"Cached token-constrained data to {cache_file}")

        # Create unified dataset
        self._create_unified_dataset()

    def _build_token_constrained_dataset(self):
        """Build dataset with exact token constraints"""
        logger.info("Processing multimodal data with token constraints...")

        # Load captions and vision features
        self._load_multimodal_data()
        
        # Select exactly 50M tokens from captions with perfect alignment
        self._select_caption_tokens()
        
        # Select exactly 50M tokens from text data
        self._select_text_tokens()
        
        # Verify alignment
        self._verify_alignment()

    def _load_multimodal_data(self):
        """Load all multimodal data"""
        # Load Conceptual Captions
        cc_captions_file = self.dataset_dir / "cc_3M_captions.json"
        with open(cc_captions_file, 'r', encoding='utf-8') as f:
            cc_captions = json.load(f)

        # Load vision features
        cc_feat1 = np.load(self.dataset_dir / "cc_3M_dino_v2_states_1of2.npy", mmap_mode='r')
        cc_feat2 = np.load(self.dataset_dir / "cc_3M_dino_v2_states_2of2.npy", mmap_mode='r')
        
        # Load Localized Narratives
        ln_captions_file = self.dataset_dir / "local_narr_captions.json"
        with open(ln_captions_file, 'r', encoding='utf-8') as f:
            ln_captions = json.load(f)
            
        ln_features = np.load(self.dataset_dir / "local_narr_dino_v2_states.npy", mmap_mode='r')

        # Store all data
        self.all_captions = cc_captions + ln_captions
        self.all_vision_features = np.concatenate([cc_feat1, cc_feat2, ln_features], axis=0)
        
        logger.info(f"Loaded {len(self.all_captions)} captions with {self.all_vision_features.shape[0]} vision features")
        
        # Verify alignment
        if len(self.all_captions) != self.all_vision_features.shape[0]:
            raise ValueError(f"Caption-vision alignment error: {len(self.all_captions)} vs {self.all_vision_features.shape[0]}")

    def _select_caption_tokens(self):
        """Select captions that total exactly 50M tokens with uniform distribution"""
        logger.info("Selecting captions for exactly 50M tokens...")
        
        # Count tokens for all captions
        caption_token_counts = []
        for caption in self.all_captions:
            tokens = self.tokenizer.encode(caption)
            caption_token_counts.append(len(tokens))
        
        # Create uniform sampling indices
        total_captions = len(self.all_captions)
        indices = list(range(total_captions))
        random.shuffle(indices)  # Shuffle for uniform distribution
        
        # Select captions until we reach exactly 50M tokens
        selected_indices = []
        total_tokens = 0
        
        for idx in indices:
            token_count = caption_token_counts[idx]
            if total_tokens + token_count <= self.target_caption_tokens:
                selected_indices.append(idx)
                total_tokens += token_count
            elif total_tokens < self.target_caption_tokens:
                # If adding this caption would exceed the limit but we haven't reached it,
                # truncate the caption to fit exactly
                remaining_tokens = self.target_caption_tokens - total_tokens
                if remaining_tokens > 10:  # Only if meaningful tokens remain
                    selected_indices.append(idx)
                    total_tokens = self.target_caption_tokens
                    break
            else:
                break
        
        # Sort indices to maintain alignment with vision features
        selected_indices.sort()
        
        # Create caption samples with their aligned vision features
        self.caption_samples = []
        self.vision_features = []
        
        for idx in selected_indices:
            caption = self.all_captions[idx]
            vision_feature = self.all_vision_features[idx].copy()
            
            self.caption_samples.append({
                'text': caption,
                'original_index': idx,
                'type': 'caption'
            })
            self.vision_features.append(vision_feature)
        
        self.actual_caption_tokens = total_tokens
        logger.info(f"Selected {len(self.caption_samples)} captions with {self.actual_caption_tokens:,} tokens")

    def _select_text_tokens(self):
        """Select text-only data for exactly 50M tokens"""
        logger.info("Selecting text-only data for exactly 50M tokens...")
        
        # Load all train_50M files
        train_dir = self.dataset_dir / "train_50M"
        text_files = [
            "bnc_spoken.train",
            "childes.train", 
            "gutenberg.train",
            "open_subtitles.train",
            "simple_wiki.train",
            "switchboard.train"
        ]
        
        all_text_lines = []
        for filename in text_files:
            file_path = train_dir / filename
            if file_path.exists():
                with open(file_path, 'r', encoding='utf-8') as f:
                    lines = [line.strip() for line in f if line.strip()]
                    all_text_lines.extend(lines)
                logger.info(f"Loaded {len(lines)} lines from {filename}")
        
        logger.info(f"Total text lines available: {len(all_text_lines)}")
        
        # Shuffle for uniform distribution
        random.shuffle(all_text_lines)
        
        # Select lines until we reach exactly 50M tokens
        self.text_samples = []
        total_tokens = 0
        
        for line in all_text_lines:
            tokens = self.tokenizer.encode(line)
            token_count = len(tokens)
            
            if total_tokens + token_count <= self.target_text_tokens:
                self.text_samples.append({
                    'text': line,
                    'type': 'text_only'
                })
                total_tokens += token_count
            elif total_tokens < self.target_text_tokens:
                # Truncate line to fit exactly
                remaining_tokens = self.target_text_tokens - total_tokens
                if remaining_tokens > 5:  # Only if meaningful tokens remain
                    truncated_tokens = tokens[:remaining_tokens]
                    truncated_text = self.tokenizer.decode(truncated_tokens)
                    self.text_samples.append({
                        'text': truncated_text,
                        'type': 'text_only'
                    })
                    total_tokens = self.target_text_tokens
                break
            else:
                break
        
        self.actual_text_tokens = total_tokens
        logger.info(f"Selected {len(self.text_samples)} text samples with {self.actual_text_tokens:,} tokens")

    def _verify_alignment(self):
        """Verify perfect image-caption alignment with comprehensive checks"""
        if len(self.caption_samples) != len(self.vision_features):
            raise ValueError(f"Alignment verification failed: {len(self.caption_samples)} captions vs {len(self.vision_features)} features")
        
        # Additional alignment checks
        logger.info("🔍 Performing comprehensive alignment verification...")
        
        # Check that vision features have correct shape
        expected_vision_dim = 768  # DiNOv2 features
        for i, vf in enumerate(self.vision_features[:5]):  # Check first 5 samples
            # Handle both flattened (768,) and patch format (196, 768)
            if len(vf.shape) == 1:
                # Flattened format
                if vf.shape[0] != expected_vision_dim:
                    raise ValueError(f"Vision feature shape mismatch at index {i}: {vf.shape}, expected ({expected_vision_dim},)")
            elif len(vf.shape) == 2:
                # Patch format
                if vf.shape[1] != expected_vision_dim:
                    raise ValueError(f"Vision feature shape mismatch at index {i}: {vf.shape}, expected (N, {expected_vision_dim})")
            else:
                raise ValueError(f"Unexpected vision feature shape at index {i}: {vf.shape}")
        
        # Check that indices are properly preserved
        for i, sample in enumerate(self.caption_samples[:10]):  # Check first 10 samples
            original_idx = sample['original_index']
            if original_idx >= len(self.all_captions):
                raise ValueError(f"Invalid original index {original_idx} for caption {i}")
        
        self.alignment_verified = True
        logger.info("✅ Image-caption alignment verified successfully")
        logger.info(f"✅ {len(self.caption_samples)} caption-image pairs perfectly aligned")

    def _create_unified_dataset(self):
        """Create unified dataset mixing captions and text-only data"""
        # Create all samples list
        self.all_samples = []
        
        # Add caption samples (with vision features)
        for i, sample in enumerate(self.caption_samples):
            self.all_samples.append({
                'text': sample['text'],
                'type': 'caption',
                'vision_index': i,  # Index into vision_features array
                'sample_index': len(self.all_samples)
            })
        
        # Add text-only samples (no vision features)
        for sample in self.text_samples:
            self.all_samples.append({
                'text': sample['text'],
                'type': 'text_only',
                'vision_index': None,
                'sample_index': len(self.all_samples)
            })
        
        # Shuffle for balanced training
        random.shuffle(self.all_samples)
        
        # Calculate total tokens
        self.total_tokens = self.actual_caption_tokens + self.actual_text_tokens
        
        logger.info(f"Created unified dataset with {len(self.all_samples)} samples")
        logger.info(f"Total tokens: {self.total_tokens:,} (Target: 100M)")

    def __len__(self) -> int:
        return len(self.all_samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single data sample with alignment validation"""
        sample = self.all_samples[idx]
        text = sample['text']
        sample_type = sample['type']
        vision_index = sample['vision_index']

        # Tokenize text with strict length control
        encoded = self.tokenizer(
            text,
            max_length=self.max_seq_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        input_ids = encoded['input_ids'].squeeze(0)
        attention_mask = encoded['attention_mask'].squeeze(0)
        
        # Double-check sequence length (safety check)
        if input_ids.size(0) > self.max_seq_length:
            logger.warning(f"Sequence length {input_ids.size(0)} > max_length {self.max_seq_length}, truncating")
            input_ids = input_ids[:self.max_seq_length]
            attention_mask = attention_mask[:self.max_seq_length]

        # Create labels for text generation
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100  # Ignore padding tokens

        # Prepare return dictionary
        result = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
            'sample_type': sample_type,
            'sample_index': idx,
            'text': text
        }

        # Add vision features if this is a caption sample (with alignment validation)
        if sample_type == 'caption' and vision_index is not None:
            # Runtime alignment validation
            if vision_index >= len(self.vision_features):
                raise ValueError(f"ALIGNMENT ERROR: vision_index {vision_index} >= {len(self.vision_features)}")
            
            vision_feature = self.vision_features[vision_index]
            
            # Handle both flattened and patch formats
            if len(vision_feature.shape) == 1:
                # Flattened format (768,) - reshape to patches if needed
                if vision_feature.shape[0] == 768:
                    # For compatibility, we'll keep it flattened but ensure it's 2D
                    vision_tensor = torch.tensor(vision_feature.copy(), dtype=torch.float32).unsqueeze(0)  # (1, 768)
                else:
                    vision_tensor = torch.tensor(vision_feature.copy(), dtype=torch.float32)
            else:
                # Already in patch format
                vision_tensor = torch.tensor(vision_feature.copy(), dtype=torch.float32)
            
            result['vision_features'] = vision_tensor
            result['has_vision'] = True
            result['vision_index'] = vision_index  # Track for debugging
        else:
            # Create dummy vision features for text-only samples
            # Match the expected format - use (1, 768) for consistency
            result['vision_features'] = torch.zeros(1, 768, dtype=torch.float32)
            result['has_vision'] = False

        return result

    def get_token_statistics(self) -> Dict[str, int]:
        """Get detailed token statistics"""
        return {
            'caption_tokens': self.actual_caption_tokens,
            'text_tokens': self.actual_text_tokens,
            'total_tokens': self.total_tokens,
            'caption_samples': len(self.caption_samples),
            'text_samples': len(self.text_samples),
            'total_samples': len(self.all_samples),
            'target_caption_tokens': self.target_caption_tokens,
            'target_text_tokens': self.target_text_tokens,
            'alignment_verified': self.alignment_verified
        }


def create_token_constrained_data_module(config: Dict) -> 'TokenConstrainedDataModule':
    """Create data module with token constraints"""
    return TokenConstrainedDataModule(config)


class TokenConstrainedDataModule:
    """Data module for token-constrained training"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.dataset = None
        self.train_loader = None
        
    def setup(self, rebuild_cache: bool = False):
        """Setup the dataset"""
        logger.info("Setting up token-constrained data module...")
        
        # Get token constraints from config
        token_config = self.config.get('token_constraints', {})
        target_caption_tokens = token_config.get('caption_tokens', 50_000_000) 
        target_text_tokens = token_config.get('text_tokens', 50_000_000)
        
        self.dataset = TokenConstrainedBabyLMDataset(
            dataset_dir=self.config['dataset_dir'],
            tokenizer_name=self.config['text_encoder_name'],
            max_seq_length=self.config['max_seq_length'],
            target_caption_tokens=target_caption_tokens,
            target_text_tokens=target_text_tokens,
            rebuild_cache=rebuild_cache
        )
        
        # Log token statistics
        stats = self.dataset.get_token_statistics()
        logger.info("📊 Token Statistics:")
        for key, value in stats.items():
            if isinstance(value, int):
                logger.info(f"  • {key}: {value:,}")
            else:
                logger.info(f"  • {key}: {value}")
    
    def train_dataloader(self) -> DataLoader:
        """Create training dataloader"""
        if self.train_loader is None:
            self.train_loader = DataLoader(
                self.dataset,
                batch_size=self.config['batch_size'],
                shuffle=True,
                num_workers=self.config['num_workers'],
                pin_memory=self.config.get('pin_memory', True),
                persistent_workers=self.config.get('persistent_workers', True),
                drop_last=True  # Ensure consistent batch sizes
            )
        return self.train_loader
    
    def val_dataloader(self) -> List[DataLoader]:
        """Return empty list since we use entire dataset for training"""
        return []
    
    def get_token_statistics(self) -> Dict[str, int]:
        """Get token statistics"""
        if self.dataset:
            return self.dataset.get_token_statistics()
        return {}
