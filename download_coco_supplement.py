#!/usr/bin/env python3
"""
COCO Dataset Download Script for Tiny-BitGen
Downloads COCO dataset from Kaggle to replace BabyLM data
COCO already has image-caption pairs - no alignment needed!
Just extracts and organizes the existing annotations.
"""

import os
import json
import subprocess
import logging
from pathlib import Path
from tqdm import tqdm
import requests
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
import pickle
import shutil

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def check_existing_data(data_dir: str = "./data") -> Dict:
    """Check what data already exists"""
    data_path = Path(data_dir)
    
    status = {
        'coco_exists': False,
        'coco_count': 0,
        'features_cache_exists': False,
        'existing_features_count': 0
    }
    
    # Check COCO
    coco_dir = data_path / "coco"
    if coco_dir.exists():
        coco_images = []
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG', '*.webp', '*.WEBP']
        for ext in image_extensions:
            coco_images.extend(coco_dir.glob(f"**/{ext}"))
        status['coco_exists'] = len(coco_images) > 0
        status['coco_count'] = len(coco_images)
    
    # Check existing features cache
    cache_dir = data_path / "vision_features_cache"
    if cache_dir.exists():
        features_file = cache_dir / "all_features.npy"
        if features_file.exists():
            try:
                features = np.load(features_file)
                status['features_cache_exists'] = True
                status['existing_features_count'] = len(features)
            except:
                pass
    
    return status


def install_kagglehub():
    """Install kagglehub if not available"""
    try:
        import kagglehub
        logger.info("✅ kagglehub already installed")
        return True
    except ImportError:
        logger.info("📦 Installing kagglehub...")
        try:
            subprocess.run(["pip", "install", "kagglehub"], check=True)
            logger.info("✅ kagglehub installed successfully")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"❌ Failed to install kagglehub: {e}")
            return False


def download_coco_dataset(data_dir: str = "./data") -> Dict:
    """Download COCO dataset from Kaggle"""
    
    if not install_kagglehub():
        return None
    
    try:
        import kagglehub
    except ImportError:
        logger.error("❌ kagglehub not available after installation")
        return None
    
    data_path = Path(data_dir)
    coco_dir = data_path / "coco"
    coco_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("🚀 Downloading COCO dataset from Kaggle...")
    
    try:
        # Use the correct kagglehub API - dataset_download
        logger.info("📦 Downloading COCO image-caption dataset...")
        downloaded_path = kagglehub.dataset_download("nikhil7280/coco-image-caption")
        logger.info(f"✅ Dataset downloaded to: {downloaded_path}")
        
        # Copy to our data directory structure
        source_path = Path(downloaded_path)
        
        # Copy all images to a single directory for unified training
        all_images_dest = coco_dir / "images"
        all_images_dest.mkdir(exist_ok=True)
        
        # Copy train2014 images
        train_source = source_path / "train2014" / "train2014"
        if train_source.exists():
            logger.info("📂 Copying train2014 images to unified directory...")
            for img_file in train_source.glob("*.jpg"):
                dest_path = all_images_dest / img_file.name
                if not dest_path.exists():
                    shutil.copy2(img_file, dest_path)
            logger.info("✅ train2014 images copied")
        
        # Copy val2017 images to the same unified directory
        val_source = source_path / "val2017" / "val2017"
        if val_source.exists():
            logger.info("📂 Copying val2017 images to unified directory...")
            for img_file in val_source.glob("*.jpg"):
                dest_path = all_images_dest / img_file.name
                if not dest_path.exists():
                    shutil.copy2(img_file, dest_path)
            logger.info("✅ val2017 images copied")
        
        # Copy annotations
        ann_source = source_path / "annotations_trainval2014" / "annotations"
        ann_dest = coco_dir / "annotations_trainval2014"
        
        if ann_source.exists() and not ann_dest.exists():
            logger.info("📂 Copying annotations_trainval2014...")
            shutil.copytree(ann_source, ann_dest)
            logger.info("✅ annotations_trainval2014 copied")
        elif ann_dest.exists():
            logger.info("✅ annotations_trainval2014 already exist")
        
        # Copy val2017 annotations
        ann_source_2017 = source_path / "annotations_trainval2017" / "annotations"
        ann_dest_2017 = coco_dir / "annotations_trainval2017"
        
        if ann_source_2017.exists() and not ann_dest_2017.exists():
            logger.info("📂 Copying annotations_trainval2017...")
            shutil.copytree(ann_source_2017, ann_dest_2017)
            logger.info("✅ annotations_trainval2017 copied")
        elif ann_dest_2017.exists():
            logger.info("✅ annotations_trainval2017 already exist")
        
        return {
            'downloaded_path': downloaded_path,
            'coco_dir': str(coco_dir),
            'success': True
        }
        
    except Exception as e:
        logger.error(f"❌ Failed to download COCO dataset: {e}")
        # Try alternative API if available
        try:
            logger.info("🔄 Trying alternative kagglehub API...")
            downloaded_path = kagglehub.download("nikhil7280/coco-image-caption")
            logger.info(f"✅ Dataset downloaded with alternative API to: {downloaded_path}")
            
            # Copy files with alternative path structure
            source_path = Path(downloaded_path)
            
            # Copy all images to unified directory
            all_images_dest = coco_dir / "images"
            all_images_dest.mkdir(exist_ok=True)
            
            # Copy train2014 images
            train_source = source_path / "train2014" / "train2014"
            if train_source.exists():
                logger.info("📂 Copying train2014 images to unified directory...")
                for img_file in train_source.glob("*.jpg"):
                    dest_path = all_images_dest / img_file.name
                    if not dest_path.exists():
                        shutil.copy2(img_file, dest_path)
                logger.info("✅ train2014 images copied")
            
            # Copy val2017 images to the same unified directory
            val_source = source_path / "val2017" / "val2017"
            if val_source.exists():
                logger.info("📂 Copying val2017 images to unified directory...")
                for img_file in val_source.glob("*.jpg"):
                    dest_path = all_images_dest / img_file.name
                    if not dest_path.exists():
                        shutil.copy2(img_file, dest_path)
                logger.info("✅ val2017 images copied")
            
            # Copy annotations
            ann_source = source_path / "annotations_trainval2014" / "annotations"
            ann_dest = coco_dir / "annotations_trainval2014"
            
            if ann_source.exists() and not ann_dest.exists():
                logger.info("📂 Copying annotations_trainval2014...")
                shutil.copytree(ann_source, ann_dest)
                logger.info("✅ annotations_trainval2014 copied")
            
            # Copy val2017 annotations
            ann_source_2017 = source_path / "annotations_trainval2017" / "annotations"
            ann_dest_2017 = coco_dir / "annotations_trainval2017"
            
            if ann_source_2017.exists() and not ann_dest_2017.exists():
                logger.info("📂 Copying annotations_trainval2017...")
                shutil.copytree(ann_source_2017, ann_dest_2017)
                logger.info("✅ annotations_trainval2017 copied")
            
            return {
                'downloaded_path': downloaded_path,
                'coco_dir': str(coco_dir),
                'success': True
            }
        except Exception as e2:
            logger.error(f"❌ Alternative API also failed: {e2}")
            return None


def process_coco_captions(data_dir: str = "./data") -> List[Dict]:
    """Process COCO captions and create training data pairs"""
    
    data_path = Path(data_dir)
    coco_dir = data_path / "coco"
    
    if not coco_dir.exists():
        logger.error("❌ COCO directory not found")
        return []
    
    logger.info("📝 Processing COCO captions...")
    
    # Use unified images directory
    images_dir = coco_dir / "images"
    if not images_dir.exists():
        # Fallback to separate directories if unified doesn't exist
        train_images_dir = coco_dir / "train2014"
        val_images_dir = coco_dir / "val2017"
    else:
        train_images_dir = images_dir
        val_images_dir = images_dir
    
    # Process train2014 captions
    trainval_captions_dir = coco_dir / "annotations_trainval2014"
    trainval_captions_filepath = trainval_captions_dir / "captions_train2014.json"
    
    image_caption_pairs = []
    
    if trainval_captions_filepath.exists():
        logger.info("📖 Processing train2014 captions...")
        
        with open(trainval_captions_filepath, 'r') as f:
            trainval_data = json.load(f)
        
        # Create DataFrame from annotations
        trainval_captions_df = pd.json_normalize(trainval_data, "annotations")
        
        # Check images in unified directory first, then fallback to train2014 directory
        for _, row in trainval_captions_df.iterrows():
            image_id = row["image_id"]
            image_filename = f'COCO_train2014_{image_id:012d}.jpg'
            
            # Try unified directory first
            image_path = images_dir / image_filename if images_dir.exists() else train_images_dir / image_filename
            
            if image_path.exists():
                caption = row["caption"]
                processed_caption = caption.lower().strip()
                
                image_caption_pairs.append({
                    'image_id': f'coco_train_{image_id}',
                    'image_path': str(image_path),
                    'caption': processed_caption,
                    'original_caption': caption,
                    'dataset': 'coco_train2014'
                })
        
        logger.info(f"   Found {len([p for p in image_caption_pairs if p['dataset'] == 'coco_train2014']):,} valid train2014 image-caption pairs")
    
    # Process val2017 captions
    test_captions_dir = coco_dir / "annotations_trainval2017"
    test_captions_filepath = test_captions_dir / "captions_val2017.json"
    
    if test_captions_filepath.exists():
        logger.info("📖 Processing val2017 captions...")
        
        with open(test_captions_filepath, 'r') as f:
            test_data = json.load(f)
        
        # Create DataFrame from annotations
        test_captions_df = pd.json_normalize(test_data, "annotations")
        
        for _, row in test_captions_df.iterrows():
            image_id = row["image_id"]
            image_filename = f'{image_id:012d}.jpg'
            
            # Try unified directory first
            image_path = images_dir / image_filename if images_dir.exists() else val_images_dir / image_filename
            
            if image_path.exists():
                caption = row["caption"]
                processed_caption = caption.lower().strip()
                
                image_caption_pairs.append({
                    'image_id': f'coco_val_{image_id}',
                    'image_path': str(image_path),
                    'caption': processed_caption,
                    'original_caption': caption,
                    'dataset': 'coco_val2017'
                })
        
        logger.info(f"   Found {len([p for p in image_caption_pairs if p['dataset'] == 'coco_val2017']):,} valid val2017 image-caption pairs")
    
    logger.info(f"✅ Total COCO image-caption pairs: {len(image_caption_pairs):,}")
    return image_caption_pairs


def validate_coco_images(image_caption_pairs: List[Dict]) -> List[Dict]:
    """Validate that COCO images exist and are readable"""
    
    if not image_caption_pairs:
        logger.warning("⚠️ No image-caption pairs to validate")
        return []
    
    logger.info(f"✅ Validating {len(image_caption_pairs):,} COCO images...")
    
    valid_pairs = []
    
    for pair in tqdm(image_caption_pairs, desc="Validating images"):
        try:
            # Just check if file exists and is a valid image
            from PIL import Image
            img_path = Path(pair['image_path'])
            
            if img_path.exists():
                # Quick validation - try to open image
                with Image.open(img_path) as img:
                    # Convert to RGB to ensure compatibility
                    img = img.convert('RGB')
                    # Add image dimensions to metadata
                    pair['image_width'] = img.width
                    pair['image_height'] = img.height
                    
                valid_pairs.append(pair)
            else:
                logger.debug(f"Image not found: {img_path}")
                
        except Exception as e:
            logger.debug(f"Invalid image {pair['image_path']}: {e}")
            continue
    
    logger.info(f"   ✅ Validated {len(valid_pairs):,} valid COCO images")
    return valid_pairs


def save_coco_data(coco_pairs: List[Dict], data_dir: str = "./data") -> Dict:
    """Save COCO data for training"""
    
    data_path = Path(data_dir)
    data_path.mkdir(exist_ok=True)
    
    logger.info("💾 Saving COCO data for training...")
    
    if not coco_pairs:
        logger.error("❌ No pairs to save")
        return None
    
    # Save image-caption pairs
    pairs_file = data_path / "coco_image_caption_pairs.json"
    with open(pairs_file, 'w') as f:
        json.dump(coco_pairs, f, indent=2)
    
    # Also save with the name expected by the dataset loader
    aligned_pairs_file = data_path / "aligned_pairs.json"
    with open(aligned_pairs_file, 'w') as f:
        json.dump(coco_pairs, f, indent=2)
    
    # Update metadata to indicate on-the-fly feature processing
    cache_dir = data_path / "vision_features_cache"
    cache_dir.mkdir(exist_ok=True)
    
    metadata = {
        'num_samples': len(coco_pairs),
        'feature_type': 'on_the_fly_dinov2',
        'vision_model': 'facebook/dinov2-base',
        'use_dummy_vision': False,
        'extract_vision_features': True,
        'on_the_fly_processing': True,
        'datasets': ['coco'],
        'coco_count': len(coco_pairs),
        'created_by': 'download_coco_supplement.py',
        'note': 'Features extracted on-the-fly during training - no pre-alignment needed'
    }
    
    metadata_file = cache_dir / "cache_metadata.pkl"
    with open(metadata_file, 'wb') as f:
        pickle.dump(metadata, f)
    
    # Save summary
    summary = {
        'total_samples': len(coco_pairs),
        'coco_samples': len(coco_pairs),
        'datasets_included': ['coco'],
        'feature_extraction': 'on_the_fly',
        'ready_for_training': True
    }
    
    summary_file = data_path / "coco_dataset_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info("✅ COCO data saved successfully!")
    logger.info(f"   📊 Total samples: {len(coco_pairs):,}")
    logger.info(f"   🖼️ COCO: {len(coco_pairs):,}")
    logger.info(f"   ⚡ Features will be extracted on-the-fly during training")
    logger.info(f"   💾 Ready for training")
    
    return summary


def main():
    """Main function to download and process COCO dataset"""
    
    logger.info("🚀 COCO Dataset Download for Tiny-BitGen")
    logger.info("=" * 50)
    
    # Check existing data
    logger.info("🔍 Checking existing data...")
    status = check_existing_data()
    
    logger.info(f"   COCO: {status['coco_count']:,} images" if status['coco_exists'] else "   COCO: Not found")
    logger.info(f"   Features Cache: {status['existing_features_count']:,} features" if status['features_cache_exists'] else "   Features Cache: Not found")
    
    # Download COCO if not exists
    if not status['coco_exists']:
        logger.info("\n📦 Downloading COCO dataset...")
        download_result = download_coco_dataset()
        
        if not download_result or not download_result['success']:
            logger.error("❌ Failed to download COCO dataset")
            return
        
        logger.info("✅ COCO dataset downloaded successfully")
    else:
        logger.info("\n✅ COCO dataset already exists")
    
    # Process COCO captions
    logger.info("\n📝 Processing COCO captions...")
    coco_pairs = process_coco_captions()
    
    if not coco_pairs:
        logger.error("❌ No COCO image-caption pairs found")
        return
    
    # Validate COCO images
    logger.info("\n✅ Validating COCO images...")
    valid_coco_pairs = validate_coco_images(coco_pairs)
    
    if not valid_coco_pairs:
        logger.error("❌ No valid COCO images found")
        return
    
    # Save COCO data
    logger.info("\n💾 Saving COCO data...")
    save_result = save_coco_data(valid_coco_pairs)
    
    if save_result:
        logger.info("\n🎉 SUCCESS!")
        logger.info(f"   📊 Total dataset size: {save_result['total_samples']:,} samples")
        logger.info(f"   🖼️ COCO contribution: {save_result['coco_samples']:,}")
        logger.info(f"   ⚡ Feature extraction: {save_result['feature_extraction']}")
        logger.info(f"   💾 All data saved to: ./data/")
        logger.info(f"\n🚀 Ready to train with: python train_coco.py --config configs/bitmar_coco.yaml")
    else:
        logger.error("❌ Saving failed")


if __name__ == "__main__":
    main()