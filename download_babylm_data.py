"""
Download BabyLM Multimodal Dataset for BitMar
Downloads the complete BabyLM multimodal dataset following the official structure:
- Text-only data from train_50M.zip 
- Precomputed DiNOv2 visual embeddings + captions from Conceptual Captions 3M
"""

import os
import sys
import requests
import json
import numpy as np
import zipfile
from pathlib import Path
import logging
from tqdm import tqdm
import shutil

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def download_file(url: str, filepath: str, chunk_size: int = 8192) -> bool:
    """Download file with progress bar"""
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()

        total_size = int(response.headers.get('content-length', 0))

        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, 'wb') as f, tqdm(
            desc=filepath.name,
            total=total_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
        ) as pbar:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))

        logger.info(f"✅ Downloaded: {filepath}")
        return True

    except Exception as e:
        logger.error(f"❌ Failed to download {url}: {e}")
        return False


def extract_zip(zip_path: str, extract_to: str) -> bool:
    """Extract ZIP file"""
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        logger.info(f"✅ Extracted {zip_path} to {extract_to}")
        return True
    except Exception as e:
        logger.error(f"❌ Failed to extract {zip_path}: {e}")
        return False


def check_file_exists(filepath: str) -> bool:
    """Check if file exists and is not empty"""
    path = Path(filepath)
    return path.exists() and path.stat().st_size > 0


def verify_existing_data():
    """Verify if BabyLM data already exists in the expected location"""
    logger.info("🔍 Checking for existing BabyLM dataset...")

    # Check parent directory for BabyLM data (FIXED PATH)
    dataset_dir = Path(
        "../babylm_dataset")  # This points to D:\BabyLM\babylm_dataset\

    # ALL required files from OSF dataset (including train_50M.zip)
    required_files = [
        "cc_3M_captions.json",  # Conceptual Captions 3M captions
        "cc_3M_dino_v2_states_1of2.npy",  # DiNOv2 embeddings part 1
        "cc_3M_dino_v2_states_2of2.npy",  # DiNOv2 embeddings part 2
        "local_narr_captions.json",  # Localized Narratives captions
        "local_narr_dino_v2_states.npy",  # Localized Narratives DiNOv2 embeddings
        "train_50M.zip",  # Text-only training data (now required)
    ]

    # Check for extracted train_50M files
    train_50M_files = [
        "train_50M/bnc_spoken.train",
        "train_50M/childes.train",
        "train_50M/gutenberg.train",
        "train_50M/open_subtitles.train",
        "train_50M/simple_wiki.train",
        "train_50M/switchboard.train"
    ]

    # Optional files
    optional_files = [
        "README.pdf",  # Dataset documentation
    ]

    existing_files = []
    missing_files = []

    # Check required files (including train_50M.zip)
    for filename in required_files:
        filepath = dataset_dir / filename
        if check_file_exists(filepath):
            size_mb = filepath.stat().st_size / (1024 * 1024)
            logger.info(f"✅ {filename}: Found ({size_mb:.1f} MB)")
            existing_files.append(filename)
        else:
            logger.warning(f"❌ {filename}: Missing")
            missing_files.append(filename)

    # Check for extracted train_50M files
    train_50M_extracted = True
    train_50M_dir = dataset_dir / "train_50M"

    if train_50M_dir.exists():
        logger.info("📁 Checking extracted train_50M files:")
        extracted_count = 0
        for filename in train_50M_files:
            filepath = dataset_dir / filename
            if check_file_exists(filepath):
                size_mb = filepath.stat().st_size / (1024 * 1024)
                logger.info(f"  ✅ {filename}: Found ({size_mb:.1f} MB)")
                extracted_count += 1
            else:
                logger.warning(f"  ❌ {filename}: Missing")

        if extracted_count == len(train_50M_files):
            logger.info(
                f"✅ All {len(train_50M_files)} train_50M text files found!")
        else:
            logger.warning(
                f"⚠️  Only {extracted_count}/{len(train_50M_files)} train_50M files found")
            train_50M_extracted = False
    else:
        logger.warning("❌ train_50M directory not found - extraction needed")
        train_50M_extracted = False

    # Check optional files
    for filename in optional_files:
        filepath = dataset_dir / filename
        if check_file_exists(filepath):
            size_mb = filepath.stat().st_size / (1024 * 1024)
            logger.info(f"📁 {filename}: Found ({size_mb:.1f} MB)")

    # Check if we have everything we need
    if not missing_files and train_50M_extracted:
        logger.info("🎉 Complete BabyLM multimodal dataset found!")
        logger.info("📊 Dataset summary:")
        logger.info(
            f"  • Required files: {len(existing_files)}/{len(required_files)}")
        logger.info(f"  • Text files: {len(train_50M_files)} train_50M files")
        logger.info(f"  • Vision files: DiNOv2 embeddings + captions")
        return True, dataset_dir
    else:
        if missing_files:
            logger.warning(
                f"⚠️  Missing {len(missing_files)} required files: {missing_files}")
        if not train_50M_extracted:
            logger.warning("⚠️  train_50M.zip needs to be extracted")
        return False, dataset_dir


def download_babylm_multimodal_dataset():
    """Download BabyLM multimodal dataset from OSF"""
    logger.info("🚀 Starting BabyLM Multimodal Dataset Download")
    logger.info("=" * 60)

    # Target directory - external to BitMar project
    dataset_dir = Path("../babylm_dataset")
    dataset_dir.mkdir(exist_ok=True)

    # Official BabyLM multimodal dataset URL
    dataset_url = "https://files.osf.io/v1/resources/ad7qg/providers/osfstorage/6603014bb3a1e301127dfa59/?zip="
    zip_filename = dataset_dir / "babylm_multimodal.zip"

    logger.info(f"\n  Dataset directory: {dataset_dir.absolute()}")
    logger.info(f"📥 Downloading from: {dataset_url}")

    # Expected files after extraction
    expected_files = [
        "cc_3M_captions.json",  # 136.1 MB
        "cc_3M_dino_v2_states_1of2.npy",  # 3.5 GB
        "cc_3M_dino_v2_states_2of2.npy",  # 3.5 GB
        "local_narr_captions.json",  # 139.2 MB
        "local_narr_dino_v2_states.npy",  # 2.4 GB
        "train_50M.zip",  # 90.2 MB
        "README.pdf"  # 63.0 kB
    ]

    logger.info("\n📋 Expected dataset files (~9.8 GB total):")
    for filename in expected_files:
        logger.info(f"   📄 {filename}")

    # Download the ZIP file
    logger.info(f"\n📥 Downloading BabyLM dataset...")
    if download_file(dataset_url, str(zip_filename)):
        logger.info("✅ Download completed successfully!")

        # Extract ZIP file
        logger.info("📦 Extracting dataset...")
        if extract_zip(str(zip_filename), str(dataset_dir)):
            logger.info("✅ Extraction completed successfully!")

            # Clean up ZIP file
            zip_filename.unlink()
            logger.info("🧹 Cleaned up ZIP file")

            # Verify extraction
            logger.info("🔍 Verifying extracted files...")
            verified_count = 0

            for filename in expected_files:
                filepath = dataset_dir / filename
                if check_file_exists(filepath):
                    size_mb = filepath.stat().st_size / (1024 * 1024)
                    logger.info(f"✅ {filename}: Verified ({size_mb:.1f} MB)")
                    verified_count += 1
                else:
                    logger.warning(f"⚠️  {filename}: Missing after extraction")

            # Extract train_50M.zip if it exists
            train_50m_zip = dataset_dir / "train_50M.zip"

            if train_50m_zip.exists():
                logger.info("📦 Extracting train_50M.zip...")
                extract_result = extract_train_50M_if_needed(dataset_dir)

                if extract_result:
                    logger.info("✅ train_50M.zip extracted successfully!")
                else:
                    logger.error("❌ Failed to extract train_50M.zip")
            else:
                logger.warning(
                    "⚠️  train_50M.zip not found - text-only data will be unavailable")

            if verified_count >= 5:  # At least the core multimodal files
                logger.info("🎉 Dataset download and setup completed!")
                logger.info(
                    f"✅ {verified_count}/{len(expected_files)} files verified")

                # Additional verification for core files
                try:
                    verify_core_files(dataset_dir)
                    return True
                except Exception as e:
                    logger.error(f"❌ Core file verification failed: {e}")
                    return False
            else:
                logger.error(
                    f"❌ Only {verified_count}/{len(expected_files)} files verified")
                return False

        else:
            logger.error("❌ Failed to extract dataset")
            return False
    else:
        logger.error("❌ Failed to download dataset")
        return False


def verify_core_files(dataset_dir: Path):
    """Verify core multimodal files"""
    logger.info("🔍 Verifying core dataset files...")

    # Test loading Conceptual Captions
    cc_captions_path = dataset_dir / "cc_3M_captions.json"
    with open(cc_captions_path, 'r') as f:
        cc_captions = json.load(f)
    logger.info(f"📝 Conceptual Captions: {len(cc_captions)} entries")

    # Test loading Localized Narratives
    ln_captions_path = dataset_dir / "local_narr_captions.json"
    with open(ln_captions_path, 'r') as f:
        ln_captions = json.load(f)
    logger.info(f"📝 Localized Narratives: {len(ln_captions)} entries")

    # Test vision features shapes
    cc_feat1 = np.load(
        dataset_dir / "cc_3M_dino_v2_states_1of2.npy", mmap_mode='r')
    cc_feat2 = np.load(
        dataset_dir / "cc_3M_dino_v2_states_2of2.npy", mmap_mode='r')
    ln_feat = np.load(
        dataset_dir / "local_narr_dino_v2_states.npy", mmap_mode='r')

    logger.info(f"🖼️  CC features part 1: {cc_feat1.shape}")
    logger.info(f"🖼️  CC features part 2: {cc_feat2.shape}")
    logger.info(f"🖼️  LN features: {ln_feat.shape}")

    # Verify alignment
    total_cc_features = cc_feat1.shape[0] + cc_feat2.shape[0]
    if total_cc_features == len(cc_captions):
        logger.info("✅ Conceptual Captions alignment verified!")
    else:
        raise ValueError(
            f"CC alignment error: {total_cc_features} features vs {len(cc_captions)} captions")

    if ln_feat.shape[0] == len(ln_captions):
        logger.info("✅ Localized Narratives alignment verified!")
    else:
        raise ValueError(
            f"LN alignment error: {ln_feat.shape[0]} features vs {len(ln_captions)} captions")

    total_samples = total_cc_features + ln_feat.shape[0]
    logger.info(f"🎯 Total multimodal samples: {total_samples:,}")

    return True


def extract_train_50M_if_needed(dataset_dir: Path):
    """Extract train_50M.zip if not already extracted"""
    zip_path = dataset_dir / "train_50M.zip"
    extract_path = dataset_dir / "train_50M"

    if not zip_path.exists():
        logger.warning(
            f"train_50M.zip not found at {zip_path}. Text-only training will be skipped.")
        return None

    if extract_path.exists() and any(extract_path.iterdir()):
        logger.info(f"✅ train_50M already extracted at {extract_path}")
        return extract_path

    logger.info(f"📦 Extracting train_50M.zip to {extract_path}...")
    extract_path.mkdir(exist_ok=True)

    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_path.parent)

        # Verify extraction
        expected_files = [
            "bnc_spoken.train",
            "childes.train",
            "gutenberg.train",
            "open_subtitles.train",
            "simple_wiki.train",
            "switchboard.train"
        ]

        extracted_files = []
        for filename in expected_files:
            file_path = extract_path / filename
            if file_path.exists():
                size_mb = file_path.stat().st_size / (1024 * 1024)
                logger.info(f"  ✅ {filename}: {size_mb:.1f} MB")
                extracted_files.append(filename)
            else:
                logger.warning(f"  ❌ {filename}: Missing after extraction")

        if len(extracted_files) == len(expected_files):
            logger.info(
                f"✅ Successfully extracted train_50M.zip! Found {len(extracted_files)} text files")
        else:
            logger.warning(
                f"⚠️  Only {len(extracted_files)}/{len(expected_files)} files extracted successfully")

        return extract_path

    except Exception as e:
        logger.error(f"❌ Failed to extract train_50M.zip: {e}")
        return None


def create_test_dataset():
    """Create a small test dataset for development"""
    logger.info("🧪 Creating test dataset for development...")

    dataset_dir = Path("../babylm_dataset")
    dataset_dir.mkdir(exist_ok=True)

    # Create test captions (100 samples)
    test_captions = [
        f"A beautiful landscape with mountains and trees in scene {i}"
        for i in range(100)
    ]

    captions_file = dataset_dir / "cc_3M_captions.json"
    with open(captions_file, 'w') as f:
        json.dump(test_captions, f)

    # Create test vision features (DiNOv2 base is 768 dimensions)
    test_features_1 = np.random.randn(50, 768).astype(np.float32)
    test_features_2 = np.random.randn(50, 768).astype(np.float32)

    feat1_file = dataset_dir / "cc_3M_dino_v2_states_1of2.npy"
    feat2_file = dataset_dir / "cc_3M_dino_v2_states_2of2.npy"

    np.save(feat1_file, test_features_1)
    np.save(feat2_file, test_features_2)

    logger.info(f"✅ Created test captions: {len(test_captions)} samples")
    logger.info(
        f"✅ Created test features: {test_features_1.shape} + {test_features_2.shape}")
    logger.info("🧪 Test dataset ready for development!")

    return True


def main():
    """Main function to download or verify BabyLM dataset"""
    logger.info("🚀 BabyLM Dataset Setup for BitMar")
    logger.info("=" * 50)

    # First check if data already exists
    data_exists, dataset_dir = verify_existing_data()

    if data_exists:
        logger.info("✅ Dataset is complete. No download needed!")
        return True

    # If train_50M.zip exists but not extracted, extract it
    train_zip_path = dataset_dir / "train_50M.zip"
    train_dir = dataset_dir / "train_50M"

    if train_zip_path.exists() and not train_dir.exists():
        logger.info("📦 Extracting train_50M.zip...")
        extract_result = extract_train_50M_if_needed(dataset_dir)
        if extract_result:
            logger.info("✅ train_50M.zip extracted successfully!")
            # Check again after extraction
            data_exists, _ = verify_existing_data()
            if data_exists:
                return True
        else:
            logger.error("❌ Failed to extract train_50M.zip")

    # If we get here, some files are missing - try to download them
    logger.info("📥 Some files are missing. Attempting to download...")

    try:
        success = download_babylm_multimodal_dataset()
        if success:
            logger.info("✅ Download completed successfully!")
            # Extract train_50M.zip if it was downloaded using the proper function
            extract_result = extract_train_50M_if_needed(dataset_dir)
            if extract_result:
                logger.info("✅ train_50M.zip extracted successfully!")

            # Final verification
            data_exists, _ = verify_existing_data()
            return data_exists
        else:
            logger.error("❌ Download failed!")
            return False
    except Exception as e:
        logger.error(f"❌ Download failed with error: {e}")
        logger.info("📋 To complete the dataset manually, you need to:")
        logger.info("1. Download the files from the BabyLM OSF repository")
        logger.info("2. Place them in: " + str(dataset_dir.absolute()))
        logger.info("3. Run this script again to verify")
        return False


if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)
