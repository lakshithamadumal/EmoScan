#!/usr/bin/env python3
"""
EmoScan: Model Downloader Utility
Downloads and sets up pre-trained models for emotion recognition
"""

import os
import sys
import requests
import zipfile
import tarfile
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ModelDownloader:
    """Utility class for downloading and setting up emotion recognition models"""
    
    def __init__(self):
        self.models_dir = Path("models")
        self.models_dir.mkdir(exist_ok=True)
        
        # Model URLs and configurations
        self.model_configs = {
            'deepface_models': {
                'url': 'https://github.com/serengil/deepface_models/releases/download/v1.0/facial_expression_model_weights.h5',
                'filename': 'facial_expression_model_weights.h5',
                'description': 'DeepFace facial expression recognition model'
            },
            'opencv_haar': {
                'url': 'https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml',
                'filename': 'haarcascade_frontalface_default.xml',
                'description': 'OpenCV Haar Cascade for face detection'
            }
        }
    
    def download_file(self, url, filename, description):
        """Download a file from URL with progress tracking"""
        filepath = self.models_dir / filename
        
        if filepath.exists():
            logger.info(f"Model {description} already exists at {filepath}")
            return True
        
        logger.info(f"Downloading {description} from {url}")
        
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            downloaded = 0
            
            with open(filepath, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        
                        # Progress tracking
                        if total_size > 0:
                            progress = (downloaded / total_size) * 100
                            sys.stdout.write(f"\rDownloading {description}: {progress:.1f}%")
                            sys.stdout.flush()
            
            print(f"\n✅ Successfully downloaded {description}")
            return True
            
        except Exception as e:
            logger.error(f"Error downloading {description}: {e}")
            if filepath.exists():
                filepath.unlink()
            return False
    
    def download_all_models(self):
        """Download all required models"""
        logger.info("Starting model download process...")
        
        success_count = 0
        total_models = len(self.model_configs)
        
        for model_name, config in self.model_configs.items():
            logger.info(f"Processing {model_name}...")
            
            if self.download_file(
                config['url'], 
                config['filename'], 
                config['description']
            ):
                success_count += 1
        
        logger.info(f"Download completed: {success_count}/{total_models} models downloaded successfully")
        
        if success_count == total_models:
            logger.info("🎉 All models downloaded successfully!")
            return True
        else:
            logger.warning(f"⚠️  {total_models - success_count} models failed to download")
            return False
    
    def verify_models(self):
        """Verify that all required models are present and valid"""
        logger.info("Verifying downloaded models...")
        
        missing_models = []
        
        for model_name, config in self.model_configs.items():
            filepath = self.models_dir / config['filename']
            
            if not filepath.exists():
                missing_models.append(config['description'])
                logger.warning(f"Missing model: {config['description']}")
            else:
                file_size = filepath.stat().st_size
                logger.info(f"✅ {config['description']}: {file_size:,} bytes")
        
        if missing_models:
            logger.error(f"Missing models: {', '.join(missing_models)}")
            return False
        else:
            logger.info("🎉 All models verified successfully!")
            return True
    
    def setup_deepface_models(self):
        """Setup DeepFace models in the correct directory structure"""
        logger.info("Setting up DeepFace models...")
        
        try:
            # DeepFace expects models in a specific directory structure
            deepface_dir = Path.home() / ".deepface" / "weights"
            deepface_dir.mkdir(parents=True, exist_ok=True)
            
            # Copy or symlink the downloaded model
            source_model = self.models_dir / "facial_expression_model_weights.h5"
            target_model = deepface_dir / "facial_expression_model_weights.h5"
            
            if source_model.exists():
                if target_model.exists():
                    target_model.unlink()
                
                # Create symlink on Unix systems, copy on Windows
                if os.name == 'nt':  # Windows
                    import shutil
                    shutil.copy2(source_model, target_model)
                else:  # Unix-like systems
                    target_model.symlink_to(source_model)
                
                logger.info("✅ DeepFace models setup completed")
                return True
            else:
                logger.warning("⚠️  DeepFace model not found, skipping setup")
                return False
                
        except Exception as e:
            logger.error(f"Error setting up DeepFace models: {e}")
            return False
    
    def create_model_info(self):
        """Create a model information file"""
        info_file = self.models_dir / "model_info.txt"
        
        info_content = """EmoScan Model Information
============================

This directory contains pre-trained models for facial emotion recognition.

Models included:
1. facial_expression_model_weights.h5 - DeepFace emotion recognition model
2. haarcascade_frontalface_default.xml - OpenCV face detection cascade

Model Sources:
- DeepFace: https://github.com/serengil/deepface
- OpenCV: https://opencv.org/

Usage:
- The DeepFace model is automatically used by the emotion detection system
- The Haar cascade is used for face detection in video streams

For more information, see the main README.md file.
"""
        
        try:
            with open(info_file, 'w') as f:
                f.write(info_content)
            logger.info("✅ Model information file created")
        except Exception as e:
            logger.error(f"Error creating model info file: {e}")

def main():
    """Main function for model downloader"""
    print("🤖 EmoScan Model Downloader")
    print("=" * 40)
    
    downloader = ModelDownloader()
    
    # Download all models
    if downloader.download_all_models():
        # Verify models
        if downloader.verify_models():
            # Setup DeepFace models
            downloader.setup_deepface_models()
            # Create model info
            downloader.create_model_info()
            
            print("\n🎉 Model setup completed successfully!")
            print("You can now run EmoScan with: python main.py")
        else:
            print("\n❌ Model verification failed!")
            sys.exit(1)
    else:
        print("\n❌ Model download failed!")
        sys.exit(1)

if __name__ == "__main__":
    main() 