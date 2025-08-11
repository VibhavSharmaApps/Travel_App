#!/usr/bin/env python3
"""
Training Environment Setup Script
================================

This script sets up the complete training environment for ATIS and travel domain models.
It downloads required datasets, pre-trained models, and creates necessary directories.

Author: Travel Bot Team
Date: 2024
"""

import os
import sys
import subprocess
import logging
from pathlib import Path
from typing import List, Dict

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TrainingSetup:
    """Setup class for training environment"""
    
    def __init__(self):
        self.base_dir = Path(__file__).parent.parent
        self.training_dir = self.base_dir / "training"
        self.models_dir = self.training_dir / "models"
        self.data_dir = self.training_dir / "data"
        self.logs_dir = self.training_dir / "logs"
        
    def setup_directories(self):
        """Create necessary directories for training"""
        logger.info("Creating training directories...")
        
        directories = [
            self.training_dir,
            self.models_dir,
            self.data_dir,
            self.logs_dir,
            self.training_dir / "checkpoints",
            self.training_dir / "results"
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            logger.info(f"✅ Created: {directory}")
    
    def install_dependencies(self):
        """Install training dependencies"""
        logger.info("Installing training dependencies...")
        
        try:
            # Install from requirements_training.txt
            requirements_file = self.base_dir / "requirements_training.txt"
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", "-r", str(requirements_file)
            ])
            logger.info("✅ Training dependencies installed successfully")
            
        except subprocess.CalledProcessError as e:
            logger.error(f"❌ Failed to install dependencies: {e}")
            raise
    
    def download_datasets(self):
        """Download required datasets"""
        logger.info("Downloading datasets...")
        
        try:
            # Import after installation
            from datasets import load_dataset
            
            # Download ATIS dataset
            logger.info("Downloading ATIS dataset...")
            atis_dataset = load_dataset("atis_intents")
            logger.info(f"✅ ATIS dataset downloaded: {len(atis_dataset['train'])} training samples")
            
            # Download MultiWOZ dataset (optional)
            logger.info("Downloading MultiWOZ dataset...")
            multiwoz_dataset = load_dataset("multiwoz_v2.1")
            logger.info(f"✅ MultiWOZ dataset downloaded: {len(multiwoz_dataset['train'])} conversations")
            
            # Download SNIPS dataset (optional)
            logger.info("Downloading SNIPS dataset...")
            snips_dataset = load_dataset("snips_built_in_intents")
            logger.info(f"✅ SNIPS dataset downloaded: {len(snips_dataset['train'])} samples")
            
        except Exception as e:
            logger.error(f"❌ Failed to download datasets: {e}")
            raise
    
    def download_pretrained_models(self):
        """Download pre-trained models"""
        logger.info("Downloading pre-trained models...")
        
        try:
            from transformers import AutoTokenizer, AutoModel
            
            models_to_download = [
                "bert-base-uncased",
                "distilbert-base-uncased",
                "roberta-base",
                "facebook/blenderbot-400M-distill"
            ]
            
            for model_name in models_to_download:
                logger.info(f"Downloading {model_name}...")
                
                # Download tokenizer
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                tokenizer.save_pretrained(self.models_dir / model_name.replace("/", "_"))
                
                # Download model
                model = AutoModel.from_pretrained(model_name)
                model.save_pretrained(self.models_dir / model_name.replace("/", "_"))
                
                logger.info(f"✅ Downloaded {model_name}")
                
        except Exception as e:
            logger.error(f"❌ Failed to download models: {e}")
            raise
    
    def create_training_config(self):
        """Create training configuration file"""
        logger.info("Creating training configuration...")
        
        config_content = """
# Training Configuration
TRAINING_CONFIG = {
    # Model settings
    'model_name': 'bert-base-uncased',
    'max_length': 128,
    'batch_size': 16,
    'learning_rate': 2e-5,
    'num_epochs': 3,
    
    # Data settings
    'train_split': 0.8,
    'val_split': 0.1,
    'test_split': 0.1,
    
    # Training settings
    'warmup_steps': 500,
    'weight_decay': 0.01,
    'gradient_accumulation_steps': 1,
    
    # Logging settings
    'logging_steps': 100,
    'eval_steps': 500,
    'save_steps': 1000,
    
    # Output settings
    'output_dir': './training/models',
    'logging_dir': './training/logs',
}
"""
        
        config_file = self.training_dir / "config.py"
        with open(config_file, 'w') as f:
            f.write(config_content)
        
        logger.info(f"✅ Created training config: {config_file}")
    
    def run_setup(self):
        """Run complete setup process"""
        logger.info("🚀 Starting training environment setup...")
        
        try:
            # Step 1: Create directories
            self.setup_directories()
            
            # Step 2: Install dependencies
            self.install_dependencies()
            
            # Step 3: Download datasets
            self.download_datasets()
            
            # Step 4: Download pre-trained models
            self.download_pretrained_models()
            
            # Step 5: Create configuration
            self.create_training_config()
            
            logger.info("🎉 Training environment setup completed successfully!")
            logger.info("")
            logger.info("Next steps:")
            logger.info("1. Run: python training/train_atis.py")
            logger.info("2. Run: python training/train_custom_travel.py")
            logger.info("3. Check logs in: training/logs/")
            
        except Exception as e:
            logger.error(f"❌ Setup failed: {e}")
            raise

def main():
    """Main setup function"""
    setup = TrainingSetup()
    setup.run_setup()

if __name__ == "__main__":
    main() 