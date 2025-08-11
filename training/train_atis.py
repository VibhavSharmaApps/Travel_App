#!/usr/bin/env python3
"""
ATIS Training Script
===================

This script trains a custom intent classification model on the ATIS dataset.
It includes data preprocessing, model training, evaluation, and model saving.

Usage:
    python training/train_atis.py

Author: Travel Bot Team
Date: 2024
"""

import os
import sys
import logging
import json
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from datetime import datetime

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    TrainingArguments, 
    Trainer,
    DataCollatorWithPadding
)
from datasets import load_dataset, Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import pandas as pd

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training/logs/atis_training.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class ATISTrainer:
    """ATIS Dataset Trainer"""
    
    def __init__(self, config: Dict):
        """
        Initialize ATIS trainer
        
        Args:
            config: Training configuration dictionary
        """
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = None
        self.model = None
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.intent_map = {}
        self.reverse_intent_map = {}
        
        logger.info(f"Using device: {self.device}")
    
    def load_atis_dataset(self) -> Dict:
        """
        Load and preprocess ATIS dataset
        
        Returns:
            Dictionary containing train, validation, and test datasets
        """
        logger.info("Loading ATIS dataset...")
        
        try:
            # Load ATIS dataset from Hugging Face
            dataset = load_dataset("atis_intents")
            
            # Extract intents and create mapping
            intents = sorted(list(set(dataset['train']['intent'])))
            self.intent_map = {intent: idx for idx, intent in enumerate(intents)}
            self.reverse_intent_map = {idx: intent for intent, idx in self.intent_map.items()}
            
            logger.info(f"Found {len(intents)} intents: {intents}")
            logger.info(f"Training samples: {len(dataset['train'])}")
            logger.info(f"Test samples: {len(dataset['test'])}")
            
            return dataset
            
        except Exception as e:
            logger.error(f"Failed to load ATIS dataset: {e}")
            raise
    
    def preprocess_data(self, dataset: Dict) -> Tuple[Dataset, Dataset, Dataset]:
        """
        Preprocess ATIS data for training
        
        Args:
            dataset: Raw ATIS dataset
            
        Returns:
            Tuple of (train_dataset, val_dataset, test_dataset)
        """
        logger.info("Preprocessing ATIS data...")
        
        # Prepare training data
        train_texts = dataset['train']['text']
        train_intents = [self.intent_map[intent] for intent in dataset['train']['intent']]
        
        # Prepare test data
        test_texts = dataset['test']['text']
        test_intents = [self.intent_map[intent] for intent in dataset['test']['intent']]
        
        # Split training data into train and validation
        train_texts, val_texts, train_intents, val_intents = train_test_split(
            train_texts, train_intents, 
            test_size=0.2, 
            random_state=42,
            stratify=train_intents
        )
        
        # Create datasets
        train_data = {
            'text': train_texts,
            'intent': train_intents
        }
        
        val_data = {
            'text': val_texts,
            'intent': val_intents
        }
        
        test_data = {
            'text': test_texts,
            'intent': test_intents
        }
        
        self.train_dataset = Dataset.from_dict(train_data)
        self.val_dataset = Dataset.from_dict(val_data)
        self.test_dataset = Dataset.from_dict(test_data)
        
        logger.info(f"Train samples: {len(self.train_dataset)}")
        logger.info(f"Validation samples: {len(self.val_dataset)}")
        logger.info(f"Test samples: {len(self.test_dataset)}")
        
        return self.train_dataset, self.val_dataset, self.test_dataset
    
    def setup_tokenizer_and_model(self):
        """Setup tokenizer and model"""
        logger.info("Setting up tokenizer and model...")
        
        model_name = self.config['model_name']
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Add padding token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model
        num_labels = len(self.intent_map)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels,
            ignore_mismatched_sizes=True
        )
        
        logger.info(f"Model: {model_name}")
        logger.info(f"Number of labels: {num_labels}")
        logger.info(f"Model parameters: {self.model.num_parameters():,}")
    
    def tokenize_function(self, examples):
        """
        Tokenize examples for training
        
        Args:
            examples: Batch of examples
            
        Returns:
            Tokenized examples
        """
        return self.tokenizer(
            examples['text'],
            padding='max_length',
            truncation=True,
            max_length=self.config['max_length'],
            return_tensors='pt'
        )
    
    def compute_metrics(self, eval_pred):
        """
        Compute evaluation metrics
        
        Args:
            eval_pred: Tuple of (predictions, labels)
            
        Returns:
            Dictionary of metrics
        """
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, predictions, average='weighted'
        )
        accuracy = accuracy_score(labels, predictions)
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
    
    def train_model(self):
        """Train the model"""
        logger.info("Starting model training...")
        
        # Tokenize datasets
        train_dataset = self.train_dataset.map(
            self.tokenize_function, 
            batched=True, 
            remove_columns=self.train_dataset.column_names
        )
        val_dataset = self.val_dataset.map(
            self.tokenize_function, 
            batched=True, 
            remove_columns=self.val_dataset.column_names
        )
        
        # Setup training arguments
        training_args = TrainingArguments(
            output_dir=self.config['output_dir'],
            num_train_epochs=self.config['num_epochs'],
            per_device_train_batch_size=self.config['batch_size'],
            per_device_eval_batch_size=self.config['batch_size'],
            learning_rate=self.config['learning_rate'],
            warmup_steps=self.config['warmup_steps'],
            weight_decay=self.config['weight_decay'],
            logging_dir=self.config['logging_dir'],
            logging_steps=self.config['logging_steps'],
            evaluation_strategy="steps",
            eval_steps=self.config['eval_steps'],
            save_steps=self.config['save_steps'],
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            greater_is_better=True,
            save_total_limit=3,
            dataloader_num_workers=4,
            remove_unused_columns=False,
        )
        
        # Setup data collator
        data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)
        
        # Initialize trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
            compute_metrics=self.compute_metrics,
        )
        
        # Train the model
        logger.info("Training started...")
        trainer.train()
        
        # Save the model
        model_save_path = Path(self.config['output_dir']) / "atis_intent_classifier"
        trainer.save_model(str(model_save_path))
        self.tokenizer.save_pretrained(str(model_save_path))
        
        # Save intent mapping
        intent_mapping_file = model_save_path / "intent_mapping.json"
        with open(intent_mapping_file, 'w') as f:
            json.dump(self.intent_map, f, indent=2)
        
        logger.info(f"Model saved to: {model_save_path}")
        
        return trainer
    
    def evaluate_model(self, trainer):
        """
        Evaluate the trained model
        
        Args:
            trainer: Trained trainer instance
        """
        logger.info("Evaluating model...")
        
        # Tokenize test dataset
        test_dataset = self.test_dataset.map(
            self.tokenize_function, 
            batched=True, 
            remove_columns=self.test_dataset.column_names
        )
        
        # Evaluate on test set
        results = trainer.evaluate(test_dataset)
        
        logger.info("Test Results:")
        for key, value in results.items():
            logger.info(f"  {key}: {value:.4f}")
        
        # Save results
        results_file = Path(self.config['output_dir']) / "test_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Results saved to: {results_file}")
        
        return results
    
    def run_training(self):
        """Run complete training pipeline"""
        logger.info("🚀 Starting ATIS training pipeline...")
        
        try:
            # Step 1: Load dataset
            dataset = self.load_atis_dataset()
            
            # Step 2: Preprocess data
            self.preprocess_data(dataset)
            
            # Step 3: Setup model and tokenizer
            self.setup_tokenizer_and_model()
            
            # Step 4: Train model
            trainer = self.train_model()
            
            # Step 5: Evaluate model
            results = self.evaluate_model(trainer)
            
            logger.info("🎉 ATIS training completed successfully!")
            
            return trainer, results
            
        except Exception as e:
            logger.error(f"❌ Training failed: {e}")
            raise

def main():
    """Main training function"""
    
    # Training configuration
    config = {
        'model_name': 'bert-base-uncased',
        'max_length': 128,
        'batch_size': 16,
        'learning_rate': 2e-5,
        'num_epochs': 3,
        'warmup_steps': 500,
        'weight_decay': 0.01,
        'logging_steps': 100,
        'eval_steps': 500,
        'save_steps': 1000,
        'output_dir': './training/models',
        'logging_dir': './training/logs',
    }
    
    # Create trainer and run training
    trainer = ATISTrainer(config)
    trainer.run_training()

if __name__ == "__main__":
    main() 