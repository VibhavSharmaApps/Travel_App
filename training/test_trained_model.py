#!/usr/bin/env python3
"""
Test Trained ATIS Model
======================

This script tests the trained ATIS intent classification model.
It loads the trained model and tests it on sample queries.

Usage:
    python training/test_trained_model.py

Author: Travel Bot Team
Date: 2024
"""

import sys
import json
import torch
import logging
from pathlib import Path
from typing import List, Dict

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ATISModelTester:
    """Test trained ATIS model"""
    
    def __init__(self, model_path: str):
        """
        Initialize model tester
        
        Args:
            model_path: Path to trained model
        """
        self.model_path = Path(model_path)
        self.tokenizer = None
        self.model = None
        self.intent_map = {}
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.load_model()
    
    def load_model(self):
        """Load trained model and tokenizer"""
        logger.info(f"Loading model from: {self.model_path}")
        
        try:
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            
            # Load model
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_path)
            self.model.to(self.device)
            self.model.eval()
            
            # Load intent mapping
            intent_mapping_file = self.model_path / "intent_mapping.json"
            if intent_mapping_file.exists():
                with open(intent_mapping_file, 'r') as f:
                    self.intent_map = json.load(f)
            
            # Create reverse mapping
            self.reverse_intent_map = {v: k for k, v in self.intent_map.items()}
            
            logger.info(f"Model loaded successfully")
            logger.info(f"Number of intents: {len(self.intent_map)}")
            logger.info(f"Intents: {list(self.intent_map.keys())}")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def predict_intent(self, text: str) -> Dict:
        """
        Predict intent for given text
        
        Args:
            text: Input text
            
        Returns:
            Dictionary with intent and confidence
        """
        try:
            # Tokenize input
            inputs = self.tokenizer(
                text,
                return_tensors='pt',
                truncation=True,
                padding=True,
                max_length=128
            )
            
            # Move to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Get prediction
            with torch.no_grad():
                outputs = self.model(**inputs)
                probabilities = torch.softmax(outputs.logits, dim=1)
                predicted_id = torch.argmax(probabilities, dim=1).item()
                confidence = probabilities[0][predicted_id].item()
            
            # Get intent name
            intent = self.reverse_intent_map.get(predicted_id, "unknown")
            
            return {
                'text': text,
                'intent': intent,
                'confidence': confidence,
                'intent_id': predicted_id
            }
            
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            return {
                'text': text,
                'intent': 'error',
                'confidence': 0.0,
                'intent_id': -1
            }
    
    def test_sample_queries(self):
        """Test model on sample queries"""
        logger.info("Testing model on sample queries...")
        
        # Sample ATIS queries
        sample_queries = [
            "I want to fly from Boston to New York",
            "What's the weather like in Chicago?",
            "Book me a hotel in Los Angeles",
            "What time does my flight depart?",
            "I need to cancel my reservation",
            "Show me flights to Miami",
            "What's the gate number for my flight?",
            "I want to change my booking",
            "What's the baggage allowance?",
            "Is my flight on time?"
        ]
        
        results = []
        for query in sample_queries:
            result = self.predict_intent(query)
            results.append(result)
            
            logger.info(f"Query: {query}")
            logger.info(f"Intent: {result['intent']} (confidence: {result['confidence']:.3f})")
            logger.info("-" * 50)
        
        return results
    
    def test_custom_queries(self, queries: List[str]):
        """
        Test model on custom queries
        
        Args:
            queries: List of custom queries to test
        """
        logger.info(f"Testing {len(queries)} custom queries...")
        
        results = []
        for query in queries:
            result = self.predict_intent(query)
            results.append(result)
            
            logger.info(f"Query: {query}")
            logger.info(f"Intent: {result['intent']} (confidence: {result['confidence']:.3f})")
            logger.info("-" * 50)
        
        return results
    
    def interactive_test(self):
        """Interactive testing mode"""
        logger.info("Starting interactive testing mode...")
        logger.info("Type 'quit' to exit")
        logger.info("-" * 50)
        
        while True:
            try:
                query = input("Enter your query: ").strip()
                
                if query.lower() in ['quit', 'exit', 'q']:
                    break
                
                if not query:
                    continue
                
                result = self.predict_intent(query)
                
                print(f"Intent: {result['intent']}")
                print(f"Confidence: {result['confidence']:.3f}")
                print(f"Intent ID: {result['intent_id']}")
                print("-" * 50)
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                logger.error(f"Error: {e}")
        
        logger.info("Interactive testing ended")

def main():
    """Main testing function"""
    
    # Model path
    model_path = "./training/models/atis_intent_classifier"
    
    if not Path(model_path).exists():
        logger.error(f"Model not found at: {model_path}")
        logger.info("Please run the training script first: python training/train_atis.py")
        return
    
    # Create tester
    tester = ATISModelTester(model_path)
    
    # Test sample queries
    logger.info("=" * 60)
    logger.info("TESTING SAMPLE QUERIES")
    logger.info("=" * 60)
    sample_results = tester.test_sample_queries()
    
    # Interactive testing
    logger.info("=" * 60)
    logger.info("INTERACTIVE TESTING")
    logger.info("=" * 60)
    tester.interactive_test()
    
    logger.info("Testing completed!")

if __name__ == "__main__":
    main() 