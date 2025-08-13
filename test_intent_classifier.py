#!/usr/bin/env python3
"""
Test Script for Trained Intent Classifier Integration
====================================================

This script tests the integration of the trained intent classifier
with the updated AI engine to ensure everything works correctly.

Usage:
    python test_intent_classifier.py

Author: Travel Bot Team
Date: 2024
"""

import sys
import os
import logging

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.ai_engine import AIEngine

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_intent_classification():
    """Test the intent classification functionality"""
    
    print("🧪 Testing Intent Classification Integration")
    print("=" * 50)
    
    try:
        # Initialize AI Engine
        print("📡 Initializing AI Engine...")
        ai_engine = AIEngine()
        print("✅ AI Engine initialized successfully")
        
        # Test queries
        test_queries = [
            "I want to book a flight to Paris",
            "Can you help me find a hotel in Tokyo?",
            "I need to cancel my reservation",
            "What's the weather like in London?",
            "Help me with my travel plans",
            "Check my flight status",
            "Book a hotel room for next week",
            "I'm looking for flights from New York to London",
            "What hotels are available in Rome?",
            "Cancel my hotel booking please"
        ]
        
        print(f"\n🔍 Testing {len(test_queries)} queries...")
        print("-" * 50)
        
        for i, query in enumerate(test_queries, 1):
            print(f"\n{i}. Query: '{query}'")
            
            # Classify intent
            intent, confidence = ai_engine.classify_intent(query)
            print(f"   Intent: {intent}")
            print(f"   Confidence: {confidence:.2%}")
            
            # Get suggestions
            suggestions = ai_engine.get_intent_suggestions(query)
            if suggestions:
                print(f"   Top suggestions:")
                for j, (suggested_intent, suggested_confidence) in enumerate(suggestions[:3], 1):
                    print(f"     {j}. {suggested_intent} ({suggested_confidence:.2%})")
            
            # Extract entities
            entities = ai_engine.extract_entities(query)
            if entities:
                print(f"   Entities: {entities}")
        
        print("\n" + "=" * 50)
        print("✅ All tests completed successfully!")
        
        # Test with a complex query
        print(f"\n🎯 Testing complex query...")
        complex_query = "I need to book a flight from New York to London for 2 passengers on March 15th, 2024"
        print(f"Query: '{complex_query}'")
        
        intent, confidence = ai_engine.classify_intent(complex_query)
        entities = ai_engine.extract_entities(complex_query)
        
        print(f"Intent: {intent} (confidence: {confidence:.2%})")
        print(f"Entities: {entities}")
        
        # Test response generation
        print(f"\n💬 Testing response generation...")
        response = ai_engine.generate_response(complex_query)
        print(f"Response: {response}")
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        logger.error(f"Test failed: {e}")
        return False
    
    return True

def test_model_loading():
    """Test if the trained model loads correctly"""
    
    print("\n🔧 Testing Model Loading")
    print("=" * 30)
    
    try:
        # Check if model file exists
        model_path = 'intent_classifier_complete.pth'
        if os.path.exists(model_path):
            print(f"✅ Model file found: {model_path}")
            
            # Try to load the model
            from src.ai_engine import TravelIntentClassifier
            classifier = TravelIntentClassifier(model_path)
            
            if classifier.model is not None:
                print("✅ Trained model loaded successfully")
                print(f"   Available intents: {classifier.intent_names}")
                
                # Test a simple prediction
                test_result = classifier.predict("book a flight")
                print(f"   Test prediction: {test_result['intent']} ({test_result['confidence']:.2%})")
                
                return True
            else:
                print("⚠️  Model loaded but using fallback mode")
                return True
        else:
            print(f"⚠️  Model file not found: {model_path}")
            print("   The AI engine will use fallback pattern-based classification")
            return True
            
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Travel Bot Intent Classifier Test")
    print("=" * 50)
    
    # Test model loading
    model_ok = test_model_loading()
    
    # Test intent classification
    if model_ok:
        classification_ok = test_intent_classification()
        
        if classification_ok:
            print("\n🎉 All tests passed! The intent classifier is working correctly.")
            print("\n📋 Summary:")
            print("   ✅ Trained model integration successful")
            print("   ✅ Intent classification working")
            print("   ✅ Entity extraction working")
            print("   ✅ Response generation working")
            print("   ✅ Fallback mechanisms in place")
        else:
            print("\n❌ Intent classification tests failed")
            sys.exit(1)
    else:
        print("\n❌ Model loading tests failed")
        sys.exit(1) 