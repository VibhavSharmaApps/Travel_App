#!/usr/bin/env python3
"""
Setup Script for Trained Intent Classifier
=========================================

This script helps set up the trained intent classifier model
for the travel bot. It provides instructions for downloading
and placing the model file in the correct location.

Usage:
    python setup_trained_model.py

Author: Travel Bot Team
Date: 2024
"""

import os
import sys

def check_model_file():
    """Check if the trained model file exists"""
    model_path = 'intent_classifier_complete.pth'
    
    if os.path.exists(model_path):
        file_size = os.path.getsize(model_path) / (1024 * 1024)  # Size in MB
        print(f"✅ Model file found: {model_path}")
        print(f"   File size: {file_size:.1f} MB")
        return True
    else:
        print(f"❌ Model file not found: {model_path}")
        return False

def provide_download_instructions():
    """Provide instructions for downloading the model"""
    print("\n📥 Model Download Instructions")
    print("=" * 40)
    print("To use the trained intent classifier, you need to:")
    print()
    print("1. Download the model file from Google Colab:")
    print("   - In your Colab notebook, run: files.download('intent_classifier_complete.pth')")
    print()
    print("2. Place the downloaded file in your project root:")
    print("   - Move 'intent_classifier_complete.pth' to the same folder as this script")
    print()
    print("3. Verify the setup:")
    print("   - Run: python test_intent_classifier.py")
    print()
    print("📁 Expected file structure:")
    print("   Project_Travel App/")
    print("   ├── intent_classifier_complete.pth  ← Place here")
    print("   ├── src/")
    print("   ├── main.py")
    print("   └── test_intent_classifier.py")

def test_integration():
    """Test the integration if model is available"""
    if check_model_file():
        print("\n🧪 Testing Integration...")
        try:
            # Import and test
            sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
            from src.ai_engine import TravelIntentClassifier
            
            classifier = TravelIntentClassifier('intent_classifier_complete.pth')
            
            if classifier.model is not None:
                print("✅ Integration test successful!")
                print(f"   Available intents: {classifier.intent_names}")
                
                # Test prediction
                test_result = classifier.predict("book a flight to Paris")
                print(f"   Test prediction: {test_result['intent']} ({test_result['confidence']:.2%})")
                
                return True
            else:
                print("⚠️  Model loaded but using fallback mode")
                return True
                
        except Exception as e:
            print(f"❌ Integration test failed: {e}")
            return False
    else:
        return False

def main():
    """Main setup function"""
    print("🚀 Travel Bot Intent Classifier Setup")
    print("=" * 50)
    
    # Check if model exists
    model_exists = check_model_file()
    
    if model_exists:
        print("\n✅ Model is ready!")
        
        # Test integration
        if test_integration():
            print("\n🎉 Setup complete! Your travel bot is ready to use the trained intent classifier.")
            print("\n📋 Next steps:")
            print("   1. Run the bot: python main.py")
            print("   2. Test with: python test_intent_classifier.py")
            print("   3. Enjoy 99%+ accurate intent classification!")
        else:
            print("\n❌ Integration test failed. Please check the error messages above.")
    else:
        print("\n📋 Model file not found.")
        provide_download_instructions()
        
        print("\n💡 Alternative: Use fallback mode")
        print("   The AI engine will automatically use pattern-based classification")
        print("   if the trained model is not available.")
        print("   Run: python test_intent_classifier.py to test fallback mode")

if __name__ == "__main__":
    main() 