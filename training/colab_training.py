#!/usr/bin/env python3
"""
ATIS Training for Google Colab
==============================

This script is designed to be run in Google Colab.
Copy and paste each section into separate Colab cells.

Author: Travel Bot Team
Date: 2024
"""

# ============================================================================
# CELL 1: Install Dependencies
# ============================================================================

"""
# Install required packages
!pip install torch transformers datasets tokenizers accelerate
!pip install pandas numpy scikit-learn tensorboard tqdm
!pip install matplotlib seaborn

print("✅ Dependencies installed successfully!")
"""

# ============================================================================
# CELL 2: Check GPU and Import Libraries
# ============================================================================

"""
import torch
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from datetime import datetime

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

# Check GPU availability
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
else:
    print("⚠️ No GPU detected. Training will be slower on CPU.")
"""

# ============================================================================
# CELL 3: Download ATIS Dataset
# ============================================================================

"""
# Download ATIS dataset
print("📥 Downloading ATIS dataset...")
atis_dataset = load_dataset("atis_intents")

print(f"✅ ATIS dataset downloaded!")
print(f"   Training samples: {len(atis_dataset['train'])}")
print(f"   Test samples: {len(atis_dataset['test'])}")

# Show sample data
print("\\n📋 Sample ATIS data:")
for i in range(3):
    print(f"   {i+1}. {atis_dataset['train'][i]['text']} → {atis_dataset['train'][i]['intent']}")
"""

# ============================================================================
# CELL 4: Data Preprocessing
# ============================================================================

"""
# Extract intents and create mapping
intents = sorted(list(set(atis_dataset['train']['intent'])))
intent_map = {intent: idx for idx, intent in enumerate(intents)}
reverse_intent_map = {idx: intent for intent, idx in intent_map.items()}

print(f"🎯 Found {len(intents)} intents:")
for intent in intents:
    print(f"   - {intent}")

# Prepare training data
train_texts = atis_dataset['train']['text']
train_intents = [intent_map[intent] for intent in atis_dataset['train']['intent']]

# Prepare test data
test_texts = atis_dataset['test']['text']
test_intents = [intent_map[intent] for intent in atis_dataset['test']['intent']]

# Split training data into train and validation
train_texts, val_texts, train_intents, val_intents = train_test_split(
    train_texts, train_intents, 
    test_size=0.2, 
    random_state=42,
    stratify=train_intents
)

# Create datasets
train_dataset = Dataset.from_dict({
    'text': train_texts,
    'intent': train_intents
})

val_dataset = Dataset.from_dict({
    'text': val_texts,
    'intent': val_intents
})

test_dataset = Dataset.from_dict({
    'text': test_texts,
    'intent': test_intents
})

print(f"\\n📊 Dataset splits:")
print(f"   Train: {len(train_dataset)} samples")
print(f"   Validation: {len(val_dataset)} samples")
print(f"   Test: {len(test_dataset)} samples")
"""

# ============================================================================
# CELL 5: Setup Model and Tokenizer
# ============================================================================

"""
# Model configuration
model_name = 'bert-base-uncased'
max_length = 128
num_labels = len(intent_map)

print(f"🤖 Loading {model_name}...")

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Load model
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=num_labels,
    ignore_mismatched_sizes=True
)

print(f"✅ Model loaded successfully!")
print(f"   Model parameters: {model.num_parameters():,}")
print(f"   Number of labels: {num_labels}")
"""

# ============================================================================
# CELL 6: Tokenize Data
# ============================================================================

"""
def tokenize_function(examples):
    return tokenizer(
        examples['text'],
        padding='max_length',
        truncation=True,
        max_length=max_length,
        return_tensors='pt'
    )

print("🔤 Tokenizing datasets...")

# Tokenize datasets
train_dataset_tokenized = train_dataset.map(
    tokenize_function, 
    batched=True, 
    remove_columns=train_dataset.column_names
)

val_dataset_tokenized = val_dataset.map(
    tokenize_function, 
    batched=True, 
    remove_columns=val_dataset.column_names
)

test_dataset_tokenized = test_dataset.map(
    tokenize_function, 
    batched=True, 
    remove_columns=test_dataset.column_names
)

print("✅ Tokenization completed!")
"""

# ============================================================================
# CELL 7: Setup Training
# ============================================================================

"""
def compute_metrics(eval_pred):
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

# Training arguments
training_args = TrainingArguments(
    output_dir="./atis_model",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    learning_rate=2e-5,
    warmup_steps=500,
    weight_decay=0.01,
    logging_steps=100,
    evaluation_strategy="steps",
    eval_steps=500,
    save_steps=1000,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    greater_is_better=True,
    save_total_limit=3,
    dataloader_num_workers=4,
    remove_unused_columns=False,
)

# Data collator
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# Initialize trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset_tokenized,
    eval_dataset=val_dataset_tokenized,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

print("🎯 Trainer setup completed!")
"""

# ============================================================================
# CELL 8: Train the Model
# ============================================================================

"""
print("🚀 Starting training...")
print("⏱️  This will take 10-30 minutes depending on your GPU")

# Train the model
trainer.train()

print("✅ Training completed!")
"""

# ============================================================================
# CELL 9: Evaluate the Model
# ============================================================================

"""
print("📊 Evaluating model on test set...")

# Evaluate on test set
results = trainer.evaluate(test_dataset_tokenized)

print("\\n🎯 Test Results:")
for key, value in results.items():
    if key.startswith('eval_'):
        metric_name = key.replace('eval_', '')
        print(f"   {metric_name.capitalize()}: {value:.4f}")

print("\\n✅ Evaluation completed!")
"""

# ============================================================================
# CELL 10: Save the Model
# ============================================================================

"""
import os

# Save the model
model_save_path = "./atis_intent_classifier"
trainer.save_model(model_save_path)
tokenizer.save_pretrained(model_save_path)

# Save intent mapping
with open(os.path.join(model_save_path, "intent_mapping.json"), 'w') as f:
    json.dump(intent_map, f, indent=2)

print(f"✅ Model saved to: {model_save_path}")
print(f"📁 Files saved:")
for file in os.listdir(model_save_path):
    print(f"   - {file}")
"""

# ============================================================================
# CELL 11: Test the Model
# ============================================================================

"""
def predict_intent(text):
    inputs = tokenizer(
        text,
        return_tensors='pt',
        truncation=True,
        padding=True,
        max_length=max_length
    )
    
    with torch.no_grad():
        outputs = model(**inputs)
        probabilities = torch.softmax(outputs.logits, dim=1)
        predicted_id = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0][predicted_id].item()
    
    intent = reverse_intent_map.get(predicted_id, "unknown")
    
    return {
        'text': text,
        'intent': intent,
        'confidence': confidence
    }

# Test sample queries
test_queries = [
    "I want to fly from Boston to New York",
    "What's the weather like in Chicago?",
    "Book me a hotel in Los Angeles",
    "What time does my flight depart?",
    "I need to cancel my reservation"
]

print("🧪 Testing model on sample queries:")
print("=" * 60)

for query in test_queries:
    result = predict_intent(query)
    print(f"Query: {result['text']}")
    print(f"Intent: {result['intent']}")
    print(f"Confidence: {result['confidence']:.3f}")
    print("-" * 50)
"""

# ============================================================================
# CELL 12: Download the Model
# ============================================================================

"""
from google.colab import files
import zipfile

# Create a zip file of the model
zip_path = "atis_model.zip"
with zipfile.ZipFile(zip_path, 'w') as zipf:
    for root, dirs, files in os.walk(model_save_path):
        for file in files:
            file_path = os.path.join(root, file)
            arcname = os.path.relpath(file_path, model_save_path)
            zipf.write(file_path, arcname)

print(f"📦 Model zipped as: {zip_path}")
print("📥 Downloading model...")

# Download the zip file
files.download(zip_path)

print("✅ Model downloaded successfully!")
print("\\n🎉 Training completed! You can now use this model in your travel bot.")
"""

# ============================================================================
# CELL 13: Interactive Testing (Optional)
# ============================================================================

"""
# Interactive testing
print("🔍 Interactive Testing Mode")
print("Type 'quit' to exit")
print("-" * 50)

while True:
    try:
        query = input("Enter your query: ").strip()
        
        if query.lower() in ['quit', 'exit', 'q']:
            break
        
        if not query:
            continue
        
        result = predict_intent(query)
        
        print(f"Intent: {result['intent']}")
        print(f"Confidence: {result['confidence']:.3f}")
        print("-" * 50)
        
    except KeyboardInterrupt:
        break
    except Exception as e:
        print(f"Error: {e}")

print("Interactive testing ended")
"""

print("""
🎯 INSTRUCTIONS FOR GOOGLE COLAB:

1. Go to https://colab.research.google.com/
2. Create a new notebook
3. Enable GPU: Runtime → Change runtime type → GPU
4. Copy each cell (between the triple quotes) into separate Colab cells
5. Run cells 1-12 in order
6. Download your trained model!

Expected time: 15-30 minutes
Cost: FREE
""") 