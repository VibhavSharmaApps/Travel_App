# Training Guide: Customizing Transformers for Travel Domain

## Overview

This guide explains how to train and fine-tune transformer models on travel-specific datasets like ATIS (Airline Travel Information System) and other relevant datasets to improve the travel bot's natural language understanding capabilities.

## Table of Contents

1. [Understanding the ATIS Dataset](#understanding-the-atis-dataset)
2. [Data Preparation](#data-preparation)
3. [Model Architecture Selection](#model-architecture-selection)
4. [Training Process](#training-process)
5. [Fine-tuning Strategies](#fine-tuning-strategies)
6. [Evaluation and Testing](#evaluation-and-testing)
7. [Integration with Travel Bot](#integration-with-travel-bot)
8. [Additional Datasets](#additional-datasets)

## Understanding the ATIS Dataset

### What is ATIS?

The ATIS (Airline Travel Information System) dataset is a widely-used benchmark for natural language understanding in the travel domain. It contains:

- **Intent Classification**: 18 different intents (e.g., flight, ground_service, meal)
- **Slot Filling**: 127 different slot types (e.g., departure_city, arrival_city, date)
- **Domain**: Airline travel information queries

### Dataset Statistics

```
Total utterances: ~5,000
Training set: ~4,000
Test set: ~1,000
Intents: 18
Slots: 127
Vocabulary size: ~900
```

### Sample ATIS Data

```json
{
  "text": "I want to fly from Boston to New York on March 15th",
  "intent": "flight",
  "slots": {
    "departure_city": "Boston",
    "arrival_city": "New York",
    "date": "March 15th"
  }
}
```

## Data Preparation

### 1. Download and Preprocess ATIS

```python
# Download ATIS dataset
from datasets import load_dataset

# Load ATIS dataset from Hugging Face
atis_dataset = load_dataset("atis_intents")

# Alternative: Download from original source
# https://github.com/huggingface/datasets/tree/master/datasets/atis_intents
```

### 2. Data Cleaning and Preprocessing

```python
import pandas as pd
import re
from typing import List, Dict

def preprocess_atis_data(data: List[Dict]) -> List[Dict]:
    """
    Preprocess ATIS data for training
    
    Args:
        data: Raw ATIS data
        
    Returns:
        Preprocessed data ready for training
    """
    processed_data = []
    
    for item in data:
        # Clean text
        text = re.sub(r'[^\w\s]', '', item['text'].lower())
        
        # Extract intent and slots
        intent = item.get('intent', 'unknown')
        slots = item.get('slots', {})
        
        processed_item = {
            'text': text,
            'intent': intent,
            'slots': slots,
            'entities': extract_entities(slots)
        }
        
        processed_data.append(processed_item)
    
    return processed_data

def extract_entities(slots: Dict) -> List[Dict]:
    """
    Extract entities from slot annotations
    
    Args:
        slots: Slot annotations from ATIS
        
    Returns:
        List of entity dictionaries
    """
    entities = []
    
    for slot_type, value in slots.items():
        if value:
            entities.append({
                'type': slot_type,
                'value': value,
                'start': 0,  # Simplified - would need actual position
                'end': len(value)
            })
    
    return entities
```

### 3. Create Custom Travel Dataset

```python
def create_travel_dataset():
    """
    Create a custom travel dataset combining ATIS with additional travel data
    """
    # Base ATIS data
    atis_data = load_atis_dataset()
    
    # Additional travel intents
    travel_intents = [
        "book_hotel",
        "check_booking",
        "cancel_booking",
        "modify_booking",
        "weather_info",
        "travel_updates",
        "baggage_info",
        "customs_info"
    ]
    
    # Create synthetic data for additional intents
    synthetic_data = create_synthetic_travel_data(travel_intents)
    
    # Combine datasets
    combined_data = atis_data + synthetic_data
    
    return combined_data

def create_synthetic_travel_data(intents: List[str]) -> List[Dict]:
    """
    Create synthetic travel data for additional intents
    
    Args:
        intents: List of intents to generate data for
        
    Returns:
        List of synthetic training examples
    """
    synthetic_data = []
    
    # Hotel booking examples
    hotel_examples = [
        "I need a hotel in New York for next week",
        "Book me a room at the Marriott in Los Angeles",
        "I want to stay at a 5-star hotel in Paris",
        "Find me a hotel near the airport in Chicago"
    ]
    
    for example in hotel_examples:
        synthetic_data.append({
            'text': example,
            'intent': 'book_hotel',
            'slots': extract_hotel_slots(example),
            'entities': extract_hotel_entities(example)
        })
    
    # Add more examples for other intents...
    
    return synthetic_data
```

## Model Architecture Selection

### 1. Intent Classification Models

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch import nn
import torch

class TravelIntentClassifier(nn.Module):
    """
    Custom intent classifier for travel domain
    """
    
    def __init__(self, model_name: str, num_intents: int):
        super().__init__()
        
        # Load pre-trained model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name, 
            num_labels=num_intents
        )
        
        # Add custom layers for travel domain
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.model.config.hidden_size, num_intents)
    
    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits
```

### 2. Named Entity Recognition (NER) Models

```python
from transformers import AutoTokenizer, AutoModelForTokenClassification

class TravelNERModel(nn.Module):
    """
    Custom NER model for travel entities
    """
    
    def __init__(self, model_name: str, num_entities: int):
        super().__init__()
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForTokenClassification.from_pretrained(
            model_name,
            num_labels=num_entities
        )
    
    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.logits
```

## Training Process

### 1. Training Configuration

```python
from transformers import TrainingArguments, Trainer
import torch

def setup_training_config():
    """
    Setup training configuration for travel models
    """
    training_args = TrainingArguments(
        output_dir="./travel_models",
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=100,
        evaluation_strategy="steps",
        eval_steps=500,
        save_steps=1000,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy"
    )
    
    return training_args
```

### 2. Data Loading and Tokenization

```python
from torch.utils.data import Dataset, DataLoader

class TravelDataset(Dataset):
    """
    Custom dataset for travel data
    """
    
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def prepare_data_loaders(train_data, val_data, tokenizer):
    """
    Prepare data loaders for training
    """
    train_dataset = TravelDataset(
        texts=[item['text'] for item in train_data],
        labels=[item['intent_id'] for item in train_data],
        tokenizer=tokenizer
    )
    
    val_dataset = TravelDataset(
        texts=[item['text'] for item in val_data],
        labels=[item['intent_id'] for item in val_data],
        tokenizer=tokenizer
    )
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    
    return train_loader, val_loader
```

### 3. Training Loop

```python
def train_travel_model(model, train_loader, val_loader, training_args):
    """
    Train the travel intent classification model
    """
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_loader.dataset,
        eval_dataset=val_loader.dataset,
        compute_metrics=compute_metrics
    )
    
    # Start training
    trainer.train()
    
    # Save the model
    trainer.save_model("./travel_models/intent_classifier")
    
    return trainer

def compute_metrics(eval_pred):
    """
    Compute evaluation metrics
    """
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    
    return {
        'accuracy': accuracy_score(labels, predictions),
        'precision': precision_score(labels, predictions, average='weighted'),
        'recall': recall_score(labels, predictions, average='weighted'),
        'f1': f1_score(labels, predictions, average='weighted')
    }
```

## Fine-tuning Strategies

### 1. Domain Adaptation

```python
def domain_adaptation_training(base_model, travel_data):
    """
    Fine-tune a pre-trained model on travel domain data
    """
    # Freeze base layers
    for param in base_model.embeddings.parameters():
        param.requires_grad = False
    
    # Unfreeze last few layers
    for param in base_model.encoder.layer[-2:].parameters():
        param.requires_grad = True
    
    # Train with lower learning rate
    training_args = TrainingArguments(
        learning_rate=1e-5,  # Lower learning rate for fine-tuning
        num_train_epochs=2,
        # ... other args
    )
    
    return train_travel_model(base_model, train_loader, val_loader, training_args)
```

### 2. Multi-task Learning

```python
class MultiTaskTravelModel(nn.Module):
    """
    Multi-task model for intent classification and NER
    """
    
    def __init__(self, model_name: str, num_intents: int, num_entities: int):
        super().__init__()
        
        self.shared_encoder = AutoModel.from_pretrained(model_name)
        
        # Intent classification head
        self.intent_classifier = nn.Linear(
            self.shared_encoder.config.hidden_size, 
            num_intents
        )
        
        # NER head
        self.ner_classifier = nn.Linear(
            self.shared_encoder.config.hidden_size, 
            num_entities
        )
    
    def forward(self, input_ids, attention_mask, task_type='intent'):
        outputs = self.shared_encoder(input_ids=input_ids, attention_mask=attention_mask)
        
        if task_type == 'intent':
            pooled_output = outputs.pooler_output
            return self.intent_classifier(pooled_output)
        else:
            sequence_output = outputs.last_hidden_state
            return self.ner_classifier(sequence_output)
```

## Evaluation and Testing

### 1. Model Evaluation

```python
def evaluate_travel_model(model, test_data, tokenizer):
    """
    Evaluate the trained travel model
    """
    model.eval()
    predictions = []
    true_labels = []
    
    with torch.no_grad():
        for item in test_data:
            inputs = tokenizer(
                item['text'],
                return_tensors='pt',
                truncation=True,
                padding=True
            )
            
            outputs = model(**inputs)
            pred = torch.argmax(outputs.logits, dim=1)
            predictions.append(pred.item())
            true_labels.append(item['intent_id'])
    
    # Calculate metrics
    accuracy = accuracy_score(true_labels, predictions)
    precision = precision_score(true_labels, predictions, average='weighted')
    recall = recall_score(true_labels, predictions, average='weighted')
    f1 = f1_score(true_labels, predictions, average='weighted')
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }
```

### 2. Error Analysis

```python
def analyze_errors(model, test_data, tokenizer, intent_map):
    """
    Analyze model errors to identify improvement areas
    """
    errors = []
    
    for item in test_data:
        prediction = predict_intent(model, item['text'], tokenizer)
        true_intent = intent_map[item['intent_id']]
        
        if prediction != true_intent:
            errors.append({
                'text': item['text'],
                'predicted': prediction,
                'true': true_intent
            })
    
    # Group errors by type
    error_types = {}
    for error in errors:
        error_type = f"{error['true']} -> {error['predicted']}"
        if error_type not in error_types:
            error_types[error_type] = []
        error_types[error_type].append(error['text'])
    
    return error_types
```

## Integration with Travel Bot

### 1. Model Loading and Inference

```python
class TravelAIEngine:
    """
    Enhanced AI engine with custom trained models
    """
    
    def __init__(self, model_path: str):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.intent_map = self.load_intent_map()
    
    def predict_intent(self, text: str) -> Tuple[str, float]:
        """
        Predict intent using custom trained model
        """
        inputs = self.tokenizer(
            text,
            return_tensors='pt',
            truncation=True,
            padding=True
        )
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            probabilities = torch.softmax(outputs.logits, dim=1)
            predicted_id = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][predicted_id].item()
        
        intent = self.intent_map[predicted_id]
        return intent, confidence
    
    def extract_entities(self, text: str) -> List[Dict]:
        """
        Extract entities using custom NER model
        """
        # Implementation for entity extraction
        pass
```

### 2. Model Updates

```python
def update_travel_model(new_data: List[Dict], model_path: str):
    """
    Update the travel model with new data
    """
    # Load existing model
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # Prepare new training data
    new_dataset = prepare_dataset(new_data, tokenizer)
    
    # Fine-tune on new data
    training_args = TrainingArguments(
        output_dir="./updated_models",
        num_train_epochs=1,  # Fewer epochs for updates
        learning_rate=5e-6,  # Very low learning rate
        # ... other args
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=new_dataset
    )
    
    trainer.train()
    trainer.save_model("./updated_models/travel_model")
```

## Additional Datasets

### 1. MultiWOZ Dataset

```python
# Multi-domain dialogue dataset
multiwoz_dataset = load_dataset("multiwoz_v2.1")

# Extract travel-related conversations
travel_conversations = []
for conversation in multiwoz_dataset['train']:
    if 'restaurant' in conversation['domains'] or 'hotel' in conversation['domains']:
        travel_conversations.append(conversation)
```

### 2. SNIPS Dataset

```python
# SNIPS NLU benchmark
snips_dataset = load_dataset("snips_built_in_intents")

# Travel-related intents
travel_intents = [
    "BookRestaurant",
    "GetWeather",
    "SearchCreativeWork"
]
```

### 3. Custom Travel Dataset Creation

```python
def create_custom_travel_dataset():
    """
    Create a comprehensive travel dataset
    """
    dataset = {
        'intents': [
            'book_flight',
            'book_hotel',
            'check_booking',
            'cancel_booking',
            'modify_booking',
            'weather_info',
            'travel_updates',
            'baggage_info',
            'customs_info',
            'meal_preference',
            'seat_preference',
            'special_assistance'
        ],
        'entities': [
            'departure_city',
            'arrival_city',
            'date',
            'time',
            'airline',
            'flight_number',
            'hotel_name',
            'room_type',
            'number_of_passengers',
            'price_range'
        ],
        'examples': []
    }
    
    # Add training examples for each intent
    for intent in dataset['intents']:
        examples = generate_examples_for_intent(intent)
        dataset['examples'].extend(examples)
    
    return dataset
```

## Best Practices

### 1. Data Quality

- Ensure diverse training data covering different travel scenarios
- Include edge cases and ambiguous queries
- Balance intent distribution in training data
- Regular data validation and cleaning

### 2. Model Performance

- Use appropriate model size for your use case
- Implement caching for frequently used predictions
- Monitor model performance in production
- Regular model retraining with new data

### 3. Deployment

- Version control for models
- A/B testing for model updates
- Fallback mechanisms for model failures
- Monitoring and alerting for model performance

## Conclusion

Training custom transformer models on travel-specific datasets like ATIS can significantly improve the travel bot's natural language understanding capabilities. The key is to:

1. Start with pre-trained models and fine-tune on travel data
2. Use domain-specific datasets and create synthetic data when needed
3. Implement proper evaluation and monitoring
4. Regularly update models with new data and user feedback

This approach will result in a more accurate and contextually aware travel assistant that can better understand user queries and provide more relevant responses. 