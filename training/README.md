# Training Guide: ATIS and Travel Domain Models

This directory contains all the scripts and resources needed to train custom transformer models on ATIS and other travel datasets.

## 📁 Directory Structure

```
training/
├── README.md                 # This file
├── config.py                 # Training configuration
├── train_atis.py            # ATIS training script
├── test_trained_model.py    # Model testing script
├── models/                  # Trained models (created after training)
├── data/                    # Downloaded datasets
├── logs/                    # Training logs
├── checkpoints/             # Model checkpoints
└── results/                 # Training results
```

## 🚀 Quick Start

### Step 1: Setup Training Environment

```bash
# Run the setup script to install dependencies and download datasets
python scripts/setup_training.py
```

This will:
- Install all required Python packages
- Download ATIS, MultiWOZ, and SNIPS datasets
- Download pre-trained models (BERT, RoBERTa, etc.)
- Create necessary directories
- Setup training configuration

### Step 2: Train ATIS Model

```bash
# Train intent classification model on ATIS dataset
python training/train_atis.py
```

This will:
- Load and preprocess ATIS dataset
- Train a BERT-based intent classifier
- Save the trained model
- Generate evaluation results

### Step 3: Test Trained Model

```bash
# Test the trained model on sample queries
python training/test_trained_model.py
```

This will:
- Load the trained model
- Test on sample queries
- Provide interactive testing mode

## 📊 ATIS Dataset Overview

The ATIS (Airline Travel Information System) dataset contains:

- **5,000+ utterances** for airline travel queries
- **18 different intents** (flight, ground_service, meal, etc.)
- **127 slot types** (departure_city, arrival_city, date, etc.)
- **Natural language queries** about airline travel

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

## 🔧 Training Configuration

The training configuration is defined in `training/config.py`:

```python
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
```

## 🎯 Training Process

### 1. Data Preprocessing

The training script automatically:

- Downloads ATIS dataset from Hugging Face
- Extracts intents and creates mapping
- Splits data into train/validation/test sets
- Tokenizes text using the selected model's tokenizer

### 2. Model Training

The training process:

- Loads pre-trained BERT model
- Adds classification head for intent prediction
- Trains using Hugging Face Trainer
- Saves best model based on validation F1 score
- Generates training logs and metrics

### 3. Model Evaluation

After training:

- Evaluates model on test set
- Computes accuracy, precision, recall, and F1 score
- Saves evaluation results
- Provides detailed performance metrics

## 📈 Expected Results

With the default configuration, you should expect:

- **Accuracy**: 95%+ on ATIS test set
- **F1 Score**: 0.95+ for intent classification
- **Training Time**: 10-30 minutes (depending on hardware)
- **Model Size**: ~110MB (BERT-base)

## 🖥️ Hardware Requirements

### Minimum Requirements
- **RAM**: 8GB
- **Storage**: 5GB free space
- **GPU**: Optional (CPU training works but is slower)

### Recommended Requirements
- **RAM**: 16GB+
- **Storage**: 10GB free space
- **GPU**: NVIDIA GPU with 4GB+ VRAM
- **CUDA**: 11.8+ (for GPU acceleration)

## 🔍 Monitoring Training

### TensorBoard Logs

```bash
# Start TensorBoard to monitor training
tensorboard --logdir training/logs

# Open http://localhost:6006 in your browser
```

### Training Logs

Training logs are saved to `training/logs/atis_training.log` and include:

- Training progress
- Validation metrics
- Model checkpoints
- Error messages

## 🧪 Testing Your Model

### Automated Testing

```bash
python training/test_trained_model.py
```

This will test the model on predefined sample queries.

### Interactive Testing

The testing script also provides an interactive mode where you can:

- Enter custom queries
- See intent predictions
- View confidence scores
- Test edge cases

### Sample Test Queries

```python
test_queries = [
    "I want to fly from Boston to New York",
    "What's the weather like in Chicago?",
    "Book me a hotel in Los Angeles",
    "What time does my flight depart?",
    "I need to cancel my reservation"
]
```

## 🔄 Model Integration

### Using Trained Model in Travel Bot

To integrate the trained model with the travel bot:

1. **Copy trained model** to the bot's model directory
2. **Update AI engine** to use the trained model
3. **Test integration** with the bot

```python
# Example integration
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Load trained model
model_path = "./training/models/atis_intent_classifier"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

# Use for prediction
def predict_intent(text):
    inputs = tokenizer(text, return_tensors='pt')
    outputs = model(**inputs)
    return outputs.logits
```

## 🛠️ Troubleshooting

### Common Issues

1. **Out of Memory Error**
   - Reduce batch size in config
   - Use smaller model (distilbert-base-uncased)
   - Enable gradient accumulation

2. **Slow Training**
   - Use GPU if available
   - Reduce max_length
   - Use smaller model

3. **Poor Performance**
   - Increase training epochs
   - Adjust learning rate
   - Add more training data

4. **Model Not Found**
   - Run setup script first
   - Check internet connection
   - Verify model path

### Getting Help

- Check training logs in `training/logs/`
- Verify dataset download in `training/data/`
- Ensure all dependencies are installed
- Check GPU availability with `nvidia-smi`

## 📚 Additional Resources

### Datasets
- [ATIS Dataset](https://huggingface.co/datasets/atis_intents)
- [MultiWOZ Dataset](https://huggingface.co/datasets/multiwoz_v2.1)
- [SNIPS Dataset](https://huggingface.co/datasets/snips_built_in_intents)

### Models
- [BERT Documentation](https://huggingface.co/bert-base-uncased)
- [Transformers Library](https://huggingface.co/docs/transformers/)
- [Training Examples](https://huggingface.co/docs/transformers/training)

### Papers
- [ATIS Paper](https://www.aclweb.org/anthology/H90-1021/)
- [BERT Paper](https://arxiv.org/abs/1810.04805)
- [Intent Classification Survey](https://arxiv.org/abs/2004.05000)

## 🎉 Next Steps

After training your ATIS model:

1. **Test thoroughly** with various queries
2. **Integrate with travel bot** for real-world testing
3. **Fine-tune** on your specific travel domain
4. **Collect feedback** and improve the model
5. **Deploy** in production environment

Happy training! 🚀 