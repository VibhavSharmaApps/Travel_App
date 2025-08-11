# ☁️ Cloud Training Setup Guide

This guide shows you how to train your ATIS model in the cloud when your local machine doesn't have enough resources.

## 🚀 **Option 1: Google Colab (FREE - Recommended)**

### **Step 1: Setup Google Colab**
1. Go to [Google Colab](https://colab.research.google.com/)
2. Sign in with your Google account
3. Create a new notebook

### **Step 2: Enable GPU**
1. Go to **Runtime** → **Change runtime type**
2. Set **Hardware accelerator** to **GPU**
3. Click **Save**

### **Step 3: Upload and Run Training**
1. Upload the `training/colab_training.ipynb` notebook
2. Run all cells sequentially
3. Download your trained model

**Time:** 15-30 minutes
**Cost:** FREE

---

## 🏆 **Option 2: Kaggle Notebooks (FREE)**

### **Step 1: Setup Kaggle**
1. Go to [Kaggle](https://www.kaggle.com/)
2. Create an account
3. Go to **Code** → **New Notebook**

### **Step 2: Enable GPU**
1. Click **Settings** (gear icon)
2. Set **Accelerator** to **GPU**
3. Click **Save**

### **Step 3: Run Training**
```python
# Install dependencies
!pip install transformers datasets torch

# Clone your repository or upload files
!git clone https://github.com/your-repo/travel-bot.git

# Run training
!cd travel-bot && python training/train_atis.py
```

**Time:** 15-30 minutes
**Cost:** FREE

---

## 💰 **Option 3: AWS SageMaker (PAID)**

### **Step 1: Setup AWS Account**
1. Create AWS account
2. Set up billing
3. Go to SageMaker console

### **Step 2: Create Notebook Instance**
1. Click **Create notebook instance**
2. Choose **ml.p3.2xlarge** (1x V100 GPU)
3. Set up IAM role and security

### **Step 3: Upload and Train**
```bash
# Upload your training files
# Run training script
python training/train_atis.py
```

**Time:** 5-15 minutes
**Cost:** ~$3-5 total

---

## 🔧 **Option 4: Google Cloud AI Platform (PAID)**

### **Step 1: Setup GCP**
1. Create Google Cloud account
2. Enable billing
3. Go to AI Platform

### **Step 2: Create Notebook Instance**
1. Click **New notebook**
2. Choose **Tesla T4** or **V100** GPU
3. Set up environment

### **Step 3: Run Training**
```bash
# Upload training files
# Run training
python training/train_atis.py
```

**Time:** 5-15 minutes
**Cost:** ~$2-8 total

---

## 📊 **Cost Comparison**

| Platform | GPU | Cost | Time | Best For |
|----------|-----|------|------|----------|
| **Google Colab** | Tesla T4 | FREE | 15-30 min | Beginners, experiments |
| **Kaggle** | Tesla P100 | FREE | 15-30 min | Data science projects |
| **AWS SageMaker** | V100 | $3-5 | 5-15 min | Production, enterprise |
| **Google Cloud** | T4/V100 | $2-8 | 5-15 min | Google ecosystem |

---

## 🎯 **Recommended Workflow**

### **For Beginners (FREE)**
1. **Start with Google Colab**
2. Use the provided notebook
3. Train your model
4. Download and test locally

### **For Production (PAID)**
1. **Use AWS SageMaker** or **Google Cloud**
2. Set up proper CI/CD pipeline
3. Train multiple models
4. Deploy to production

---

## 📁 **File Structure for Cloud**

```
your-project/
├── training/
│   ├── colab_training.ipynb    # Google Colab notebook
│   ├── train_atis.py          # Training script
│   ├── test_trained_model.py  # Testing script
│   └── requirements_training.txt
├── data/
│   ├── flights.xlsx
│   └── hotels.xlsx
└── src/
    └── [your bot files]
```

---

## 🔄 **Integration Steps**

### **After Cloud Training:**

1. **Download your model** from the cloud platform
2. **Extract the model files** to your local project
3. **Update your AI engine** to use the trained model
4. **Test the integration** with your travel bot

### **Example Integration:**
```python
# In src/ai_engine.py
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Load your trained model
model_path = "./training/models/atis_intent_classifier"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

# Use for predictions
def classify_intent(text):
    inputs = tokenizer(text, return_tensors='pt')
    outputs = model(**inputs)
    return outputs.logits
```

---

## 🛠️ **Troubleshooting Cloud Issues**

### **Common Problems:**

1. **"Out of Memory" Error**
   - Reduce batch size
   - Use smaller model
   - Enable gradient accumulation

2. **Session Timeout**
   - Save checkpoints frequently
   - Use persistent storage
   - Resume from checkpoint

3. **Slow Training**
   - Check GPU availability
   - Optimize data loading
   - Use mixed precision training

### **Getting Help:**
- Check platform documentation
- Use platform support forums
- Monitor resource usage

---

## 🎉 **Success Checklist**

After cloud training, verify:

- ✅ Model files downloaded successfully
- ✅ Model loads without errors
- ✅ Predictions work correctly
- ✅ Integration with bot successful
- ✅ Performance meets expectations

**You're ready to use your cloud-trained model! 🚀** 