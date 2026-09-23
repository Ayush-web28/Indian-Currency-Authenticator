# 💵 Indian Fake Currency Detection Pipeline

**AI-powered multi-stage fake Indian currency detection** using ResNet50 deep learning models with **3-stage sequential classification pipeline**.

A robust system that first detects whether an image contains currency, then classifies it as real or fake with high accuracy.

---

## 🎯 Pipeline Overview

```
Input Image
    ↓
Stage 1: Binary Classifier (Currency vs Non-Currency)
    ├─ Is this CURRENCY? (91.96% accuracy)
    │  ├─ YES → Continue to Stage 2
    │  └─ NO → Output: NON-CURRENCY
    ↓
Stage 2: Real/Fake Classifier (Real vs Fake Currency)
    ├─ Is it REAL CURRENCY? (Standard logic)
    │  ├─ YES → Output: ✓ REAL CURRENCY (with metrics)
    │  └─ NO → Output: ✗ FAKE CURRENCY (with metrics)
```

---

## 📊 Model Performance

### **Stage 1: Binary Classifier (Currency Detection)**
| Metric | Score |
|--------|-------|
| **Test Accuracy** | 91.96% |
| **Precision** | High |
| **Recall** | High |
| **AUC-ROC** | 0.95+ |

**Dataset:** 7,442 existing currency images + ~5,700 downloaded non-currency images (people, furniture, food, animals, vehicles, books, scenes)

### **Stage 2: Real/Fake Classifier**
| Metric | Score |
|--------|-------|
| **Test Accuracy** | High |
| **Architecture** | ResNet50 (custom head) |
| **Training Logic** | Standard (prob > 0.5 = REAL) |

**Dataset:** Indian Currency Real vs Fake Notes (₹10, ₹20, ₹50, ₹100, ₹200, ₹500, ₹2000)

---

## 🧠 Model Architecture (Both Stages)

**Framework:** PyTorch  
**Base Model:** ResNet50 (pretrained on ImageNet)  
**Input Size:** 224×224 pixels  
**Output:** Binary classification (Sigmoid activation)

**Custom Classification Head:**
```
Dropout(0.5) 
  ↓
Linear(2048 → 256) 
  ↓
ReLU 
  ↓
Dropout(0.3) 
  ↓
Linear(256 → 1) 
  ↓
Sigmoid (output: 0-1)
```

---

## 🏋️ Training Configuration

**Both Stages Use:**
- **Batch Size:** 32
- **Learning Rate:** 3e-4
- **Optimizer:** Adam (weight_decay=1e-4)
- **Loss Function:** Weighted Binary Crossentropy (handles class imbalance)
- **Scheduler:** ReduceLROnPlateau (factor=0.5, patience=3)
- **Early Stopping:** Patience=5 epochs
- **Data Augmentation:**
  - RandomHorizontalFlip (50%)
  - RandomVerticalFlip (50%)
  - RandomRotation (15°)
  - ColorJitter (brightness=0.2)
- **Train/Val/Test Split:** 70% / 15% / 15%

**Training Time:** ~60-90 minutes per stage (GPU optimized)

---

## 📁 Project Structure

```
model/
├── best_model.pth                    (Stage 1: Binary Classifier - 92 MB)
└── best_real_fake_model.pth          (Stage 2: Real/Fake Classifier - 92 MB)

Training Scripts:
├── real_fake_classifier_phase1_organize.py   (Data organization)
├── real_fake_classifier_training_phase2.py   (Model training)
├── real_fake_classifier_eval_phase3.py       (Evaluation & metrics)
└── colab_full_pipeline_standard_logic.py     (Complete testing pipeline)
```

---

## 🚀 Usage

### **Option 1: Complete 3-Stage Pipeline (Recommended)**

```python
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load Stage 1: Binary Classifier
binary_model = models.resnet50(weights='DEFAULT')
num_features = binary_model.fc.in_features
binary_model.fc = torch.nn.Sequential(
    torch.nn.Dropout(0.5),
    torch.nn.Linear(num_features, 256),
    torch.nn.ReLU(),
    torch.nn.Dropout(0.3),
    torch.nn.Linear(256, 1),
    torch.nn.Sigmoid()
)
binary_model.load_state_dict(torch.load('model/best_model.pth', map_location=device))
binary_model = binary_model.to(device).eval()

# Load Stage 2: Real/Fake Classifier
real_fake_model = models.resnet50(weights='DEFAULT')
real_fake_model.fc = torch.nn.Sequential(
    torch.nn.Dropout(0.5),
    torch.nn.Linear(num_features, 256),
    torch.nn.ReLU(),
    torch.nn.Dropout(0.3),
    torch.nn.Linear(256, 1),
    torch.nn.Sigmoid()
)
real_fake_model.load_state_dict(torch.load('model/best_real_fake_model.pth', map_location=device))
real_fake_model = real_fake_model.to(device).eval()

# Inference
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
])

image = Image.open('test_image.jpg').convert('RGB')
img_tensor = transform(image).unsqueeze(0).to(device)

# Stage 1: Currency Detection
with torch.no_grad():
    stage1_prob = binary_model(img_tensor).item()

if stage1_prob > 0.5:
    print("✅ Stage 1: CURRENCY DETECTED")
    
    # Stage 2: Real/Fake Classification
    with torch.no_grad():
        stage2_prob = real_fake_model(img_tensor).item()
    
    if stage2_prob > 0.5:
        print(f"✓ Stage 2: REAL CURRENCY ({stage2_prob*100:.2f}% confidence)")
    else:
        print(f"✗ Stage 2: FAKE CURRENCY ({(1-stage2_prob)*100:.2f}% confidence)")
else:
    print("⊘ Stage 1: NON-CURRENCY (skipped Stage 2)")
```

### **Option 2: Colab Training & Testing**

Complete training pipeline available in Colab with:
- Automatic data organization
- Model training with progress monitoring
- Evaluation metrics and visualizations
- Interactive image upload and testing

See training scripts for full implementation.

---

## 🎛️ Requirements

```bash
torch>=2.0.0
torchvision>=0.15.0
Pillow>=10.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
matplotlib>=3.7.0
seaborn>=0.12.0
```

---

## 📊 Output Metrics

When testing an image, the pipeline provides:

**Stage 1 Output:**
- Classification: CURRENCY or NON-CURRENCY
- Confidence: 0-100%
- Raw Probability: 0.0-1.0

**Stage 2 Output (if Currency detected):**
- Classification: REAL or FAKE
- Confidence: 0-100%
- % Real: Probability of being real
- % Fake: Probability of being fake
- Authenticity Grade: A/B/C/D
- Risk Level: 🟢 Low / 🟡 Medium / 🔴 High
- Star Rating: ⭐ (1-5 stars)
- Threshold Distance: Distance from decision boundary (0.5)

---

## 🔍 Key Features

✅ **Two-Stage Pipeline** - First filters non-currency, then classifies real/fake  
✅ **High Accuracy** - Stage 1: 91.96% | Stage 2: Well-trained on denominations  
✅ **Comprehensive Metrics** - Grade, risk level, confidence scoring  
✅ **GPU Optimized** - ~200-300ms per complete inference  
✅ **Production-Ready** - Tested on diverse Indian currency notes  
✅ **Easy to Use** - Simple Python API or Colab notebook  

---

## ⚠️ Limitations

- Designed specifically for Indian currency notes (₹)
- Works best with clear, well-lit images
- Best accuracy with front/back side images
- Should be combined with other authentication methods
- Non-currency filtering helps reduce false positives

---

## 📈 Training from Scratch

To retrain the models:

```bash
# Phase 1: Organize data
python real_fake_classifier_phase1_organize.py

# Phase 2: Train model
python real_fake_classifier_training_phase2.py

# Phase 3: Evaluate model
python real_fake_classifier_eval_phase3.py
```

Or use the provided Colab notebook for interactive training.

---

## 📦 Model Files

| File | Size | Purpose |
|------|------|---------|
| `best_model.pth` | 92 MB | Stage 1: Binary Classifier (Git LFS) |
| `best_real_fake_model.pth` | 92 MB | Stage 2: Real/Fake Classifier (Git LFS) |

**Storage:** Models stored using Git LFS for GitHub compatibility.

---

## 📄 License

This project is provided for educational and research purposes.

---

## 🙏 Acknowledgments

- **Framework:** PyTorch
- **Base Model:** ResNet50 (torchvision)
- **Dataset:** [Kaggle - Indian Currency Real vs Fake Notes](https://www.kaggle.com/datasets/preetrank/indian-currency-real-vs-fake-notes-dataset)
- **Architecture:** Tested on diverse Indian currency denominations

---

## 📧 Citation

```
Indian Fake Currency Detection Pipeline
- Framework: PyTorch
- Architecture: Two-stage ResNet50 classifiers
- Stage 1: Currency Detection (91.96% accuracy)
- Stage 2: Real/Fake Classification
- Dataset: Kaggle + custom non-currency dataset
- Production-Ready: Yes
```

---

**Built for Indian Currency Fraud Detection and Authentication** 💵✓
