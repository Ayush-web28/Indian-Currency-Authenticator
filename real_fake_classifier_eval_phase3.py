"""
PHASE 3: Evaluate Real/Fake Classifier (CORRECT Labels)
Test on held-out set with confusion matrix and metrics
"""

import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import os
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

print("="*70)
print("PHASE 3: Evaluate Real/Fake Classifier")
print("="*70)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Paths
base_path = '/content/drive/MyDrive/AI_Dataset/real_fake_classifier_data'
real_path = os.path.join(base_path, 'real')
fake_path = os.path.join(base_path, 'fake')

# Transforms
test_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
])

class RealFakeDataset(Dataset):
    def __init__(self, file_paths, labels, transforms=None):
        self.file_paths = file_paths
        self.labels = labels
        self.transforms = transforms

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        try:
            img = Image.open(self.file_paths[idx]).convert('RGB')
            if self.transforms:
                img = self.transforms(img)
            return img, self.labels[idx]
        except:
            return torch.zeros(3, 224, 224), self.labels[idx]

# Load test data
print("\n📂 Loading test data...")
file_paths = []
labels = []

# Real (label = 1)
for file in os.listdir(real_path):
    if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
        file_paths.append(os.path.join(real_path, file))
        labels.append(1)

# Fake (label = 0)
for file in os.listdir(fake_path):
    if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
        file_paths.append(os.path.join(fake_path, file))
        labels.append(0)

file_paths = np.array(file_paths)
labels = np.array(labels)

# Create test set
indices = np.arange(len(file_paths))
train_idx, temp_idx = train_test_split(indices, test_size=0.3, random_state=42, stratify=labels)
val_idx, test_idx = train_test_split(temp_idx, test_size=0.5, random_state=42, stratify=labels[temp_idx])

test_paths, test_labels = file_paths[test_idx], labels[test_idx]
test_dataset = RealFakeDataset(test_paths, test_labels, test_transforms)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=2)

print(f"✓ Test set: {len(test_paths)} images")
print(f"  Real (label=1): {np.sum(test_labels == 1)}")
print(f"  Fake (label=0): {np.sum(test_labels == 0)}")

# Load model
print("\n🔧 Loading best model...")
model = models.resnet50(weights='DEFAULT')
num_features = model.fc.in_features
model.fc = nn.Sequential(
    nn.Dropout(0.5),
    nn.Linear(num_features, 256),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(256, 1),
    nn.Sigmoid()
)

best_model_path = os.path.join(base_path, 'best_real_fake_model.pth')
if os.path.exists(best_model_path):
    model.load_state_dict(torch.load(best_model_path))
    print(f"✓ Best model loaded")
else:
    print(f"⚠️  Best model not found at {best_model_path}")
    print(f"   Using final model instead")
    final_model_path = os.path.join(base_path, 'final_real_fake_model.pth')
    model.load_state_dict(torch.load(final_model_path))

model = model.to(device)
model.eval()

# Evaluate
print("\n📊 Evaluating on test set...")
all_preds = []
all_probs = []
test_correct = 0

with torch.no_grad():
    for images, labels_batch in test_loader:
        images = images.to(device)
        outputs = model(images)
        probs = outputs.cpu().numpy().flatten()
        preds = (probs > 0.5).astype(int)  # STANDARD logic

        all_probs.extend(probs)
        all_preds.extend(preds)
        test_correct += (preds == labels_batch.numpy()).sum()

test_acc = test_correct / len(test_dataset)
all_preds = np.array(all_preds)
all_probs = np.array(all_probs)

print(f"\n✅ Test Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")
print(f"✅ AUC-ROC Score: {roc_auc_score(test_labels, all_probs):.4f}")

# Confusion Matrix
cm = confusion_matrix(test_labels, all_preds)
print(f"\n📈 Confusion Matrix:")
print(f"   True Neg (Fake): {cm[0,0]} | False Pos (Real): {cm[0,1]}")
print(f"   False Neg (Fake): {cm[1,0]} | True Pos (Real): {cm[1,1]}")

# Classification Report
print("\n📋 Classification Report:")
print(classification_report(test_labels, all_preds, target_names=['Fake', 'Real']))

# Visualizations
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Confusion Matrix Heatmap
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0], cbar=False,
            xticklabels=['Fake', 'Real'],
            yticklabels=['Fake', 'Real'])
axes[0].set_title('Confusion Matrix', fontsize=14, fontweight='bold')
axes[0].set_ylabel('True Label')
axes[0].set_xlabel('Predicted Label')

# Probability Distribution
axes[1].hist(all_probs[test_labels == 0], bins=30, alpha=0.6, label='Fake', color='red')
axes[1].hist(all_probs[test_labels == 1], bins=30, alpha=0.6, label='Real', color='green')
axes[1].axvline(0.5, color='black', linestyle='--', linewidth=2, label='Threshold (0.5)')
axes[1].set_xlabel('Prediction Probability')
axes[1].set_ylabel('Frequency')
axes[1].set_title('Prediction Confidence Distribution', fontsize=14, fontweight='bold')
axes[1].legend()

plt.tight_layout()
eval_plot_path = os.path.join(base_path, 'evaluation_metrics.png')
plt.savefig(eval_plot_path, dpi=150, bbox_inches='tight')
print(f"\n✓ Evaluation plots saved to {eval_plot_path}")

print("\n" + "="*70)
print("✓ PHASE 3 COMPLETE!")
print("="*70)
print(f"\n✅ Model ready for production with STANDARD logic:")
print(f"   is_real = stage2_prob > 0.5")
