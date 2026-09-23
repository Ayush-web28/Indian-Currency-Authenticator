"""
PHASE 2: RETRAIN Real/Fake Classifier with CORRECT Labels
- Real currency images → Label = 1 (REAL)
- Fake currency images → Label = 0 (FAKE)
- Use standard logic: is_real = stage2_prob > 0.5
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import os
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

print("="*70)
print("PHASE 2: Retrain Real/Fake Classifier (CORRECT Labels)")
print("="*70)

# Device setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\n📱 Device: {device}")

# Paths - Update these with your actual paths
base_path = '/content/drive/MyDrive/AI_Dataset/real_fake_classifier_data'
real_path = os.path.join(base_path, 'real')
fake_path = os.path.join(base_path, 'fake')

# Create directories if needed
os.makedirs(real_path, exist_ok=True)
os.makedirs(fake_path, exist_ok=True)

print(f"\n📂 Data paths:")
print(f"   Real: {real_path}")
print(f"   Fake: {fake_path}")

# Data transforms
train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(0.5),
    transforms.RandomVerticalFlip(0.5),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
])

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

# Load data
print("\n📂 Loading data with CORRECT labels...")
file_paths = []
labels = []

# REAL currency images → label = 1
real_count = 0
if os.path.exists(real_path):
    for file in os.listdir(real_path):
        if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            file_paths.append(os.path.join(real_path, file))
            labels.append(1)  # REAL = 1
            real_count += 1

# FAKE currency images → label = 0
fake_count = 0
if os.path.exists(fake_path):
    for file in os.listdir(fake_path):
        if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            file_paths.append(os.path.join(fake_path, file))
            labels.append(0)  # FAKE = 0
            fake_count += 1

file_paths = np.array(file_paths)
labels = np.array(labels)

print(f"\n✓ Real (label=1): {real_count} images")
print(f"✓ Fake (label=0): {fake_count} images")
print(f"✓ Total: {len(labels)} images")

if len(labels) == 0:
    print("\n⚠️  No images found! Please add images to:")
    print(f"   Real: {real_path}")
    print(f"   Fake: {fake_path}")
    exit()

# Split data (70% train, 15% val, 15% test)
indices = np.arange(len(file_paths))
train_idx, temp_idx = train_test_split(
    indices, test_size=0.3, random_state=42, stratify=labels
)
val_idx, test_idx = train_test_split(
    temp_idx, test_size=0.5, random_state=42, stratify=labels[temp_idx]
)

train_paths, train_labels = file_paths[train_idx], labels[train_idx]
val_paths, val_labels = file_paths[val_idx], labels[val_idx]
test_paths, test_labels = file_paths[test_idx], labels[test_idx]

print(f"\n📊 Data split:")
print(f"   Train: {len(train_paths)}")
print(f"   Val: {len(val_paths)}")
print(f"   Test: {len(test_paths)}")

# Create datasets & dataloaders
train_dataset = RealFakeDataset(train_paths, train_labels, train_transforms)
val_dataset = RealFakeDataset(val_paths, val_labels, test_transforms)
test_dataset = RealFakeDataset(test_paths, test_labels, test_transforms)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=2)

print("✓ DataLoaders created")

# Model setup
print("\n🔧 Setting up model...")
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

model = model.to(device)
print("✓ ResNet50 loaded with custom classification head")

# Class weights for imbalance handling
class_counts = np.bincount(labels)
class_weights = torch.tensor(
    len(labels) / (2.0 * class_counts), dtype=torch.float32
).to(device)
print(f"✓ Class weights: Real={class_weights[1]:.4f}, Fake={class_weights[0]:.4f}")

# Loss & Optimizer
criterion = nn.BCEWithLogitsLoss(pos_weight=class_weights[1] / class_weights[0])
optimizer = optim.Adam(model.parameters(), lr=3e-4, weight_decay=1e-4)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

print("✓ Optimizer: Adam (lr=3e-4, weight_decay=1e-4)")
print("✓ Scheduler: ReduceLROnPlateau")

# Training
print("\n🚀 Starting training...")
num_epochs = 50
best_val_loss = float('inf')
patience = 5
patience_counter = 0
train_losses, val_losses = [], []
train_accs, val_accs = [], []

for epoch in range(num_epochs):
    # Train
    model.train()
    train_loss, train_correct = 0.0, 0

    for images, labels_batch in train_loader:
        images = images.to(device)
        labels_batch = labels_batch.float().unsqueeze(1).to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels_batch)
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * images.size(0)
        train_correct += ((outputs > 0.5).float() == labels_batch).sum().item()

    train_loss /= len(train_dataset)
    train_acc = train_correct / len(train_dataset)
    train_losses.append(train_loss)
    train_accs.append(train_acc)

    # Validate
    model.eval()
    val_loss, val_correct = 0.0, 0

    with torch.no_grad():
        for images, labels_batch in val_loader:
            images = images.to(device)
            labels_batch = labels_batch.float().unsqueeze(1).to(device)

            outputs = model(images)
            loss = criterion(outputs, labels_batch)
            val_loss += loss.item() * images.size(0)
            val_correct += ((outputs > 0.5).float() == labels_batch).sum().item()

    val_loss /= len(val_dataset)
    val_acc = val_correct / len(val_dataset)
    val_losses.append(val_loss)
    val_accs.append(val_acc)

    print(f"Epoch [{epoch+1:2d}/{num_epochs}] | "
          f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
          f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

    scheduler.step(val_loss)

    # Save best model
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        best_model_path = os.path.join(base_path, 'best_real_fake_model.pth')
        torch.save(model.state_dict(), best_model_path)
        print(f"  ✓ Best model saved (Val Loss: {val_loss:.4f})")
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"\n✓ Early stopping at epoch {epoch+1}")
            break

# Save final model
final_model_path = os.path.join(base_path, 'final_real_fake_model.pth')
torch.save(model.state_dict(), final_model_path)
print(f"\n✓ Final model saved to {final_model_path}")

# Plot training history
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(train_losses, label='Train Loss', marker='o')
axes[0].plot(val_losses, label='Val Loss', marker='s')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].set_title('Training & Validation Loss')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(train_accs, label='Train Accuracy', marker='o')
axes[1].plot(val_accs, label='Val Accuracy', marker='s')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Accuracy')
axes[1].set_title('Training & Validation Accuracy')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(base_path, 'training_history.png'), dpi=150, bbox_inches='tight')
print("✓ Training history plots saved")

print("\n" + "="*70)
print("✓ PHASE 2 COMPLETE!")
print("="*70)
print(f"\n📊 Final Results:")
print(f"   Best Model: {best_model_path}")
print(f"   Best Val Loss: {best_val_loss:.4f}")
print(f"   Final Val Acc: {val_accs[-1]:.4f}")
print(f"\n✓ Now use this model with STANDARD logic:")
print(f"   is_real = stage2_prob > 0.5")
