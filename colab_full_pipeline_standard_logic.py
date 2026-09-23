"""
3-Stage Currency Detection Pipeline (STANDARD Logic)
After retraining Stage 2 with correct labels
Stage 1: Currency vs Non-Currency (Binary Classification)
Stage 2: Real vs Fake (STANDARD logic - is_real = stage2_prob > 0.5)
"""

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import os
from datetime import datetime
from google.colab import files

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load Binary Classifier (Currency vs Non-Currency)
print("Loading Stage 1: Binary Classifier...")
binary_model = models.resnet50(weights='DEFAULT')
num_features = binary_model.fc.in_features
binary_model.fc = nn.Sequential(
    nn.Dropout(0.5),
    nn.Linear(num_features, 256),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(256, 1),
    nn.Sigmoid()
)
binary_model.load_state_dict(torch.load('/content/drive/MyDrive/AI_Dataset/binary_classifier_data/best_model.pth'))
binary_model = binary_model.to(device)
binary_model.eval()
print("✓ Stage 1 Ready")

# Load Real/Fake Classifier (NEW - with STANDARD logic)
print("Loading Stage 2: Real/Fake Classifier (STANDARD logic)...")
try:
    real_fake_model = models.resnet50(weights='DEFAULT')
    real_fake_model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(num_features, 256),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(256, 1),
        nn.Sigmoid()
    )
    # NEW: Use retrained model with correct labels
    real_fake_model.load_state_dict(torch.load('/content/drive/MyDrive/AI_Dataset/real_fake_classifier_data/best_real_fake_model.pth'))
    real_fake_model = real_fake_model.to(device)
    real_fake_model.eval()
    print("✓ Stage 2 Ready (STANDARD logic)")
except Exception as e:
    print(f"❌ Stage 2 error: {str(e)}")
    real_fake_model = None

# Image transforms
transforms_test = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

print("\n✅ 3-Stage Pipeline Ready (STANDARD Logic)!")
print("   Stage 1: Currency Detection")
print("   Stage 2: Real/Fake Classification (STANDARD logic)")

def test_image(image_path):
    """Complete 3-stage pipeline test with STANDARD logic"""
    img = Image.open(image_path).convert('RGB')
    img_tensor = transforms_test(img).unsqueeze(0).to(device)

    # Stage 1: Binary Classification
    with torch.no_grad():
        stage1_out = binary_model(img_tensor)
        stage1_prob = stage1_out.item()

    is_currency = stage1_prob > 0.5
    stage1_confidence = stage1_prob if is_currency else (1 - stage1_prob)

    # Stage 2: Real/Fake Classification (if Currency)
    stage2_prob = None
    stage2_confidence = None
    is_real = None

    if is_currency and real_fake_model is not None:
        with torch.no_grad():
            stage2_out = real_fake_model(img_tensor)
            stage2_prob = stage2_out.item()

        # STANDARD logic (after retraining with correct labels)
        is_real = stage2_prob > 0.5  # Real = 1, Fake = 0
        stage2_confidence = stage2_prob if is_real else (1 - stage2_prob)

    # Calculate additional metrics
    if stage2_prob is not None:
        percent_real = stage2_prob * 100  # Now this represents actual probability of REAL
        percent_fake = (1 - stage2_prob) * 100  # Probability of FAKE
        threshold_distance = abs(stage2_prob - 0.5) * 100

        # Authenticity Grade (A/B/C/D) - Based on result, not just confidence
        if is_real:
            # Real currency grades (A = best/most authentic)
            if stage2_confidence >= 0.95:
                grade = 'A'  # Definitely authentic
            elif stage2_confidence >= 0.80:
                grade = 'B'  # Probably authentic
            elif stage2_confidence >= 0.60:
                grade = 'C'  # Uncertain authenticity
            else:
                grade = 'D'  # Low confidence in authenticity
        else:
            # Fake currency grades (D = worst/definitely fake)
            if stage2_confidence >= 0.95:
                grade = 'D'  # Definitely fake
            elif stage2_confidence >= 0.80:
                grade = 'C'  # Probably fake
            elif stage2_confidence >= 0.60:
                grade = 'B'  # Uncertain if fake
            else:
                grade = 'A'  # Low confidence in fake detection

        # Risk Level - Based on result and confidence
        if is_real:
            # Real currency = low risk
            if stage2_confidence >= 0.90:
                risk_level = "🟢 Low Risk"
            elif stage2_confidence >= 0.70:
                risk_level = "🟡 Medium Risk"
            else:
                risk_level = "🔴 High Risk"
        else:
            # Fake currency = risk is opposite (high confidence in fake = high risk)
            if stage2_confidence >= 0.90:
                risk_level = "🔴 High Risk"  # Definitely fake = dangerous
            elif stage2_confidence >= 0.70:
                risk_level = "🟡 Medium Risk"
            else:
                risk_level = "🟢 Low Risk"  # Low confidence in fake = less risk

        # Star Rating
        stars = min(5, max(1, int(stage2_confidence * 5)))
        star_rating = "⭐" * stars + "☆" * (5 - stars)

    # Print Results
    print("\n" + "="*70)
    print("🔍 COMPLETE PIPELINE ANALYSIS (STANDARD Logic)")
    print("="*70)

    print("\n📊 STAGE 1: BINARY CLASSIFICATION")
    print("-" * 70)
    if is_currency:
        print("✅ Result: 💵 CURRENCY DETECTED")
    else:
        print("✅ Result: 📋 NON-CURRENCY")
    print(f"   Confidence: {stage1_confidence*100:.2f}%")
    print(f"   Probability: {stage1_prob*100:.4f}%")

    if stage2_prob is not None:
        print("\n📊 STAGE 2: REAL/FAKE CLASSIFICATION (STANDARD Logic)")
        print("-" * 70)
        if is_real:
            print("✅ Result: ✓ REAL CURRENCY")
        else:
            print("✅ Result: ✗ FAKE CURRENCY")
        print(f"   Confidence: {stage2_confidence*100:.2f}%")
        print(f"   % Real: {percent_real:.2f}%")
        print(f"   % Fake: {percent_fake:.2f}%")
        print(f"\n   🏆 Authenticity Grade: {grade}")
        print(f"   ⚠️  Risk Level: {risk_level}")
        print(f"   📏 Threshold Distance: {threshold_distance:.2f}%")
        print(f"   ⭐ Reliability: {star_rating}")
    else:
        print("\n📊 STAGE 2: REAL/FAKE CLASSIFICATION")
        print("-" * 70)
        print("⊘ SKIPPED (Non-Currency Image)")

    print("\n" + "="*70)

    # Visualization
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.imshow(img)
    ax.axis('off')

    # Title
    if is_currency:
        title1 = "💵 CURRENCY DETECTED"
        color1 = '#228B22'
    else:
        title1 = "📋 NON-CURRENCY"
        color1 = '#00008B'

    ax.text(0.5, 1.08, title1,
           transform=ax.transAxes, fontsize=20, fontweight='bold',
           ha='center', color=color1)

    # Stage 1 Confidence
    ax.text(0.5, 1.02, f"Stage 1 Confidence: {stage1_confidence*100:.2f}%",
           transform=ax.transAxes, fontsize=14, ha='center',
           bbox=dict(boxstyle='round,pad=0.5', facecolor='#90EE90', alpha=0.8))

    # Stage 2 subtitle (if currency)
    if stage2_prob is not None:
        if is_real:
            title2 = "✓ REAL CURRENCY"
            color2 = '#006400'
            bg2 = '#90EE90'
        else:
            title2 = "✗ FAKE CURRENCY"
            color2 = '#8B0000'
            bg2 = '#FFB6C6'

        ax.text(0.5, 0.96, title2,
               transform=ax.transAxes, fontsize=16, fontweight='bold',
               ha='center', color=color2)

        stage2_metrics = f"""Confidence: {stage2_confidence*100:.2f}% | % Real: {percent_real:.2f}% | % Fake: {percent_fake:.2f}%
Grade: {grade} | {risk_level.replace('🟢', '').replace('🟡', '').replace('🔴', '').strip()} | Distance: {threshold_distance:.2f}% | {star_rating}"""
        ax.text(0.5, 0.88, stage2_metrics,
               transform=ax.transAxes, fontsize=10, ha='center',
               bbox=dict(boxstyle='round,pad=0.5', facecolor=bg2, alpha=0.8))

    plt.tight_layout()
    plt.show()

print("\n✅ Test function ready. Use: test_image('path/to/image.jpg')")

# Interactive testing
print("\n" + "="*70)
print("📤 Ready for testing")
print("="*70)
print("""
To test with Colab upload, run in next cell:

from google.colab import files
uploaded = files.upload()
for filename in uploaded:
    test_image(filename)
    import os
    os.remove(filename)
""")
