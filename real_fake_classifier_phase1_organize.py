"""
PHASE 1: Organize Real/Fake Data from Existing Folders
Copy images from /data/data/ to /real_fake_classifier_data/
"""

import shutil
import os
from pathlib import Path

print("="*70)
print("PHASE 1: Organize Real/Fake Currency Data")
print("="*70)

# Source paths (existing data)
source_real = '/content/drive/MyDrive/AI_Dataset/data/data/real'
source_fake = '/content/drive/MyDrive/AI_Dataset/data/data/fake'

# Destination paths (new organized structure)
base_path = '/content/drive/MyDrive/AI_Dataset/real_fake_classifier_data'
dest_real = os.path.join(base_path, 'real')
dest_fake = os.path.join(base_path, 'fake')

# Create destination directories
os.makedirs(dest_real, exist_ok=True)
os.makedirs(dest_fake, exist_ok=True)

print(f"\n📁 Source paths:")
print(f"   Real: {source_real}")
print(f"   Fake: {source_fake}")

print(f"\n📁 Destination paths:")
print(f"   Real: {dest_real}")
print(f"   Fake: {dest_fake}")

# Copy real currency images
print("\n🔄 Copying real currency images...")
real_count = 0
if os.path.exists(source_real):
    for file in os.listdir(source_real):
        if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp')):
            try:
                src = os.path.join(source_real, file)
                dst = os.path.join(dest_real, file)
                shutil.copy2(src, dst)
                real_count += 1
            except Exception as e:
                print(f"   ⚠️  Error copying {file}: {str(e)[:50]}")

print(f"✓ Copied {real_count} real currency images")

# Copy fake currency images
print("\n🔄 Copying fake currency images...")
fake_count = 0
if os.path.exists(source_fake):
    for file in os.listdir(source_fake):
        if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp')):
            try:
                src = os.path.join(source_fake, file)
                dst = os.path.join(dest_fake, file)
                shutil.copy2(src, dst)
                fake_count += 1
            except Exception as e:
                print(f"   ⚠️  Error copying {file}: {str(e)[:50]}")

print(f"✓ Copied {fake_count} fake currency images")

# Verify
print("\n✅ Verification:")
real_files = len([f for f in os.listdir(dest_real) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))])
fake_files = len([f for f in os.listdir(dest_fake) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))])

print(f"   Real images in destination: {real_files}")
print(f"   Fake images in destination: {fake_files}")
print(f"   Total: {real_files + fake_files}")

if real_count == 0 or fake_count == 0:
    print("\n⚠️  WARNING: Some folders may be empty!")
    print(f"   Real: {real_count} images")
    print(f"   Fake: {fake_count} images")
    print("\n   Please check source paths:")
    print(f"   - {source_real}")
    print(f"   - {source_fake}")
else:
    print("\n" + "="*70)
    print("✓ PHASE 1 COMPLETE!")
    print("="*70)
    print(f"\n✅ Ready for Phase 2 Training!")
    print(f"   - {real_count} real currency images")
    print(f"   - {fake_count} fake currency images")
    print(f"\n📋 Next: Run Phase 2 training script")
