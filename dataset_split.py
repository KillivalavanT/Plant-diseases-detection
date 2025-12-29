import os
import shutil
import random

# base dataset path
base_dir = "dataset/raw"
train_dir = "dataset/train"
test_dir = "dataset/test"

split_ratio = 0.8  # 80% train, 20% test

classes = ["Blight", "Common_Rust", "Healthy"]

# Create train/test folders
for cls in classes:
    os.makedirs(os.path.join(train_dir, cls), exist_ok=True)
    os.makedirs(os.path.join(test_dir, cls), exist_ok=True)

for cls in classes:
    class_path = os.path.join(base_dir, cls)
    images = os.listdir(class_path)
    random.shuffle(images)

    split_point = int(len(images) * split_ratio)

    train_images = images[:split_point]
    test_images = images[split_point:]

    # Move train images
    for img in train_images:
        shutil.copy(os.path.join(class_path, img), os.path.join(train_dir, cls))

    # Move test images
    for img in test_images:
        shutil.copy(os.path.join(class_path, img), os.path.join(test_dir, cls))

    print(f"{cls}: {len(train_images)} train, {len(test_images)} test")

print("Dataset split completed successfully!")
