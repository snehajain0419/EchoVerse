import numpy as np
import cv2
import os
import csv
from image_processing import func  # Assuming func processes the image

# Define absolute dataset path
path = r"C:\Users\sneha jain\Desktop\signlanguage\Sign-Language-to-Text-Conversion-main\data\train"  # Update this path if needed
path1 = "data2"

# Verify if dataset path exists
if not os.path.exists(path):
    print(f"Error: Dataset path '{path}' does not exist.")
    exit(1)

# Create necessary directories
os.makedirs("data2/train", exist_ok=True)
os.makedirs("data2/test", exist_ok=True)

# CSV Header (label + 64x64 pixels)
header = ['label'] + ["pixel" + str(i) for i in range(64 * 64)]

# Open CSV file to store pixel data
csv_path = "data2/dataset.csv"
with open(csv_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(header)  # Write the header

# Variables
total_images = 0
train_count = 0
test_count = 0
label = 0  # Class label counter

# Iterate through dataset directories
print(f"Scanning dataset directory: {path}")
for dirname in sorted(os.listdir(path)):  # Sort for consistent processing
    dir_path = os.path.join(path, dirname)
    
    if not os.path.isdir(dir_path):  # Skip if not a directory
        continue

    print(f"Processing category '{dirname}'...")

    # Create subdirectories in train/test
    train_dir = os.path.join(path1, "train", dirname)
    test_dir = os.path.join(path1, "test", dirname)
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    # Read image files
    files = [f for f in os.listdir(dir_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    if not files:
        print(f"Warning: No valid images found in '{dirname}'")
        continue

    num_train = int(0.75 * len(files))  # 75% Train, 25% Test
    i = 0

    for file in sorted(files):  # Sort for consistency
        total_images += 1
        actual_path = os.path.join(dir_path, file)
        img = cv2.imread(actual_path, 0)  # Read in grayscale

        if img is None:
            print(f"Skipping {actual_path}, could not read image.")
            continue

        bw_image = func(actual_path)  # Apply preprocessing

        if bw_image is None:
            print(f"Skipping {actual_path}, preprocessing failed.")
            continue

        # Flatten image for CSV storage
        img_flat = bw_image.flatten().tolist()
        row = [label] + img_flat

        with open(csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(row)

        # Save images in train/test split
        if i < num_train:
            train_count += 1
            cv2.imwrite(os.path.join(train_dir, file), bw_image)
        else:
            test_count += 1
            cv2.imwrite(os.path.join(test_dir, file), bw_image)

        i += 1  # Increment file counter

    label += 1  # Increment class label

# Summary
print("\nProcessing complete!")
print(f"Total images processed: {total_images}")
print(f"Training images: {train_count}")
print(f"Testing images: {test_count}")
