import os
import cv2
import numpy as np

minValue = 70

# Function to process an individual image
def func(path):
    frame = cv2.imread(path)
    
    if frame is None:
        print(f"Error: Could not read {path}")
        return None  # Skip processing if image is unreadable

    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian Blur
    blur = cv2.GaussianBlur(gray, (5, 5), 2)

    # Adaptive Thresholding
    th3 = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

    # Otsu's Thresholding
    ret, res = cv2.threshold(th3, minValue, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    return res

# Function to process all images in a directory (including subdirectories)
def process_images_in_directory(input_directory, output_directory):
    for root, _, files in os.walk(input_directory):  # Recursively scan all subfolders
        relative_path = os.path.relpath(root, input_directory)  # Get subfolder path
        output_subfolder = os.path.join(output_directory, relative_path)

        if not os.path.exists(output_subfolder):
            os.makedirs(output_subfolder)

        for filename in files:
            file_path = os.path.join(root, filename)

            # Check if it's an image file
            if file_path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                print(f"Processing {filename} in {root}...")

                # Process the image
                processed_image = func(file_path)
                if processed_image is None:
                    continue  # Skip if image processing failed

                # Construct the output file path
                output_path = os.path.join(output_subfolder, filename)

                # Save the processed image
                cv2.imwrite(output_path, processed_image)

# Input directory (now processes all subfolders automatically)
input_directory = r"C:\Users\sneha jain\Desktop\Sign-Language-to-Text-Conversion-main\Sign-Language-to-Text-Conversion-main\Source Code\data\train"

# Output base directory
output_directory = "processed_images"

# Process images from the entire train folder (including all subfolders)
process_images_in_directory(input_directory, output_directory)

print("Processing complete.")
