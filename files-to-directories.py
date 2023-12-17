# Standard Library Imports
import os
import shutil
import random

# Scientific and Data Processing Libraries
from sklearn.model_selection import train_test_split

# Define source and destination directories
src_directory = 'original_data'  # Original data path

# Dictionary to store class names and image counts
class_image_counts = {}

# Iterate over each directory in the dataset directory
for class_name in os.listdir(src_directory):
    class_path = os.path.join(src_directory, class_name)
    # Check if it's a directory
    if os.path.isdir(class_path):
        # Count the number of image files in this directory
        num_images = len([name for name in os.listdir(class_path) if os.path.isfile(os.path.join(class_path, name))])
        class_image_counts[class_name] = num_images

# Find the maximum length of class name for alignment
max_class_name_length = max(len(class_name) for class_name in class_image_counts)

# Print the class names and image counts
for class_name, count in class_image_counts.items():
    print(f"Class: {class_name:<{max_class_name_length}} Number of images: {count}")

def split_data(src_directory, dest_directory, train_size=0.8, selected_classes=None):
    classes = [d for d in os.listdir(src_directory) if os.path.isdir(os.path.join(src_directory, d)) and d in selected_classes]

    for cls in classes:
        # Create train and test directories for each class
        train_cls_dir = os.path.join(dest_directory, 'train', cls)
        test_cls_dir = os.path.join(dest_directory, 'test', cls)
        os.makedirs(train_cls_dir, exist_ok=True)
        os.makedirs(test_cls_dir, exist_ok=True)

        # Get all image files in the current class directory
        all_images = os.listdir(os.path.join(src_directory, cls))
        
        # If a limit is set and the class is Dolphin or Sharks, apply the limit
        if cls in ['Dolphin', 'Sharks']:
            all_images = random.sample(all_images, min(len(all_images), 500))

        
        # Split images into train and test sets
        train_images, test_images = train_test_split(all_images, train_size=train_size, random_state=1)

        # Copy images to train and test directories
        for img in train_images:
            src_path = os.path.join(src_directory, cls, img)
            dst_path = os.path.join(train_cls_dir, img)
            if not os.path.exists(dst_path):
                shutil.copy(src_path, dst_path)

        for img in test_images:
            src_path = os.path.join(src_directory, cls, img)
            dst_path = os.path.join(test_cls_dir, img)
            if not os.path.exists(dst_path):
                shutil.copy(src_path, dst_path)

                
dest_directory = 'images'  # Destination path

# Specify your five selected classes here
selected_classes = ['Dolphin', 'Eel', 'Penguin', 'Seal', 'Sharks']

# Create the destination directory if it doesn't exist
os.makedirs(dest_directory, exist_ok=True)

# Split the data for the selected classes
split_data(src_directory, dest_directory, selected_classes=selected_classes)
