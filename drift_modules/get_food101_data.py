from pathlib import Path
import os
import shutil
import torch
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import random
import pathlib

data_dir = pathlib.Path("./data")

# Get training data
train_data = datasets.Food101(root=data_dir,download=True,
                              transform = transforms.Compose([
                              transforms.ToTensor(),
                              transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]))
                              

print("Creating food-10 dataset...")
image_dir = "./data/food-101"
food_data_paths = list(Path(image_dir + "/images").glob("*/*.jpg"))

# Define the classes to include (excluding "truck")
included_classes = ["pizza", "steak", "sushi", "spaghetti_bolognese",\
                    "hot_and_sour_soup","chicken_wings","french_fries","ice_cream",\
                    "greek_salad","chocolate_cake"]

# Exclude the other classes class
food_data_paths = [path for path in food_data_paths if path.parent.stem in included_classes]

food10_dir = "./data/food-10"
img_dir = os.path.join(food10_dir, "images")

# Create subfolders for included classes
for class_name in included_classes:
    class_dir = os.path.join(img_dir, class_name)
    os.makedirs(class_dir, exist_ok=True)

# List all files in the train directory (using train_data_paths)
all_files = list(food_data_paths)

# Move files to their respective subfolders
for file_path in all_files:
    class_name = file_path.parent.name
    if class_name in included_classes:
        dest_dir = os.path.join(img_dir, class_name)
        shutil.copy(str(file_path), dest_dir)

print("Images organized into subfolders based on class names.")
