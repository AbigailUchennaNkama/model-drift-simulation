from pathlib import Path
import os
import shutil
import torch
import torchvision
import torchvision.datasets as datasets
import random
import pathlib
from timeit import default_timer as timer
from torch.types import Device
from shutil import copy
from collections import defaultdict
import pandas as pd
from tqdm.auto import tqdm
from torchvision import  transforms
device = "cuda" if torch.cuda.is_available() else "cpu"

data_dir = pathlib.Path("./data")


# Get training data
train_data = datasets.Food101(root=data_dir,
                              split="train",
                              # transform=transforms.ToTensor(),
                              download=True)

# Get testing data
test_data = datasets.Food101(root=data_dir,
                             split="test",
                             # transform=transforms.ToTensor(),
                             download=True)
# Get random 10% of training images

#Setup data paths
data_path = data_dir / "food-101" / "images"
target_classes = ["pizza", "steak", "sushi", "spaghetti_bolognese",
                  "hot_and_sour_soup", "chicken_wings", "french_fries",
                  "ice_cream", "greek_salad", "chocolate_cake",
                  "baby_back_ribs", "baklava", "beef_carpaccio",
                  "beef_tartare", "beet_salad"]

# Change amount of data to get (e.g. 0.1 = random 10%, 0.2 = random 20%)
amount_to_get = 1.0

# Create function to separate a random amount of data
def get_subset(image_path=data_path,
               data_splits=["train", "test"], 
               target_classes= ["pizza", "steak", "sushi", "spaghetti_bolognese",
                  "hot_and_sour_soup", "chicken_wings", "french_fries",
                  "ice_cream", "greek_salad", "chocolate_cake",
                  "baby_back_ribs", "baklava", "beef_carpaccio",
                  "beef_tartare", "beet_salad"],
                  amount=0.1,
                  seed=42):
    random.seed(42)
    label_splits = {}
    
    # Get labels
    for data_split in data_splits:
        print(f"[INFO] Creating image split for: {data_split}...")
        label_path = data_dir / "food-101" / "meta" / f"{data_split}.txt"
        with open(label_path, "r") as f:
            labels = [line.strip("\n") for line in f.readlines() if line.split("/")[0] in target_classes] 
        
        # Get random subset of target classes image ID's
        number_to_sample = round(amount * len(labels))
        print(f"[INFO] Getting random subset of {number_to_sample} images for {data_split}...")
        sampled_images = random.sample(labels, k=number_to_sample)
        
        # Apply full paths
        image_paths = [pathlib.Path(str(image_path / sample_image) + ".jpg") for sample_image in sampled_images]
        label_splits[data_split] = image_paths
    return label_splits

label_splits = get_subset(amount=amount_to_get)

# Create target directory path
target_dir_name = f"./data/food15c{str(int(amount_to_get*100))}_percent"
print(f"Creating directory: '{target_dir_name}'")

# Setup the directories
target_dir = pathlib.Path(target_dir_name)
target_dir.mkdir(parents=True, exist_ok=True)

import shutil

for image_split in label_splits.keys():
    for image_path in label_splits[str(image_split)]:
        dest_dir = target_dir / image_split / image_path.parent.stem / image_path.name
        if not dest_dir.parent.is_dir():
            dest_dir.parent.mkdir(parents=True, exist_ok=True)
        print(f"[INFO] Copying {image_path} to {dest_dir}...")
        shutil.copy(image_path, dest_dir)

# test and creat folders with correct predictions
from data_path_setup import create_dataloaders
from get_correct_preds_df import pred_and_store
from load_model import load_custom_pretrained_model
from data_path_setup import create_dataloaders


manual_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
    ])

#load pretrained model

loaded_food_model_c15 = load_custom_pretrained_model(model_path='./food_model.pth', num_classes=15)

dataloaders, class_names, dataset_sizes =  create_dataloaders(val_size=2000, data_paths='./data/food15c100_percent/train')

test_pred_dicts = pred_and_store(data_dir='./data/food15c100_percent',
                                 model=loaded_food_model_c15,
                                 transform=manual_transforms,
                                 class_names=class_names,
                                 device=device)

test_pred_df = pd.DataFrame(test_pred_dicts)
correct_preds = test_pred_df[test_pred_df["correct"]==True]
correct_pred_img = list(correct_preds["image_path"])
correct_pred_img = [str(path) for path in correct_pred_img]

# Create a defaultdict to store lists of file paths for each category
category_lists = defaultdict(lambda: [])

# Loop through the file paths and categorize them
for file_path in correct_pred_img:
    # Split the file path using '/'
    parts = file_path.split('/')
    # The category is the second-to-last part of the path
    category = parts[-2]
    # Append the file path to the corresponding category list
    category_lists[category].append(file_path)

output_dir = './data/food15c100_percent/correct_test'

# Create the output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Create a folder for each category and copy the images with category name in the filename
for category, paths in category_lists.items():
    category_dir = os.path.join(output_dir, category)
    os.makedirs(category_dir, exist_ok=True)
    for path in paths:
        # Get the filename from the path
        filename = os.path.basename(path)
        # Modify the filename to include the category name
        new_filename = f"{category}_{filename}"
        # Copy the image to the category folder with the new filename
        copy(path, os.path.join(category_dir, new_filename))

print("Folders created and images organized.")

