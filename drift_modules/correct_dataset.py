import torch
device = "cuda" if torch.cuda.is_available() else "cpu"
from timeit import default_timer as timer
from torch.types import Device
from shutil import copy
from collections import defaultdict
import pandas as pd
from pathlib import Path
import get_correct_preds_df
from torchvision import  transforms 
import os

import load_model
from tqdm.auto import tqdm
manual_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
    ])

import data_setup

dataloaders, class_names, dataset_sizes = data_setup.create_dataloaders()
model_ft2 = load_model.load_custom_pretrained_model('/content/cifar_model .pth',10)

data_dir = "/content/data/cifar10"
test_data_paths = list(Path(data_dir+"/test").glob("*/*.png"))
test_labels = [path.parent.stem for path in test_data_paths]
test_pred_dicts = get_correct_preds_df.pred_and_store(test_paths=test_data_paths,
                                 model=model_ft2,
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

output_dir = '/content/data/cifar10/correct_test/'

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
