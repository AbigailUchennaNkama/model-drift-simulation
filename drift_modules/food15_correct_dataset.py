import os
import torch
from timeit import default_timer as timer
from torch.types import Device
from shutil import copy
from collections import defaultdict
import pandas as pd
from pathlib import Path
from tqdm.auto import tqdm
from torchvision import  transforms 
device = "cuda" if torch.cuda.is_available() else "cpu"


from get_correct_preds_df import pred_and_store
from load_model import load_custom_pretrained_model
from data_path_setup import create_dataloaders


manual_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
    ])


#get dataloaders and class names
loaded_food_model_c15 = load_custom_pretrained_model(model_path='./food_model.pth', num_classes=15)#load pretrained model
#load pretrained model
loaded_food_model_c15 = load_custom_pretrained_model(model_path='./food_model.pth', num_classes=15)

class_names=["pizza", "steak", "sushi", "spaghetti_bolognese",
                               "hot_and_sour_soup", "chicken_wings", "french_fries",
                               "ice_cream", "greek_salad", "chocolate_cake","baby_back_ribs","baklava",
                               "beef_carpaccio","beef_tartare","beet_salad"]

test_pred_dicts = pred_and_store(data_dir='./food-15',
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

output_dir = './food-15/correct_test/'

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
