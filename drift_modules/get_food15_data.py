import random
import pathlib
import shutil
from pathlib import Path
import os
import shutil
import torch
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import random
import pathlib

data_dir = pathlib.Path("../data")

# Get training data
train_data = datasets.Food101(root=data_dir, download=True,split="train")

# Get testing data
test_data = datasets.Food101(root=data_dir, download= True,split="test")

# Setup data paths
data_path = data_dir / "food-101" / "images"
target_classes = ["pizza", "steak", "sushi", "spaghetti_bolognese",
                  "hot_and_sour_soup", "chicken_wings", "french_fries",
                  "ice_cream", "greek_salad", "chocolate_cake",
                  "baby_back_ribs", "baklava", "beef_carpaccio",
                  "beef_tartare", "beet_salad"]

# Change amount of data to get (e.g. 0.1 = random 10%, 0.2 = random 20%)
amount_to_get = 0.1

# Create function to separate a random amount of data
def get_subset(image_path=data_path,
               target_classes=target_classes,
               amount=0.1,
               seed=42):
    random.seed(seed)
    label_splits = {}

    for data_split in ["train", "test"]:
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
target_dir_name = f"../data/food15c{str(int(amount_to_get*100))}_percent"
print(f"Creating directory: '{target_dir_name}'")

# Setup the directories
target_dir = pathlib.Path(target_dir_name)
target_dir.mkdir(parents=True, exist_ok=True)

for image_split in label_splits.keys():
    for image_path in label_splits[str(image_split)]:
        dest_dir = target_dir / image_split / image_path.parent.stem / image_path.name
        if not dest_dir.parent.is_dir():
            dest_dir.parent.mkdir(parents=True, exist_ok=True)
        print(f"[INFO] Copying {image_path} to {dest_dir}...")
        shutil.copy2(image_path, dest_dir)

# Zip food15 images
shutil.make_archive(target_dir, format="zip", root_dir=target_dir)

# Extract the zip file to the 'food-15' directory
extracted_dir = pathlib.Path("food-15")
extracted_dir.mkdir(parents=True, exist_ok=True)
shutil.unpack_archive(f"{target_dir}.zip", extracted_dir)
