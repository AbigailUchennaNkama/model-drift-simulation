import os
import shutil
from pathlib import Path

from torchvision.datasets.utils import download_url
import tarfile

print("Creating cifar10 dataset...")
dataset_url = "https://s3.amazonaws.com/fast-ai-imageclas/cifar10.tgz"
download_url(dataset_url, '.')

with tarfile.open("./cifar10.tgz", "r:gz") as tar:
    tar.extractall(path="./data")

print("Creating cifar9 dataset...")
data_dir = "./data/cifar10"
train_data_paths = list(Path(data_dir + "/train").glob("*/*.png"))

# Exclude the "dog" class
train_data_paths = [path for path in train_data_paths if path.parent.stem != "dog"]

data_dir = "./data/cifar9"
train_dir = os.path.join(data_dir, "train")

# Define the classes to include (excluding "dog")
<<<<<<< HEAD
included_classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'frog', 'horse', 'ship', 'truck']
=======
included_classes = ['airplaine', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship']
>>>>>>> 20a06dc3437c0eba835935e5b9dd1c9c45abd592

# Create subfolders for included classes
for class_name in included_classes:
    class_dir = os.path.join(train_dir, class_name)
    os.makedirs(class_dir, exist_ok=True)

# List all files in the train directory (using train_data_paths)
all_files = list(train_data_paths)

# Move files to their respective subfolders
for file_path in all_files:
    class_name = file_path.parent.name
    if class_name in included_classes:
        dest_dir = os.path.join(train_dir, class_name)
        shutil.copy(str(file_path), dest_dir)

print("Images organized into subfolders based on class names.")
