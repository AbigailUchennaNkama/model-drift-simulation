import os
import shutil
from pathlib import Path

from torchvision.datasets.utils import download_url
import tarfile
print("creating cifar10 dataset...")
dataset_url = "https://s3.amazonaws.com/fast-ai-imageclas/cifar10.tgz "
download_url(dataset_url, '.')
with tarfile.open("./cifar10.tgz ", "r:gz") as tar:
    tar.extractall(path="./data")

print("creating cifar09 dataset...")
data_dir = "./data/cifar10"
test_data_paths = list(Path(data_dir+"/train").glob("*/*.png"))

# Exclude the "dog" class
test_data_paths = [path for path in test_data_paths if path.parent.stem != "dog"]
test_labels = [path.parent.stem for path in test_data_paths]

data_dir = "./data/cifar9"
train_dir = os.path.join(data_dir, "train")
# Define the classes to include (excluding "dog")
included_classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'frog', 'horse', 'ship', 'truck']

# Create subfolders for included classes
for class_name in included_classes:
    class_dir = os.path.join(train_dir, class_name)
    os.makedirs(class_dir, exist_ok=True)

# List all files in the test directory (using test_data_paths)
all_files = list(test_data_paths)

# Move files to their respective subfolders
for file_path in all_files:
    class_name = file_path.parent.name
    if class_name in included_classes:
        dest_dir = os.path.join(train_dir, class_name)
        shutil.copy(str(file_path), dest_dir)
print("Images organized into subfolders based on class names while keeping copies.")
