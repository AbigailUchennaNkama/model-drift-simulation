"""
Contains functionality for creating PyTorch DataLoaders for
image classification data.
"""
import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torchvision.datasets import ImageFolder
#NUM_WORKERS = os.cpu_count()
torch.manual_seed = 42
torch.cuda.manual_seed=42

# Define data transformations
manual_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
    ])
  

data_dir = "/content/data/cifar10"

def get_data():

  """Creates training and testing DataLoaders.

    Takes in a training directory and testing directory path and turns
    them into PyTorch Datasets and then into PyTorch DataLoaders.

    Returns:
      A tuple of (train_dataloader, test_dataloader, class_names).
      Where class_names is a list of the target classes.
    Example usage:
        val_dataloader, train_dataloader, class_names = \
        = create_dataloaders()
  """

  dataset = ImageFolder(data_dir+"/train")

  correct_test_dir = ImageFolder(data_dir+"/correct_test", transform=manual_transforms)

  # Create dataloaders
  correct_test_dir = ImageFolder(data_dir+"/correct_test", transform=manual_transforms)

  #dataset_sizes = {x: len(dataloader.dataset) for x, dataloader in dataloaders.items()}
  class_names = correct_test_dir.classes

  return dataloaders, class_names
