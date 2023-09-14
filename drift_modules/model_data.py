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

    Takes in directory of all images correctly predicted by the model and turns
    them into PyTorch Datasets and then into PyTorch DataLoaders.

    Returns:
      A tuple of (model_dataloader, class_names).
      Where class_names is a list of the target classes.
    Example usage:
        model_dataloader, class_names = \
        model_dataloaders()
  """



  model_data_dir = ImageFolder(data_dir+"/correct_test", transform=manual_transforms)

  # Create dataloaders
  model_dataloader = DataLoader(model_data_dir, batch_size=64, shuffle=False)

  class_names = model_data_dir.classes

  return model_dataloader, class_names
