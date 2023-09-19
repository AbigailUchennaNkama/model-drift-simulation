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
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    'val': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}


def create_dataloaders(data_paths):

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
 
  dataset = ImageFolder(data_paths)

  val_size = 5000
  train_size = len(dataset) - val_size

  # Split dataset into train and val based on specified sizes
  train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

  # Apply transformations to train and val datasets
  train_dataset.dataset.transform = data_transforms['train']
  val_dataset.dataset.transform = data_transforms['val']

  # Create dataloaders
  dataloaders = {
        'train': torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True),
        'val': torch.utils.data.DataLoader(val_dataset, batch_size=64, shuffle=False)
    }

  dataset_sizes = {x: len(dataloader.dataset) for x, dataloader in dataloaders.items()}
  class_names = dataset.classes

  return dataloaders, class_names, dataset_sizes
