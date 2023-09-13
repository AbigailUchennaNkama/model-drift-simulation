import torch
import torch.nn as nn
import torchvision.models as models

def load_custom_pretrained_model(model_path, num_classes=10):
    """
    Load a custom pretrained model with modifications.

    Args:
        model_path (str): The path to the pretrained model checkpoint.
        num_classes (int): The number of output classes for the final classification layer.

    Returns:
        nn.Module: The loaded pretrained model.

    Example Usage:
        from drift_module.load_model import load_custom_pretrained_model
        model_path = "/content/cifar_model.pth"  # Replace with actual model path
        num_classes = 10  # Change this to the number of classes in your task
        model_ft = load_custom_pretrained_model(model_path, num_classes)

    """
    # Load a pretrained model (e.g., ResNet18)
    model = models.resnet18(weights='IMAGENET1K_V1')

    # Modify the final fully connected layer for a new classification task
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)

    # Move the model to the desired device (CPU or GPU)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Load the saved state_dict
    model.load_state_dict(torch.load(model_path, map_location=device))

    return model
