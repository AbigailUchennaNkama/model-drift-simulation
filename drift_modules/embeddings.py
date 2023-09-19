import torch
import torchvision
import torchvision.transforms as transforms
from sklearn.manifold import TSNE
from torchvision import models
from joblib import dump
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load CIFAR10 dataset
transform = transforms.Compose([
    transforms.Resize(224),  # Resize to the input size expected by ResNet18
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize with ImageNet stats
])

def get_embeddings(model,data_path):
  testset = ImageFolder(data_paths, transform=transform)
  testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False)

  # Load pre-trained ResNet18 and remove the final classification layer
  model = model
  model = torch.nn.Sequential(*(list(model.children())[:-1])).to(device) # Remove the last classification layer
  model.eval()

  # Extract embeddings
  embeddings, labels, img = [], [], []
  with torch.no_grad():
      for images, lbls in testloader:
          images,lbls = images.to(device), lbls.to(device)
          emb = model(images).squeeze(-1).squeeze(-1)  # After removing the last layer, there might be extra dimensions
          embeddings.append(emb)
          labels.append(lbls)
          img.append(images)
  embeddings = torch.cat(embeddings).cpu()
  labels = torch.cat(labels).cpu()
  img = torch.cat(img).cpu()


