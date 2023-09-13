import matplotlib.pyplot as plt
import numpy as np
from torchvision.utils import make_grid
# functions to show an image
batch_size =64
from drift_module.data_setup import *
_, class_names, _ = create_dataloaders()


def show_image(dataloader):
  dataiter = iter(dataloader)
  img, labels = next(dataiter)
  img = make_grid(img, nrow=12)
  img = img / 2 + 0.5     # unnormalize
  npimg = img.numpy()     #convert to numpy
  plt.imshow(np.transpose(npimg, (1, 2, 0)))
  print(' '.join(f'{class_names[labels[j]]:5s}' for j in range(batch_size)))
  plt.show()
