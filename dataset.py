from torchvision.datasets import ImageFolder, DatasetFolder
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from pprint import pprint
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt


def show(img):
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


if __name__ == '__main__':
    data = ImageFolder("data", transform=transforms.ToTensor())
    dataloader = DataLoader(data, batch_size=16, shuffle=True)
    batch, label = next(iter(dataloader))
    print(label)
    grid = make_grid(batch)
    show(grid)
