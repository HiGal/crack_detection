import torch
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from utils.preprocessing import showImg_grid
from torchvision import transforms


def train(root_folder, batch_size, shuffle=True):
    data = ImageFolder(root_folder, transform=transforms.ToTensor())
    dataloader = DataLoader(data, batch_size=batch_size, shuffle=shuffle)
    batch, label = next(iter(dataloader))
    print(label)
    grid = make_grid(batch)
    showImg_grid(grid)