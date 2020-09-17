import torch
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from utils.preprocessing import get_train_transforms, get_val_transforms
from models.trainer import Fitter
from efficientnet_pytorch import EfficientNet


def train(config):
    train_transforms = get_train_transforms(config)
    val_transforms = get_val_transforms(config)
    train_data = ImageFolder(config.train_folder, transform=train_transforms)
    val_data = ImageFolder(config.val_folder, transform=val_transforms)
    train_dataloader = DataLoader(train_data, batch_size=config.batch_size, shuffle=True,
                                  num_workers=config.num_workers, drop_last=False)
    val_dataloader = DataLoader(val_data, batch_size=config.batch_size, num_workers=config.num_workers,
                                drop_last=False)

    model = EfficientNet.from_pretrained('efficientnet-b0')
    model._fc = torch.nn.Linear(in_features=1280, out_features=2, bias=True)
    model._swish = torch.nn.Sigmoid()
    fitter = Fitter(model, config)
    fitter.fit(train_dataloader, val_dataloader)
