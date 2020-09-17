from scripts import train
from config import TrainConfig
from efficientnet_pytorch import EfficientNet
import torch
from models.trainer import Fitter
from torch.utils.data import DataLoader
from utils.preprocessing import get_test_transforms
from utils.datasets import TestDataset
from tqdm import tqdm
import pandas as pd
import sys

if __name__ == '__main__':
    cfg = TrainConfig

    # uncomment if you want to retrain
    # train(TrainConfig)

    test_path = sys.argv[-1]
    model = EfficientNet.from_pretrained('efficientnet-b0')
    model._fc = torch.nn.Linear(in_features=1280, out_features=2, bias=True)
    model._swish = torch.nn.Sigmoid()

    fitter = Fitter(model, cfg)
    fitter.load("efficientnet-b0/best-checkpoint-008epoch.bin")
    model = fitter.get_model()

    test_transforms = get_test_transforms(cfg)
    data = TestDataset(test_path, transform=test_transforms)
    dataloader = DataLoader(data, cfg.batch_size, num_workers=cfg.num_workers)

    submission = {"file": [], "label": []}
    for imgs, names in tqdm(dataloader):
        pred = model(imgs.to(cfg.device)).argmax(dim=1).cpu().numpy()

        submission['file'].extend(names)
        submission['label'].extend(pred)

    df = pd.DataFrame().from_dict(submission)
    df.to_csv("submission.csv",index=False)



