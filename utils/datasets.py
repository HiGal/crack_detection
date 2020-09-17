from torch.utils.data import Dataset
import os
import cv2
from pathlib import Path


class TestDataset(Dataset):

    def __init__(self, root, transform=None):
        self.root = Path(root)
        self.images = os.listdir(root)
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_name = self.images[idx]
        img_path = self.root / image_name
        img = cv2.imread(str(img_path))

        if self.transform:
            img = self.transform(image=img)['image']
        return img, image_name
