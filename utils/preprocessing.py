from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt
import numpy as np
import albumentations as A


def showImg_grid(img):
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


def get_train_transforms(cfg):
    transforms = A.Compose([
        A.CLAHE(),
        A.RandomRotate90(),
        A.Blur(),
        A.OpticalDistortion(),
        A.HueSaturationValue(),
        A.Resize(*cfg.image_size),
        A.Normalize(),
        ToTensorV2()
    ])
    return lambda img: transforms(image=np.array(img))['image']


def get_val_transforms(cfg):
    transforms = A.Compose([
        A.Resize(*cfg.image_size),
        A.Normalize(),
        ToTensorV2()
    ])
    return lambda img: transforms(image=np.array(img))['image']


def get_test_transforms(cfg):
    transforms = A.Compose([
        A.Resize(*cfg.image_size),
        A.Normalize(),
        ToTensorV2()
    ])
    return transforms
