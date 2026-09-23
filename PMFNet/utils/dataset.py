"""
Dataset classes and data loading utilities
"""
import numpy as np
import torch
from torch.utils.data import Dataset
import albumentations as A
import torchvision.transforms as transforms


class PolypDS(Dataset):
    """Polyp Dataset class"""
    def __init__(self, data_path, type=None, transform=None):
        super().__init__()
        data_np = np.load(data_path)
        self.images = data_np[f"{type}_img"]
        self.masks = data_np[f"{type}_msk"].squeeze(-1)
        self.transform = transform

    def __getitem__(self, idx):
        img = self.images[idx]
        msk = self.masks[idx]

        if self.transform is not None:
            transformed = self.transform(image=img, mask=msk)
            img = transformed["image"]
            msk = transformed["mask"]

        img = transforms.ToTensor()(img)
        msk = np.expand_dims(msk, axis=-1)
        msk = transforms.ToTensor()(msk)

        return img, msk

    def __len__(self):
        return len(self.images)


def get_augmentation_transforms(image_size=256):
    """
    Get data augmentation transformations
    
    Args:
        image_size: Target image size (default: 256)
    
    Returns:
        train_transform, val_transform: Augmentation pipelines
    """
    train_transform = A.Compose([
        A.Resize(image_size, image_size),
        A.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.2, rotate_limit=30, p=0.5),
        A.RGBShift(r_shift_limit=25, g_shift_limit=25, b_shift_limit=25, p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])

    val_transform = A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])

    return train_transform, val_transform
