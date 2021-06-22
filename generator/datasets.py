from pathlib import Path
from typing import Callable
from os import listdir

from torch.utils.data import Dataset as Base
from PIL import Image


class RealImageDataset(Base):
    def __init__(
            self,
            img_dir: Path,
            transform: Callable = None,
            target_transform: Callable = None
    ):
        self.img_dir = img_dir
        self.file_names = listdir(img_dir)
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx: int):
        img_path = str(self.img_dir / self.file_names[idx])
        image = Image.open(img_path)
        label = 1.0
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label
