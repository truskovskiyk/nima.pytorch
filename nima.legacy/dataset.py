from pathlib import Path

import pandas as pd

from torch.utils.data import Dataset
from torchvision.datasets.folder import default_loader


class AVADataset(Dataset):
    def __init__(self, path_to_csv: Path, images_path: Path, transform):
        self.df = pd.read_csv(path_to_csv)
        self.images_path = images_path
        self.transform = transform

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, item):
        row = self.df.iloc[item]

        image_id = row['image_id']
        image_path = self.images_path / f'{image_id}.jpg'
        image = default_loader(image_path)
        x = self.transform(image)

        y = row[1:].values.astype('float32')
        p = y / y.sum()

        return x, p
