import os

import pandas as pd
import numpy as np

from torch.utils.data import Dataset
from torchvision.datasets.folder import default_loader


from nima.train.utils import SCORE_NAMES


class AVADataset(Dataset):
    def __init__(self, path_to_csv: str, images_path: str, transform):
        self.df = pd.read_csv(path_to_csv)
        self.images_path = images_path
        self.transform = transform

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, item):
        row = self.df.iloc[item]
        y = np.array([row[k] for k in SCORE_NAMES])
        p = y / y.sum()

        image_id = row['image_id']
        image_path = os.path.join(self.images_path, f'{image_id}.jpg')
        image = default_loader(image_path)
        x = self.transform(image)
        return x, p.astype('float32')


def _create_train_data_part(params: TrainParams):
    train_csv_path = os.path.join(params.path_to_save_csv, 'train.csv')
    val_csv_path = os.path.join(params.path_to_save_csv, 'val.csv')

    transform = Transform()
    train_ds = AVADataset(train_csv_path, params.path_to_images, transform.train_transform)
    val_ds = AVADataset(val_csv_path, params.path_to_images, transform.val_transform)

    train_loader = DataLoader(train_ds, batch_size=params.batch_size, num_workers=params.num_workers, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=params.batch_size, num_workers=params.num_workers, shuffle=False)

    return train_loader, val_loader


def _create_val_data_part(params: TrainParams):
    val_csv_path = os.path.join(params.path_to_save_csv, 'val.csv')
    test_csv_path = os.path.join(params.path_to_save_csv, 'test.csv')

    transform = Transform()
    val_ds = AVADataset(val_csv_path, params.path_to_images, transform.val_transform)
    test_ds = AVADataset(test_csv_path, params.path_to_images, transform.val_transform)

    val_loader = DataLoader(val_ds, batch_size=params.batch_size, num_workers=params.num_workers, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=params.batch_size, num_workers=params.num_workers, shuffle=False)

    return val_loader, test_loader
