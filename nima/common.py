import os

import torch
import numpy as np
import requests
from torchvision import transforms

IMAGE_NET_MEAN = [0.485, 0.456, 0.406]
IMAGE_NET_STD = [0.229, 0.224, 0.225]


class Transform:
    def __init__(self):
        normalize = transforms.Normalize(mean=IMAGE_NET_MEAN, std=IMAGE_NET_STD)

        self._train_transform = transforms.Compose(
            [
                transforms.Resize((256, 256)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop((224, 224)),
                transforms.ToTensor(),
                normalize,
            ]
        )

        self._val_transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), normalize])

    @property
    def train_transform(self):
        return self._train_transform

    @property
    def val_transform(self):
        return self._val_transform


def get_mean_score(score):
    buckets = np.arange(1, 11)
    mu = (buckets * score).sum()
    return mu


def get_std_score(scores):
    si = np.arange(1, 11)
    mean = get_mean_score(scores)
    std = np.sqrt(np.sum(((si - mean) ** 2) * scores))
    return std


def download_file(url, local_filename, chunk_size=1024):
    if os.path.exists(local_filename):
        return local_filename
    r = requests.get(url, stream=True)
    with open(local_filename, "wb") as f:
        for chunk in r.iter_content(chunk_size=chunk_size):
            if chunk:
                f.write(chunk)
    return local_filename


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def set_up_seed(seed=42):
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)


def format_output(mean_score, std_score, prob):
    return {
        'mean_score': float(mean_score),
        'std_score': float(std_score),
        'scores': [float(x) for x in prob]
    }