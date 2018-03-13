import json
from collections import namedtuple

_SCORE_FIRST_COLUMN = 2
_SCORE_LAST_COLUMN = 12
_TAG_FIRST_COLUMN = 1
_TAG_LAST_COLUMN = 4

SCORE_NAMES = [f'score{i}' for i in range(_SCORE_FIRST_COLUMN, _SCORE_LAST_COLUMN)]
TAG_NAMES = [f'tag{i}' for i in range(_TAG_FIRST_COLUMN, _TAG_LAST_COLUMN)]


class TrainParams(namedtuple('TrainParams', ['path_to_save_csv', 'path_to_images',
                                             'experiment_dir_name', 'batch_size',
                                             'num_workers', 'num_epoch', 'init_lr'])):
    def save_params(self, file_path: str):
        with open(file_path, 'w') as f:
            json.dump(self._asdict(), f)


class ValidateParams(namedtuple('TrainParams', ['path_to_save_csv', 'path_to_model_weight',
                                                'path_to_images', 'batch_size',
                                                'num_workers'])):
    pass


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
