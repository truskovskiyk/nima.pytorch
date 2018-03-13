import torch
from torchvision.datasets.folder import default_loader

from decouple import config

from nima.model import NIMA
from nima.common import Transform, get_mean_score, get_std_score
from nima.common import download_file
from nima.inference.utils import format_output


class InferenceModel:
    @classmethod
    def create_model(cls):
        path_to_model = download_file(config('MODEL_URL'), config('MODEL_PATH'))
        return cls(path_to_model)

    def __init__(self, path_to_model):
        self.transform = Transform().val_transform
        self.model = NIMA(pretrained_base_model=False)
        state_dict = torch.load(path_to_model, map_location=lambda storage, loc: storage)
        self.model.load_state_dict(state_dict)
        self.model.eval()

    def predict_from_file(self, image_path):
        image = default_loader(image_path)
        return self.predict(image)

    def predict_from_pil_image(self, image):
        image = image.convert('RGB')
        return self.predict(image)

    def predict(self, image):
        image = self.transform(image)
        image = image.unsqueeze_(0)
        image = torch.autograd.Variable(image, volatile=True)
        prob = self.model(image).data.numpy()[0]

        mean_score = get_mean_score(prob)
        std_score = get_std_score(prob)

        return format_output(mean_score, std_score, prob)
