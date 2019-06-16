import torch
from torchvision.datasets.folder import default_loader
from pathlib import Path
from decouple import config
from PIL.Image import Image

from nima.model import create_model
from nima.common import Transform, get_mean_score, get_std_score, download_file

from nima.common import format_output

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")


class InferenceModel:

    def __init__(self, path_to_model_state: Path):
        self.transform = Transform().val_transform
        model_state = torch.load(path_to_model_state, map_location=lambda storage, loc: storage)
        self.model = create_model(model_type=model_state['model_type'], drop_out=0)
        self.model.load_state_dict(model_state['state_dict'])
        self.model = self.model.to(device)
        self.model.eval()


    def predict_from_file(self, image_path: Path):
        image = default_loader(image_path)
        return self.predict(image)

    def predict_from_pil_image(self, image: Image):
        image = image.convert('RGB')
        return self.predict(image)

    @torch.no_grad()
    def predict(self, image):
        image = self.transform(image)
        image = image.unsqueeze_(0)
        image = image.to(device)
        prob = self.model(image).data.cpu().numpy()[0]

        mean_score = get_mean_score(prob)
        std_score = get_std_score(prob)

        return format_output(mean_score, std_score, prob)
