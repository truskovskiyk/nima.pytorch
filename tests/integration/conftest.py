import torch
import os
import pytest
from PIL import Image
from pathlib import Path
from click.testing import CliRunner
from nima.model import create_model


@pytest.fixture
def data_dir() -> Path:
    return Path('tests/data')


@pytest.fixture
def model_type() -> str:
    return 'resnet18'


@pytest.yield_fixture
def state_dict_path(data_dir: Path, model_type: str):
    model = create_model(model_type=model_type, drop_out=0.0)

    best_state = {"state_dict": model.state_dict(),
                  "model_type": model_type,
                  "epoch": 100,
                  'best_loss': 0.0}

    best_state_path = data_dir / 'tmp_test_best_state_path.pth'
    torch.save(best_state, best_state_path)
    yield best_state_path

    os.remove(best_state_path)


@pytest.yield_fixture
def image_path(data_dir: Path):
    im = Image.new('RGB', (1024, 1024))

    image_path = data_dir / 'tmp_test_image.jpg'
    im.save(image_path)
    yield image_path

    os.remove(image_path)



@pytest.fixture
def cli_runner() -> CliRunner:
    runner = CliRunner()
    return runner
