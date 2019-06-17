import os
from pathlib import Path
import shutil

import pytest
import torch
from click.testing import CliRunner
from PIL import Image

from nima.model import create_model


@pytest.fixture
def data_dir() -> Path:
    return Path("tests/data")


@pytest.fixture
def model_type() -> str:
    return "resnet18"


@pytest.yield_fixture
def state_dict_path(data_dir: Path, model_type: str):
    model = create_model(model_type=model_type, drop_out=0.0)

    best_state = {"state_dict": model.state_dict(), "model_type": model_type, "epoch": 100, "best_loss": 0.0}

    best_state_path = data_dir / "tmp_test_best_state_path.pth"
    torch.save(best_state, best_state_path)
    yield best_state_path

    os.remove(best_state_path)


@pytest.yield_fixture
def image_path(data_dir: Path):
    im = Image.new("RGB", (1024, 1024))

    image_path = data_dir / "tmp_test_image.jpg"
    im.save(image_path)
    yield image_path

    os.remove(image_path)

@pytest.yield_fixture
def images_path(image_path: Path) -> Path:
    return image_path.parents[0]


@pytest.yield_fixture
def ava_csv_path(data_dir: Path):
    data = [
        "1 tmp_test_image 0 1 5 17 38 36 15 6 5 1 1 22 1396\n",
        "1 tmp_test_image 0 1 5 17 38 36 15 6 5 1 1 22 1396\n",
        "1 tmp_test_image 0 1 5 17 38 36 15 6 5 1 1 22 1396\n",
        "1 tmp_test_image 0 1 5 17 38 36 15 6 5 1 1 22 1396\n",
        "1 tmp_test_image 0 1 5 17 38 36 15 6 5 1 1 22 1396\n",
        "1 tmp_test_image 0 1 5 17 38 36 15 6 5 1 1 22 1396\n",
        "1 tmp_test_image 0 1 5 17 38 36 15 6 5 1 1 22 1396\n",
        "1 tmp_test_image 0 1 5 17 38 36 15 6 5 1 1 22 1396\n",
        "1 tmp_test_image 0 1 5 17 38 36 15 6 5 1 1 22 1396\n",
        "1 tmp_test_image 0 1 5 17 38 36 15 6 5 1 1 22 1396\n"
    ]

    temp_ava_path = data_dir / 'tmp_test_ava.csv'
    with open(temp_ava_path, 'w') as f:
        f.writelines(data)
    yield temp_ava_path
    os.remove(temp_ava_path)


@pytest.yield_fixture
def path_to_save_result(data_dir: Path):
    temp_dir_with_results = data_dir / 'temp_results'
    temp_dir_with_results.mkdir(parents=True, exist_ok=False)
    yield temp_dir_with_results
    shutil.rmtree(temp_dir_with_results)


@pytest.fixture
def cli_runner() -> CliRunner:
    runner = CliRunner()
    return runner
