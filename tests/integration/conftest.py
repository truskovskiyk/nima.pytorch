import os
from pathlib import Path
import shutil
import pandas as pd
import pytest
import torch
from click.testing import CliRunner
from PIL import Image
import io

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
def image_file_obj(image_path: Image.Image):
    byte_io = io.BytesIO()
    Image.open(image_path).save(byte_io, 'JPEG')
    byte_io.seek(0)

    yield byte_io

    byte_io.close()


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


@pytest.yield_fixture
def experiment_dir(path_to_save_result: Path):
    temp_experiment_dir = path_to_save_result / 'temp_exp'
    temp_experiment_dir.mkdir(parents=True, exist_ok=False)
    yield temp_experiment_dir
    shutil.rmtree(temp_experiment_dir)

@pytest.yield_fixture
def folder_with_csv(path_to_save_result: Path):
    df = pd.DataFrame({'image_id': ['tmp_test_image'],
     'score10': [5],
     'score11': [1],
     'score2': [0],
     'score3': [1],
     'score4': [5],
     'score5': [17],
     'score6': [38],
     'score7': [36],
     'score8': [15],
     'score9': [6]})

    df.to_csv(path_to_save_result / 'train.csv', index=False)
    df.to_csv(path_to_save_result / 'val.csv', index=False)
    df.to_csv(path_to_save_result / 'test.csv', index=False)

    yield path_to_save_result

    os.remove(path_to_save_result / 'train.csv')
    os.remove(path_to_save_result / 'val.csv')
    os.remove(path_to_save_result / 'test.csv')


@pytest.fixture
def cli_runner() -> CliRunner:
    runner = CliRunner()
    return runner
