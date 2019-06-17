import ast
from pathlib import Path

from click.testing import CliRunner

from nima.cli import get_image_score, prepare_dataset, train_model


def test_get_image_score(cli_runner: CliRunner, state_dict_path: Path, image_path: Path):
    result = cli_runner.invoke(
        get_image_score, f"--path_to_model_state {state_dict_path} --path_to_image {image_path}"
    )
    res = ast.literal_eval(result.output)

    assert "mean_score" in res
    assert "std_score" in res
    assert "scores" in res
    assert len(res["scores"]) == 10
    assert result.exit_code == 0


def test_prepare_dataset(cli_runner: CliRunner, ava_csv_path: Path, path_to_save_result: Path, images_path: Path):

    result = cli_runner.invoke(
        prepare_dataset, f"--path_to_ava_txt {ava_csv_path} --path_to_save_csv {path_to_save_result} --path_to_images {images_path}"
    )
    assert (path_to_save_result / 'train.csv').exists()
    assert (path_to_save_result / 'val.csv').exists()
    assert (path_to_save_result / 'test.csv').exists()
    assert result.exit_code == 0



def test_train_model(cli_runner: CliRunner):
    result = cli_runner.invoke(train_model, ['--path_to_ava_txt', 'test.csv'])
    # print(result.exception)
    # print(result.output)
    # print(result.exc_info[-1])
    # traceback.print_tb(result.exc_info[-1])
    pass


def test_validate_model():
    # runner = CliRunner()
    # result = runner.invoke(prepare_dataset, ['--path_to_ava_txt', 'test.csv'])
    # print(result.exception)
    # print(result.output)
    # print(result.exc_info[-1])
    # traceback.print_tb(result.exc_info[-1])
    pass
