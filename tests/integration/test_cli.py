import ast
from pathlib import Path

from click.testing import CliRunner

from nima.cli import get_image_score, prepare_dataset


def test_get_image_score(cli_runner: CliRunner, state_dict_path: Path, image_path: Path):
    result = cli_runner.invoke(
        get_image_score, ["--path_to_model_state", state_dict_path, "--path_to_image", image_path]
    )
    res = ast.literal_eval(result.output)
    assert "mean_score" in res
    assert "std_score" in res
    assert "scores" in res
    assert len(res["scores"]) == 10
    assert result.exit_code == 0


def test_prepare_dataset(cli_runner: CliRunner):
    result = cli_runner.invoke(
        prepare_dataset,
        ["--path_to_ava_txt", "test.csv", "--path_to_save_csv", "data", "--path_to_images", "path_to_imgsage"],
    )
    # print(result.exception)
    print(result.output)
    # print(result.exc_info[-1])
    # traceback.print_tb(result.exc_info[-1])
    # assert result.exit_code == 0
    pass


def test_train_model():
    # runner = CliRunner()
    # result = runner.invoke(prepare_dataset, ['--path_to_ava_txt', 'test.csv'])
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
