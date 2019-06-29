import ast
from pathlib import Path

import torch
from click.testing import CliRunner

from nima.cli import get_image_score, prepare_dataset, train_model, validate_model


class TestCli:
    def test_get_image_score(self, cli_runner: CliRunner, state_dict_path: Path, image_path: Path):
        result = cli_runner.invoke(
            get_image_score, f"--path_to_model_state {state_dict_path} --path_to_image {image_path}"
        )
        res = ast.literal_eval(result.output)

        assert "mean_score" in res
        assert "std_score" in res
        assert "scores" in res
        assert len(res["scores"]) == 10
        assert result.exit_code == 0

    def test_prepare_dataset(
        self, cli_runner: CliRunner, ava_csv_path: Path, path_to_save_result: Path, images_path: Path
    ):

        result = cli_runner.invoke(
            prepare_dataset,
            f"--path_to_ava_txt {ava_csv_path} "
            f"--path_to_save_csv {path_to_save_result} "
            f"--path_to_images {images_path}",
        )
        assert (path_to_save_result / "train.csv").exists()
        assert (path_to_save_result / "val.csv").exists()
        assert (path_to_save_result / "test.csv").exists()
        assert result.exit_code == 0

    def test_train_model(self, cli_runner: CliRunner, images_path: Path, experiment_dir: Path, folder_with_csv: Path):
        assert not (experiment_dir / "logs").exists()
        assert not (experiment_dir / "best_state.pth").exists()

        result = cli_runner.invoke(
            train_model,
            f"--path_to_save_csv {folder_with_csv} "
            f"--path_to_images {images_path} "
            f"--experiment_dir {experiment_dir} "
            f"--batch_size 1 --num_epoch 5",
        )

        assert result.exit_code == 0
        assert (experiment_dir / "logs").exists()
        assert (experiment_dir / "best_state.pth").exists()

        best_state = torch.load(experiment_dir / "best_state.pth")

        assert "state_dict" in best_state
        assert "model_type" in best_state
        assert "epoch" in best_state
        assert "best_loss" in best_state

    def test_validate_model(
        self, cli_runner: CliRunner, images_path: Path, state_dict_path: Path, folder_with_csv: Path
    ):
        result = cli_runner.invoke(
            validate_model,
            f"--path_to_model_state {state_dict_path} "
            f"--path_to_save_csv {folder_with_csv} "
            f"--path_to_images {images_path}",
        )
        assert result.exit_code == 0
