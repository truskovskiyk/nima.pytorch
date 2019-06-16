import logging
from pathlib import Path

import click

from nima.clean_dataset import clean_and_split
from nima.trainer import Trainer, validate_and_test
from nima.inference_model import InferenceModel
from nima.common import set_up_seed
from nima.api import run_api

def init_logging() -> None:
    logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")


@click.group()
def cli():
    pass


@click.command()
@click.option("--path_to_ava_txt", help="origin AVA.txt file", required=True, type=Path)
@click.option("--path_to_save_csv", help="where save train.csv|val.csv|test.csv", required=True, type=Path)
@click.option("--path_to_images", help="images directory", required=True, type=Path)
@click.option("--train_size", help="train dataset size", default=0.8, type=float)
@click.option("--num_workers", help="num workers for parallel processing", default=64, type=int)
def prepare_dataset(
        path_to_ava_txt: Path, path_to_save_csv: Path, path_to_images: Path, train_size: float, num_workers: int
):
    click.echo(f"Clean and split dataset to train|val|test in {num_workers} threads. It will takes several minutes")
    clean_and_split(
        path_to_ava_txt=path_to_ava_txt,
        path_to_save_csv=path_to_save_csv,
        path_to_images=path_to_images,
        train_size=train_size,
        num_workers=num_workers,
    )
    click.echo("Done!")


@click.command()
@click.option("--path_to_save_csv", help="where save train.csv|val.csv|test.csv", required=True, type=Path)
@click.option("--path_to_images", help="images directory", required=True, type=Path)
@click.option("--experiment_dir", help="directory name to save all logs and weight", required=True, type=Path)
@click.option("--model_type", help="res net model type", default="resnet18", type=str)
@click.option("--batch_size", help="batch size", default=128, type=int)
@click.option("--num_workers", help="number of reading workers", default=16, type=int)
@click.option("--num_epoch", help="number of epoch", default=32, type=int)
@click.option("--init_lr", help="initial learning rate", default=0.0001, type=float)
@click.option("--drop_out", help="drop out", default=0.5, type=float)
@click.option("--optimizer_type", help="optimizer type", default="adam", type=str)
def train_model(
        path_to_save_csv: Path,
        path_to_images: Path,
        experiment_dir: Path,
        model_type: str,
        batch_size: int,
        num_workers: int,
        num_epoch: int,
        init_lr: float,
        drop_out: float,
        optimizer_type: str,
):
    click.echo("Train and validate model")
    trainer = Trainer(
        path_to_save_csv=path_to_save_csv,
        path_to_images=path_to_images,
        experiment_dir=experiment_dir,
        model_type=model_type,
        batch_size=batch_size,
        num_workers=num_workers,
        num_epoch=num_epoch,
        init_lr=init_lr,
        drop_out=drop_out,
        optimizer_type=optimizer_type,
    )
    trainer.train_model()
    click.echo("Done!")


@click.command()
@click.option("--path_to_model_state", help="path to model weight .pth file", required=True, type=Path)
@click.option("--path_to_image", help="image ", required=True, type=Path)
def get_image_score(path_to_model_state, path_to_image):
    model = InferenceModel(path_to_model_state=path_to_model_state)
    result = model.predict_from_file(path_to_image)
    click.echo(result)
    click.echo("Done!")


@click.command()
@click.option("--path_to_model_state", help="path to model weight .pth file", required=True, type=Path)
@click.option("--path_to_save_csv", help="where save train.csv|val.csv|test.csv", required=True, type=Path)
@click.option("--path_to_images", help="images directory", required=True, type=Path)
@click.option("--batch_size", help="batch size", default=128, type=int)
@click.option("--num_workers", help="number of reading workers", default=16, type=int)
@click.option("--drop_out", help="drop out", default=0.0, type=float)
def validate_model(path_to_model_state, path_to_save_csv, path_to_images, batch_size, num_workers, drop_out):
    validate_and_test(path_to_model_state=path_to_model_state, path_to_save_csv=path_to_save_csv,
                      path_to_images=path_to_images, batch_size=batch_size, num_workers=num_workers, drop_out=drop_out)
    click.echo("Done!")


@click.command()
@click.option("--path_to_model_state", help="path to model weight .pth file", required=True, type=Path)
@click.option("--port", help="port for web app", default=8080, type=int)
@click.option("--host", help="host for web app", default='0.0.0.0', type=str)
def run_web_api(path_to_model_state: Path, port: int, host: str):
    run_api(path_to_model_state=path_to_model_state, port=port, host=host)
    click.echo("Done!")


def main():
    init_logging()
    set_up_seed()
    cli.add_command(prepare_dataset)
    cli.add_command(train_model)
    cli.add_command(validate_model)
    cli.add_command(get_image_score)
    cli.add_command(run_web_api)
    cli()


if __name__ == "__main__":
    main()
