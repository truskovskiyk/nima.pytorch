import logging
from pathlib import Path

import click

from nima.clean_dataset import clean_and_split
from nima.trainer import Trainer


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
@click.option("--num_workers", help="number of reading workers", default=32, type=int)
@click.option("--num_epoch", help="number of epoch", default=32, type=int)
@click.option("--init_lr", help="initial learning rate", default=0.0001, type=float)
@click.option("--drop_out", help="drop out", default=0.5, type=float)
@click.option("--optimizer_type", help="optimizer type", default="adam", type=str)
def train_model(
    path_to_save_csv: Path,
    path_to_images: Path,
    experiment_dir: Path,
    base_model: str,
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
        base_model=base_model,
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
@click.option("--path_to_model_weight", help="path to model weight .pth file", required=True)
@click.option("--path_to_image", help="image ", required=True)
def get_image_score(path_to_model_weight, path_to_image):
    # model = InferenceModel(path_to_model=path_to_model_weight)
    # result = model.predict_from_file(path_to_image)
    # click.echo(result)
    click.echo("Done!")


@click.command()
@click.option("--path_to_model_weight", help="path to model weight .pth file", required=True)
@click.option("--path_to_save_csv", help="where save train.csv|val.csv|test.csv", required=True)
@click.option("--path_to_images", help="images directory", required=True)
@click.option("--batch_size", help="batch size", required=True, type=int)
@click.option("--num_workers", help="number of reading workers", required=True, type=int)
def validate_model(path_to_model_weight, path_to_save_csv, path_to_images, batch_size, num_workers):
    # params = ValidateParams(path_to_save_csv=path_to_save_csv, path_to_model_weight=path_to_model_weight,
    #                         path_to_images=path_to_images, num_workers=num_workers, batch_size=batch_size)
    # val_loss, test_loss = start_check_model(params)
    # click.echo(f"val_loss = {val_loss}; test_loss = {test_loss}")
    click.echo("Done!")


@click.command()
@click.option("--path_to_model_weight", help="path to model weight .pth file", required=True)
@click.option("--path_to_save_csv", help="where save train.csv|val.csv|test.csv", required=True)
@click.option("--path_to_images", help="images directory", required=True)
@click.option("--batch_size", help="batch size", required=True, type=int)
@click.option("--num_workers", help="number of reading workers", required=True, type=int)
def run_web_app(path_to_model_weight, path_to_save_csv, path_to_images, batch_size, num_workers):
    click.echo("Done!")


cli.add_command(prepare_dataset)
cli.add_command(train_model)
cli.add_command(validate_model)
cli.add_command(get_image_score)
cli.add_command(run_web_app)

if __name__ == "__main__":
    init_logging()
    cli()
