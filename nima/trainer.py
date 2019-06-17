import logging
from pathlib import Path
from typing import Tuple

import torch
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from nima.common import AverageMeter, Transform
from nima.dataset import AVADataset
from nima.emd_loss import EDMLoss
from nima.model import create_model

logger = logging.getLogger(__file__)


def get_dataloaders(
    path_to_save_csv: Path, path_to_images: Path, batch_size: int, num_workers: int
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    transform = Transform()

    train_ds = AVADataset(path_to_save_csv / "train.csv", path_to_images, transform.train_transform)
    val_ds = AVADataset(path_to_save_csv / "val.csv", path_to_images, transform.val_transform)
    test_ds = AVADataset(path_to_save_csv / "test.csv", path_to_images, transform.val_transform)

    train_ds = Subset(train_ds, indices=list(range(100)))
    val_ds = Subset(val_ds, indices=list(range(100)))
    test_ds = Subset(test_ds, indices=list(range(100)))

    train_loader = DataLoader(train_ds, batch_size=batch_size, num_workers=num_workers, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    test_ds = DataLoader(test_ds, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    return train_loader, val_loader, test_ds


def validate_and_test(
    path_to_save_csv: Path,
    path_to_images: Path,
    batch_size: int,
    num_workers: int,
    drop_out: float,
    path_to_model_state: Path,
) -> None:
    _, val_loader, test_loader = get_dataloaders(
        path_to_save_csv=path_to_save_csv, path_to_images=path_to_images, batch_size=batch_size, num_workers=num_workers
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = EDMLoss().to(device)

    best_state = torch.load(path_to_model_state)

    model = create_model(best_state["model_type"], drop_out=drop_out).to(device)
    model.load_state_dict(best_state["state_dict"])

    model.eval()
    validate_losses = AverageMeter()

    with torch.no_grad():
        for (x, y) in tqdm(val_loader):
            x = x.to(device)
            y = y.to(device)
            y_pred = model(x)
            loss = criterion(p_target=y, p_estimate=y_pred)
            validate_losses.update(loss.item(), x.size(0))

    test_losses = AverageMeter()
    with torch.no_grad():
        for (x, y) in tqdm(test_loader):
            x = x.to(device)
            y = y.to(device)
            y_pred = model(x)
            loss = criterion(p_target=y, p_estimate=y_pred)
            test_losses.update(loss.item(), x.size(0))
    logger.info(f"val loss {validate_losses.avg}; test loss {test_losses.avg}")


class Trainer:
    def __init__(
        self,
        *,
        path_to_save_csv: Path,
        path_to_images: Path,
        num_epoch: int,
        model_type: str,
        num_workers: int,
        batch_size: int,
        init_lr: float,
        experiment_dir: Path,
        drop_out: float,
        optimizer_type: str,
    ):

        train_loader, val_loader, _ = get_dataloaders(
            path_to_save_csv=path_to_save_csv,
            path_to_images=path_to_images,
            batch_size=batch_size,
            num_workers=num_workers,
        )
        self.train_loader = train_loader
        self.val_loader = val_loader

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = create_model(model_type, drop_out=drop_out).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=init_lr)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=self.optimizer, mode="min", patience=5)
        self.criterion = EDMLoss().to(self.device)
        self.model_type = model_type

        experiment_dir.mkdir(exist_ok=True, parents=True)
        self.experiment_dir = experiment_dir
        self.writer = SummaryWriter(str(experiment_dir / "logs"))
        self.num_epoch = num_epoch
        self.global_train_step = 0
        self.global_val_step = 0

    def train_model(self):
        best_loss = float("inf")
        best_state = None
        for e in range(1, self.num_epoch + 1):
            train_loss = self.train()
            val_loss = self.validate()
            self.scheduler.step(metrics=val_loss)

            self.writer.add_scalar("train/loss", train_loss, global_step=e)
            self.writer.add_scalar("val/loss", val_loss, global_step=e)

            if best_state is None or val_loss < best_loss:
                best_state = {"state_dict": self.model.state_dict(), "model_type": self.model_type, "epoch": e}
                torch.save(best_state, self.experiment_dir / "best_state.pth")

    def train(self):
        self.model.train()
        train_losses = AverageMeter()
        for (x, y) in tqdm(self.train_loader):
            x = x.to(self.device)
            y = y.to(self.device)
            y_pred = self.model(x)
            loss = self.criterion(p_target=y, p_estimate=y_pred)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            train_losses.update(loss.item(), x.size(0))

            self.writer.add_scalar("train/current_loss", train_losses.val, self.global_train_step)
            self.writer.add_scalar("train/avg_loss", train_losses.avg, self.global_train_step)
            self.global_train_step += 1

        return train_losses.avg

    def validate(self):
        self.model.eval()
        validate_losses = AverageMeter()

        with torch.no_grad():
            for (x, y) in tqdm(self.val_loader):
                x = x.to(self.device)
                y = y.to(self.device)
                y_pred = self.model(x)
                loss = self.criterion(p_target=y, p_estimate=y_pred)
                validate_losses.update(loss.item(), x.size(0))

                self.writer.add_scalar("val/current_loss", validate_losses.val, self.global_val_step)
                self.writer.add_scalar("val/avg_loss", validate_losses.avg, self.global_val_step)
                self.global_val_step += 1

        return validate_losses.avg
