from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from nima.common import AverageMeter, Transform
from nima.dataset import AVADataset
from nima.emd_loss import EDMLoss
from nima.model import create_model


class Trainer:
    def __init__(
        self,
        path_to_save_csv: Path,
        path_to_images: Path,
        num_epoch: int,
        base_model: str,
        num_workers: int,
        batch_size: int,
        init_lr: float,
        experiment_dir: Path,
    ):

        use_gpu = torch.cuda.is_available()
        self.device = torch.device("cuda" if use_gpu else "cpu")

        transform = Transform()
        train_ds = AVADataset(path_to_save_csv / "train.csv", path_to_images, transform.train_transform)
        val_ds = AVADataset(path_to_save_csv / "val.csv", path_to_images, transform.val_transform)
        self.train_loader = DataLoader(train_ds, batch_size=batch_size, num_workers=num_workers, shuffle=True)
        self.val_loader = DataLoader(val_ds, batch_size=batch_size, num_workers=num_workers, shuffle=False)

        experiment_dir.mkdir(exist_ok=True)
        self.experiment_dir = experiment_dir
        self.writer = SummaryWriter(str(experiment_dir / "logs"))
        self.num_epoch = num_epoch
        self.global_train_step = 0
        self.global_val_step = 0

        self.model = create_model(base_model).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=init_lr)
        self.criterion = EDMLoss().to(self.device)

    def train_model(self):
        for e in range(1, self.num_epoch + 1):
            train_loss = self.train()
            val_loss = self.validate()

            self.writer.add_scalar("train/loss", train_loss, global_step=e)
            self.writer.add_scalar("val/loss", val_loss, global_step=e)

            torch.save(self.model.state_dict(), self.experiment_dir / f"epoch_{e}.pth")

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
