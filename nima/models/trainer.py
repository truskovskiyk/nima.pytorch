import os
from tqdm import tqdm
import torch

from torch.utils.data import DataLoader

from tensorboardX import SummaryWriter


from nima.model import NIMA
from nima.train.datasets import AVADataset
from nima.train.emd_loss import EDMLoss
from nima.common import Transform
from nima.train.utils import TrainParams, ValidateParams, AverageMeter

use_gpu = torch.cuda.is_available()
device = torch.device("cuda" if use_gpu else "cpu")

class Trainer:
    def __init__(self):
        pass

    def train(model, loader, optimizer, criterion, writer=None, global_step=None, name=None):
        model.train()
        train_losses = AverageMeter()
        for idx, (x, y) in enumerate(tqdm(loader)):
            x = x.to(device)
            y = y.to(device)
            y_pred = model(x)
            loss = criterion(p_target=y, p_estimate=y_pred)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_losses.update(loss.item(), x.size(0))

            if writer is not None:
                writer.add_scalar(f"{name}/train_loss.avg", train_losses.avg, global_step=global_step + idx)
        return train_losses.avg


    def validate(model, loader, criterion, writer=None, global_step=None, name=None):
        model.eval()
        validate_losses = AverageMeter()
        for idx, (x, y) in enumerate(tqdm(loader)):
            x = x.to(device)
            y = y.to(device)
            y_pred = model(x)
            loss = criterion(p_target=y, p_estimate=y_pred)
            validate_losses.update(loss.item(), x.size(0))

            if writer is not None:
                writer.add_scalar(f"{name}/val_loss.avg", validate_losses.avg, global_step=global_step + idx)
        return validate_losses.avg


def _create_train_data_part(params: TrainParams):
    train_csv_path = os.path.join(params.path_to_save_csv, 'train.csv')
    val_csv_path = os.path.join(params.path_to_save_csv, 'val.csv')

    transform = Transform()
    train_ds = AVADataset(train_csv_path, params.path_to_images, transform.train_transform)
    val_ds = AVADataset(val_csv_path, params.path_to_images, transform.val_transform)

    train_loader = DataLoader(train_ds, batch_size=params.batch_size, num_workers=params.num_workers, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=params.batch_size, num_workers=params.num_workers, shuffle=False)

    return train_loader, val_loader


def _create_val_data_part(params: TrainParams):
    val_csv_path = os.path.join(params.path_to_save_csv, 'val.csv')
    test_csv_path = os.path.join(params.path_to_save_csv, 'test.csv')

    transform = Transform()
    val_ds = AVADataset(val_csv_path, params.path_to_images, transform.val_transform)
    test_ds = AVADataset(test_csv_path, params.path_to_images, transform.val_transform)

    val_loader = DataLoader(val_ds, batch_size=params.batch_size, num_workers=params.num_workers, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=params.batch_size, num_workers=params.num_workers, shuffle=False)

    return val_loader, test_loader


def start_train(params: TrainParams):
    train_loader, val_loader = _create_train_data_part(params=params)
    model = NIMA()
    optimizer = torch.optim.Adam(model.parameters(), lr=params.init_lr)
    criterion = EDMLoss()
    model = model.to(device)
    criterion.to(device)

    writer = SummaryWriter(log_dir=os.path.join(params.experiment_dir_name, 'logs'))
    os.makedirs(params.experiment_dir_name, exist_ok=True)
    params.save_params(os.path.join(params.experiment_dir_name, 'params.json'))

    for e in range(params.num_epoch):
        train_loss = train(model=model, loader=train_loader, optimizer=optimizer, criterion=criterion,
                           writer=writer, global_step=len(train_loader.dataset) * e,
                           name=f"{params.experiment_dir_name}_by_batch")
        val_loss = validate(model=model, loader=val_loader, criterion=criterion,
                            writer=writer, global_step=len(train_loader.dataset) * e,
                            name=f"{params.experiment_dir_name}_by_batch")

        model_name = f"emd_loss_epoch_{e}_train_{train_loss}_{val_loss}.pth"
        torch.save(model.module.state_dict(), os.path.join(params.experiment_dir_name, model_name))
        writer.add_scalar(f"{params.experiment_dir_name}_by_epoch/train_loss", train_loss, global_step=e)
        writer.add_scalar(f"{params.experiment_dir_name}_by_epoch/val_loss", val_loss, global_step=e)

    writer.export_scalars_to_json(os.path.join(params.experiment_dir_name, 'all_scalars.json'))
    writer.close()


def start_check_model(params: ValidateParams):
    val_loader, test_loader = _create_val_data_part(params)
    model = NIMA()
    model.load_state_dict(torch.load(params.path_to_model_weight))
    criterion = EDMLoss()

    model = model.to(device)
    criterion.to(device)

    val_loss = validate(model=model, loader=val_loader, criterion=criterion)
    test_loss = validate(model=model, loader=test_loader, criterion=criterion)
    return val_loss, test_loss
