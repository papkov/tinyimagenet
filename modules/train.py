import os
import sys
from pathlib import Path
import types

import logging
import hydra
from omegaconf import DictConfig

import pandas as pd
import numpy as np

import torch
from torchvision import transforms
from torchvision import models
from torch.nn.modules import loss
from torch import optim
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
from tqdm.auto import tqdm

from dataset import TinyImagenetDataset, DatasetItem


def train(model, device, train_loader, optimizer, loss_function, epoch, writer, log):
    model.train()
    model.to(device)
    correct = 0

    for batch_idx, batch_data in enumerate(tqdm(train_loader)):
        data, target = batch_data.images.to(device), batch_data.labels.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = loss_function(output, target)
        loss.backward()
        optimizer.step()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
    log.info(
        "Train Epoch: {} [ ({:.0f}%)]\tLoss: {:.6f}".format(
            epoch, 100.0 * correct / len(train_loader.dataset), loss.item()
        )
    )
    writer.add_scalar("train_loss_plot", loss.item(), global_step=epoch)
    writer.add_scalar(
        "train_accuracy_plot",
        100.0 * correct / len(train_loader.dataset),
        global_step=epoch,
    )


def test(model, device, test_loader, loss_function, epoch, writer, log):
    model.eval()
    model.to(device)
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for idx, batch_data in enumerate(tqdm(test_loader)):
            data, target = batch_data.images.to(device), batch_data.labels.to(device)
            output = model(data)
            test_loss += loss_function(output, target).sum().item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    log.info(
        "Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)".format(
            test_loss,
            correct,
            len(test_loader.dataset),
            100.0 * correct / len(test_loader.dataset),
        )
    )
    writer.add_scalar("test_loss_plot", test_loss, global_step=epoch)
    writer.add_scalar(
        "test_accuracy_plot",
        100.0 * correct / len(test_loader.dataset),
        global_step=epoch,
    )


@hydra.main(config_path="../config/config.yaml")
def main(cfg: DictConfig):
    """
    The main training function
    :param cfg: hydra config passed through the decorator
    :return: None
    """
    # Setup logging and show config (hydra takes care of naming)
    log = logging.getLogger(__name__)
    log.info(f'Config:\n{cfg.pretty()}')

    # Data
    # Specify data paths from config
    data_root = Path(cfg.data.root)
    train_path = data_root / cfg.data.train
    val_path = data_root / cfg.data.val

    # Check if dataset is available
    log.info(f'Looking for dataset in {str(data_root)}')
    if not data_root.exists():
        log.error("Dataset not found. Terminating. See README.md for downloading details.")
        return

    # Specify results paths from config
    results_root = Path(os.getcwd())  # hydra handles results folder
    checkpoint_path = results_root / cfg.results.checkpoints.root
    tensorboard_tag = cfg.results.checkpoints.tag
    checkpoint_name = f"{cfg.results.checkpoints.name}.pth"
    # And make respective dirs if necessary
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    log.info(f"Write checkpoints to {checkpoint_path}/{checkpoint_name}")

    # Training
    # Dataset
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize([0.4802, 0.4481, 0.3975], [0.2302, 0.2265, 0.2262]),
        ]
    )

    train_dataset = TinyImagenetDataset(train_path, cfg, transform)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        collate_fn=DatasetItem.collate,
        num_workers=cfg.train.num_workers,
    )
    log.info(f"Created training dataset ({len(train_dataset)}) and loader ({len(train_loader)})")

    test_dataset = TinyImagenetDataset(val_path, cfg, transform)
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=cfg.train.batch_size,
        shuffle=False,
        collate_fn=DatasetItem.collate,
        num_workers=cfg.train.num_workers,
    )
    log.info(f"Created validation dataset ({len(test_dataset)}) and loader ({len(test_loader)})")

    # Set device
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    log.info(f'Using device={device}')

    # Set model
    # TODO: only resnets supported here, because of `fc` layer
    available_models = [k for k, v in models.__dict__.items()
                        if isinstance(v, types.FunctionType) and 'resnet' in k]
    try:
        model = eval(f"models.{cfg.model.arch}")()
    except AttributeError as e:
        log.error(f"Architecture {cfg.model.arch} not supported. "
                  f"Select one of the following: {','.join(available_models)}")
        log.error(e)
        return
    log.info(f"Created model {cfg.model.arch}")

    model.fc = nn.Linear(model.fc.in_features, cfg.data.classes)
    model = model.to(device)

    # Set optimizer
    available_optimizers = [k for k, v in torch.optim.__dict__.items() if callable(v)]
    try:
        optimizer = eval(f"optim.{cfg.optimizer.name}")(model.parameters(), **cfg.optimizer.parameters)
    except AttributeError as e:
        log.error(f"Optimizer {cfg.optimizer.name} not supported. "
                  f"Select one of the following: {', '.join(available_optimizers)}")
        log.error(e)
        return
    except TypeError as e:
        optional_parameters = eval(f"optim.{cfg.optimizer.name}").__init__.__code__.co_varnames[2:]
        log.error(f"Some optimizer parameters are wrong. "
                  f"Consider the following: {', '.join(optional_parameters)}")
        log.error(e)
        return
    set_parameters = ", ".join([f"{k}={v}" for k, v in cfg.optimizer.parameters.items()])
    log.info(f"Created optimizer {cfg.optimizer.name}({set_parameters})")

    # Set loss function
    loss_function = loss.CrossEntropyLoss()

    # Training loop
    for epoch in range(cfg.train.epochs):
        writer = SummaryWriter(results_root / tensorboard_tag)
        try:
            train(model, device, train_loader, optimizer, loss_function, epoch, writer, log)
            test(model, device, test_loader, loss_function, epoch, writer, log)
        except KeyboardInterrupt:
            log.info('Training interrupted')
            writer.close()
            break
        writer.close()


if __name__ == '__main__':
    main()
