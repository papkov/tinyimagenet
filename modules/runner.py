import os
from dataclasses import dataclass
from logging import Logger
from pathlib import Path
from typing import Any, List, Optional, Tuple, Union

import numpy as np
import torch

# from torchvision import prototype as P
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch.nn.modules import loss
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import modules
from modules import models
from modules.meter import Meter
from modules.pytorch_typing import *


def train(
    model: Model,
    device: Device,
    loader: DataLoader,
    optimizer: Optimizer,
    loss_function: Criterion,
    epoch: int,
    log: Logger,
    writer: Optional[SummaryWriter] = None,
    scheduler: Optional[Scheduler] = None,
) -> Tuple[float, float]:
    """
    Training loop
    :param model: PyTorch model to test
    :param device: torch.device or str, where to perform computations
    :param loader: PyTorch DataLoader over test dataset
    :param optimizer: PyTorch Optimizer bounded with model
    :param loss_function: criterion
    :param epoch: epoch id
    :param writer: tensorboard SummaryWriter
    :param log: Logger
    :param scheduler: optional PyTorch Scheduler
    :return: tuple(train loss, train accuracy)
    """
    model.train()
    model.to(device)

    meter_loss = Meter("loss")
    meter_corr = Meter("acc")

    tqdm_loader = tqdm(loader, desc=f"train epoch {epoch:03d}")
    for batch_idx, batch_data in enumerate(tqdm_loader):
        data, target = batch_data.images.to(device), batch_data.labels.to(device)
        optimizer.zero_grad()

        output = model(data)

        loss = loss_function(output, target)
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        pred = output.argmax(dim=1, keepdim=True)
        # Display training status
        meter_loss.add(loss.item())
        meter_corr.add(pred.eq(target.view_as(pred)).sum().item())
        tqdm_loader.set_postfix(
            {
                "loss": meter_loss.avg,
                "acc": 100 * meter_corr.avg / loader.batch_size,
                "lr": scheduler.get_lr(),
            }
        )

    # Log in file and tensorboard
    acc = 100.0 * meter_corr.sum / len(loader.dataset)
    log.info(
        "Train Epoch: {} [ ({:.0f}%)]\tLoss: {:.6f}".format(epoch, acc, meter_loss.avg)
    )
    if writer is not None:
        writer.add_scalar("train_loss", loss.item(), global_step=epoch)
        writer.add_scalar("train_acc", acc, global_step=epoch)

    return meter_loss.avg, acc


def test(
    model: Model,
    device: Device,
    loader: DataLoader,
    loss_function: Criterion,
    epoch: int,
    log: Logger,
    writer: Optional[SummaryWriter] = None,
) -> Tuple[float, float, np.ndarray]:
    """
    Test loop
    :param model: PyTorch model to test
    :param device: torch.device or str, where to perform computations
    :param loader: PyTorch DataLoader over test dataset
    :param loss_function: criterion
    :param epoch: epoch id
    :param writer: tensorboard SummaryWriter
    :param log: Logger
    :return: tuple(test loss, test accuracy, outputs)
    """
    model.eval()
    model.to(device)
    test_loss = 0.0
    correct = 0
    outputs = []
    with torch.no_grad():
        for idx, batch_data in enumerate(tqdm(loader, desc=f"test epoch {epoch:03d}")):
            data, target = batch_data.images.to(device), batch_data.labels.to(device)
            output = model(data)
            test_loss += loss_function(output, target).sum().item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            outputs.append(output.detach().cpu().numpy())

    test_loss /= len(loader)
    acc = 100.0 * correct / len(loader.dataset)
    log.info(
        "Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)".format(
            test_loss,
            correct,
            len(loader.dataset),
            acc,
        )
    )
    if writer is not None:
        writer.add_scalar("test_loss", test_loss, global_step=epoch)
        writer.add_scalar("test_acc", acc, global_step=epoch)

    return test_loss, acc, np.concatenate(outputs)


class Runner:
    def __init__(
        self,
        cfg: DictConfig,
        log: Logger,
        train_loader: DataLoader,
        test_loader: DataLoader,
    ) -> None:
        """
        Orchestrates training process by config
        :param cfg: configuration file in omegaconf format from hydra
        :param log: Logger instance
        :param train_loader: PyTorch DataLoader over training set
        :param test_loader: PyTorch DataLoader over validation set
        """
        self.log = log
        self.cfg = cfg
        self.train_loader = train_loader
        self.test_loader = test_loader

        # Set device
        self.device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        self.log.info(f"Using device={self.device}")

        self.model = instantiate(self.cfg.model)
        self.model = self.model.to(self.device)
        self.log.info(f"Model: {self.model}")

        self.optimizer = instantiate(self.cfg.optimizer, self.model.parameters())
        self.log.info(f"Optimizer: {self.optimizer}")

        # Set scheduler
        self.scheduler = None
        if "scheduler" in self.cfg:
            self.scheduler = instantiate(self.cfg.scheduler, self.optimizer)
        if self.scheduler is None:
            T_max = self.cfg.train.epochs * len(self.train_loader)
            self.log.info(
                f"Scheduler not specified. Use default CosineScheduler with T_max={T_max}"
            )
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=T_max
            )
        self.log.info(f"Scheduler: {self.scheduler}")

        # Set loss function
        self.loss_function = loss.CrossEntropyLoss()

    def fit(self) -> None:
        # Paths
        results_root = Path(os.getcwd())
        checkpoint_path = results_root / self.cfg.results.checkpoints.root
        checkpoint_path /= f"{self.cfg.results.checkpoints.name}.pth"
        # Meters
        meters = {
            m: Meter(m) for m in ["train_loss", "val_loss", "train_acc", "val_acc"]
        }
        writer = SummaryWriter(results_root / self.cfg.results.checkpoints.tag)
        # Training loop
        for epoch in range(self.cfg.train.epochs):
            try:
                train_loss, train_acc = train(
                    self.model,
                    self.device,
                    self.train_loader,
                    self.optimizer,
                    self.loss_function,
                    epoch,
                    self.log,
                    writer,
                    self.scheduler,
                )
                val_loss, val_acc, val_outputs = test(
                    self.model,
                    self.device,
                    self.test_loader,
                    self.loss_function,
                    epoch,
                    self.log,
                    writer,
                )
                # Meters
                meters["train_loss"].add(train_loss)
                meters["val_loss"].add(val_loss)
                meters["train_acc"].add(train_acc)
                meters["val_acc"].add(val_acc)

                # Checkpoint
                if meters[self.cfg.train.monitor].is_best():
                    self.log.info(f"Save the best model to {checkpoint_path}")
                    torch.save(self.model.state_dict(), checkpoint_path)

            except KeyboardInterrupt:
                self.log.info("Training interrupted")
                break
        writer.close()
