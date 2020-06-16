import os
from dataclasses import dataclass
from logging import Logger
from pathlib import Path
from types import FunctionType
from typing import Any, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torchvision
from omegaconf import DictConfig
from torch.nn.modules import loss
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

import models
from pytorch_typing import Criterion, Device, Model, Optimizer, Parameters, Scheduler


def torch_model(
    arch: str,
    n_classes: int,
    pretrained: bool,
    log: Logger,
    module_name: str = "torchvision",
) -> Model:
    """
    PyTorch Model retriever, raises AttributeError if provided model does not exist
    and TypeError if kwargs do not match
    :param arch: architecture name from torchvision.models or
                 modules.models (function by this name should exist)
    :param n_classes: number of classes for the last dense layer
    :param pretrained: is use pretrained model
                       (only valid for module_name='torchvision')
    :param log: Logger
    :param module_name: {torchvision, models}, module where to look for the function
    :return:
    """
    # Get module with models
    module = torchvision.models if module_name == "torchvision" else models
    # Get list of available architectures
    # TODO: only resnets supported here now, because of `fc` layer
    available = [
        k
        for k, v in module.__dict__.items()
        if isinstance(v, FunctionType) and "resnet" in k
    ]
    try:
        if module_name == "torchvision":
            model = getattr(module, arch)(pretrained=pretrained)
            model.fc = nn.Linear(model.fc.in_features, n_classes)
        else:
            model = getattr(module, arch)(n_classes=n_classes)
    except AttributeError as e:
        log.error(
            f"Architecture {arch} not supported. "
            f"Select one of the following: {','.join(available)}"
        )
        log.error(e)
        raise
    log.info(
        f"Created model {module_name}.{arch}(pretrained={str(pretrained)}) "
        f"with {n_classes} outputs"
    )
    return model


def torch_optimizer(
    name: str, params: Parameters, log: Logger, **kwargs: Any
) -> Optimizer:
    """
    PyTorch Optimizer retriever,
    raises AttributeError if provided optimizer does not exist,
    raises TypeError if kwargs do not match
    :param name: name of the class inherited from PyTorch Optimizer base class
    :param params: model.parameters() or dict of parameter groups
    :param log: Logger
    :param kwargs: keyword args passed to the Optimizer initialization
    :return: Optimizer
    """
    available = [
        k for k, v in torch.optim.__dict__.items() if callable(v) and k != "Optimizer"
    ]
    try:
        optimizer = getattr(torch.optim, name)(params, **kwargs)
    except AttributeError as e:
        log.error(
            f"Optimizer {name} not supported. "
            f"Select one of the following: {', '.join(available)}"
        )
        log.error(e)
        raise
    except TypeError as e:
        opt_stub = getattr(torch.optim, name)
        optional_parameters = opt_stub.__init__.__code__.co_varnames[2:]
        log.error(
            f"Some optimizer parameters are wrong. "
            f"Consider the following: {', '.join(optional_parameters)}"
        )
        log.error(e)
        raise
    set_parameters = ", ".join([f"{k}={v}" for k, v in kwargs.items()])
    log.info(f"Created optimizer {name}({set_parameters})")
    return optimizer


def torch_scheduler(
    name: str, optimizer: DataLoader, log: Logger, **kwargs: Any
) -> Union[None, Scheduler]:
    """
    PyTorch Scheduler retriever,
    raises AttributeError if provided scheduler does not exist,
    raises TypeError if kwargs do not match (returns None if error was raised)
    :param name: name of the class inherited from PyTorch Scheduler base class
    :param optimizer: PyTorch Optimizer bounded with model
    :param log: Logger
    :param kwargs: keyword args passed to the Scheduler initialization
    :return: Scheduler or None if an expected error was raised
    """
    available = [k for k, v in torch.optim.__dict__.items() if callable(v)][5:]
    try:
        scheduler = getattr(torch.optim.lr_scheduler, name)(optimizer, **kwargs)
    except AttributeError as e:
        log.error(
            f"Scheduler {name} not supported. "
            f"Select one of the following: {', '.join(available)}"
        )
        log.error(e)
        return None
    except TypeError as e:
        optional_parameters = getattr(
            torch.optim.lr_scheduler, name
        ).__init__.__code__.co_varnames[2:]
        log.error(
            f"Some scheduler parameters are wrong. "
            f"Consider the following: {', '.join(optional_parameters)}"
        )
        log.error(e)
        return None
    set_parameters = ", ".join([f"{k}={v}" for k, v in kwargs.items()])
    log.info(f"Created scheduler {name}({set_parameters})")
    return scheduler


def train(
    model: Model,
    device: Device,
    loader: DataLoader,
    optimizer: Optimizer,
    loss_function: Criterion,
    epoch: int,
    writer: SummaryWriter,
    log: Logger,
    scheduler: Optional[Union[Scheduler, None]] = None,
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

    batch_size = len(loader.dataset) / len(loader)
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
                "acc": 100 * meter_corr.avg / batch_size,
                "lr": scheduler.get_lr(),
            }
        )

    # Log in file and tensorboard
    acc = 100.0 * meter_corr.sum / len(loader.dataset)
    log.info(
        "Train Epoch: {} [ ({:.0f}%)]\tLoss: {:.6f}".format(epoch, acc, loss.item())
    )
    writer.add_scalar("train_loss", loss.item(), global_step=epoch)
    writer.add_scalar("train_accuracy", acc, global_step=epoch)

    return meter_loss.avg, acc


def test(
    model: Model,
    device: Device,
    loader: DataLoader,
    loss_function: Criterion,
    epoch: int,
    writer: SummaryWriter,
    log: Logger,
) -> Tuple[float, float]:
    """
    Test loop
    :param model: PyTorch model to test
    :param device: torch.device or str, where to perform computations
    :param loader: PyTorch DataLoader over test dataset
    :param loss_function: criterion
    :param epoch: epoch id
    :param writer: tensorboard SummaryWriter
    :param log: Logger
    :return: tuple(test loss, test accuracy)
    """
    model.eval()
    model.to(device)
    test_loss = 0.0
    correct = 0
    with torch.no_grad():
        for idx, batch_data in enumerate(tqdm(loader, desc=f"test epoch {epoch:03d}")):
            data, target = batch_data.images.to(device), batch_data.labels.to(device)
            output = model(data)
            test_loss += loss_function(output, target).sum().item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(loader.dataset)
    acc = 100.0 * correct / len(loader.dataset)
    log.info(
        "Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)".format(
            test_loss, correct, len(loader.dataset), acc,
        )
    )
    writer.add_scalar("test_loss", test_loss, global_step=epoch)
    writer.add_scalar(
        "test_accuracy", acc, global_step=epoch,
    )

    return test_loss, acc


@dataclass()
class Meter:
    name: str
    history: List[float]
    sum: float = 0
    avg: float = 0
    last: float = 0
    min: float = np.inf
    max: float = -np.inf
    extremum: str = ""
    monitor_min: bool = False

    def __init__(self, name: str) -> None:
        """
        Stores all the incremented elements, their sum and average
        Meter with name {}_loss will monitor min values in history
        :param name: {train, val, test}_{loss, acc, ...} for saving and monitoring
        """
        self.name = name
        self.monitor_min = name.endswith("loss")
        self.reset()

    def reset(self) -> None:
        """
        Restore default values
        :return:
        """
        self.history = []
        self.sum = 0
        self.avg = 0
        self.last = 0
        self.min = np.inf
        self.max = -np.inf
        self.extremum = ""

    def add(self, value: Union[int, float]) -> None:
        """
        Add a value in history and check extremums
        :param value: monitored value
        :return:
        """
        self.last = value
        self.extremum = ""

        if value < self.min:
            self.min = value
            self.extremum = "min"
        if value > self.max:
            self.max = value
            self.extremum = "max"

        self.history.append(value)
        self.sum += value
        self.avg = self.sum / len(self.history)

    def is_best(self) -> bool:
        """
        Check if the last epoch was the best according to the meter
        :return: whether last value added was the best
        """
        is_best = (self.monitor_min and self.extremum == "min") or (
            (not self.monitor_min) and self.extremum == "max"
        )
        return is_best


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
        :param Log: Logger instance
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

        # Set model
        self.model = torch_model(
            self.cfg.model.arch,
            self.cfg.data.classes,
            self.cfg.model.pretrained,
            log,
            module_name=self.cfg.model.module,
        )
        self.model = self.model.to(self.device)

        # Set optimizer
        parameters = (
            self.cfg.optimizer.parameters if "parameters" in self.cfg.optimizer else {}
        )  # keep defaults
        self.optimizer = torch_optimizer(
            self.cfg.optimizer.name, self.model.parameters(), self.log, **parameters
        )

        # Set scheduler
        self.scheduler = None
        if "scheduler" in self.cfg:
            parameters = (
                self.cfg.scheduler.parameters
                if "parameters" in self.cfg.scheduler
                else {}
            )  # keep defaults
            self.scheduler = torch_scheduler(
                self.cfg.scheduler.name, self.optimizer, self.log, **parameters
            )
        if self.scheduler is None:
            self.log.info("Scheduler not specified. Proceed without")

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
                    writer,
                    self.log,
                    self.scheduler,
                )
                val_loss, val_acc = test(
                    self.model,
                    self.device,
                    self.test_loader,
                    self.loss_function,
                    epoch,
                    writer,
                    self.log,
                )
                # Meters
                meters["train_loss"].add(train_loss)
                meters["val_loss"].add(val_loss)
                meters["train_acc"].add(train_acc)
                meters["val_acc"].add(val_acc)

                # Checkpoint
                if meters[self.cfg.train.monitor].is_best:
                    self.log.info(f"Save the best model to {checkpoint_path}")
                    torch.save(self.model.state_dict(), checkpoint_path)

            except KeyboardInterrupt:
                self.log.info("Training interrupted")
                break
        writer.close()
