import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models
from torch.nn.modules import loss
import types
from tqdm.auto import tqdm
from dataclasses import dataclass, replace
from omegaconf import DictConfig
from logging import Logger
from typing import Callable, List, Optional, Sequence, Union
from torch.utils.tensorboard import SummaryWriter
import os
from pathlib import Path


def torch_model(arch, n_classes, pretrained, log):
    # TODO: only resnets supported here, because of `fc` layer
    available = [k for k, v in models.__dict__.items()
                 if isinstance(v, types.FunctionType) and 'resnet' in k]
    try:
        model = getattr(models, arch)(pretrained=pretrained)
    except AttributeError as e:
        log.error(f"Architecture {arch} not supported. "
                  f"Select one of the following: {','.join(available)}")
        log.error(e)
        raise
    log.info(f"Created model {arch}")
    # Substitute the final FC layer
    model.fc = nn.Linear(model.fc.in_features, n_classes)
    log.info(f"Created model {arch}(pretrained={str(pretrained)}) with {n_classes} outputs")
    return model


def torch_optimizer(name, params, log, **kwargs):
    available = [k for k, v in torch.optim.__dict__.items() if callable(v) and k != 'Optimizer']
    try:
        optimizer = getattr(torch.optim, name)(params, **kwargs)
    except AttributeError as e:
        log.error(f"Optimizer {name} not supported. "
                  f"Select one of the following: {', '.join(available)}")
        log.error(e)
        raise
    except TypeError as e:
        optional_parameters = getattr(torch.optim, name).__init__.__code__.co_varnames[2:]
        log.error(f"Some optimizer parameters are wrong. "
                  f"Consider the following: {', '.join(optional_parameters)}")
        log.error(e)
        raise
    set_parameters = ", ".join([f"{k}={v}" for k, v in kwargs.items()])
    log.info(f"Created optimizer {name}({set_parameters})")
    return optimizer


def torch_scheduler(name, optimizer, log, **kwargs):
    available = [k for k, v in torch.optim.__dict__.items() if callable(v)][5:]
    try:
        scheduler = getattr(torch.optim.lr_scheduler, name)(optimizer, **kwargs)
    except AttributeError as e:
        log.error(f"Scheduler {name} not supported. "
                  f"Select one of the following: {', '.join(available)}")
        log.error(e)
        return None
    except TypeError as e:
        optional_parameters = getattr(torch.optim.lr_scheduler, name).__init__.__code__.co_varnames[2:]
        log.error(f"Some scheduler parameters are wrong. "
                  f"Consider the following: {', '.join(optional_parameters)}")
        log.error(e)
        return None
    set_parameters = ", ".join([f"{k}={v}" for k, v in kwargs.items()])
    log.info(f"Created scheduler {name}({set_parameters})")
    return scheduler


def train(model, device, train_loader, optimizer, loss_function, epoch, writer, log, scheduler=None):
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
        if scheduler is not None:
            scheduler.step()

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


@dataclass()
class Runner:
    cfg: DictConfig
    log: Logger
    train_loader: DataLoader
    test_loader: DataLoader
    model: torch.nn.Module
    optimizer: torch.optim.Optimizer
    scheduler: Union[torch.optim.lr_scheduler._LRScheduler, None]
    loss_function: torch.nn.Module
    device: torch.device

    def __init__(self, cfg: DictConfig, log: Logger, train_loader: DataLoader, test_loader: DataLoader):
        self.log = log
        self.cfg = cfg
        self.train_loader = train_loader
        self.test_loader = test_loader

        # Set device
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.log.info(f'Using device={self.device}')

        # Set model
        self.model = torch_model(self.cfg.model.arch, self.cfg.data.classes, self.cfg.model.pretrained, log)
        self.model = self.model.to(self.device)

        # Set optimizer
        parameters = self.cfg.optimizer.parameters if 'parameters' in self.cfg.optimizer else {}  # keep defaults
        self.optimizer = torch_optimizer(self.cfg.optimizer.name, self.model.parameters(), self.log, **parameters)

        # Set scheduler
        self.scheduler = None
        if 'scheduler' in self.cfg:
            parameters = self.cfg.scheduler.parameters if 'parameters' in self.cfg.scheduler else {}  # keep defaults
            self.scheduler = torch_scheduler(self.cfg.scheduler.name, self.optimizer, self.log, **parameters)
        if self.scheduler is None:
            self.log.info('Scheduler not specified. Proceed without')

        # Set loss function
        self.loss_function = loss.CrossEntropyLoss()

    def fit(self):
        # Training loop
        for epoch in range(self.cfg.train.epochs):
            writer = SummaryWriter(Path(os.getcwd()) / self.cfg.results.checkpoints.tag)
            try:
                train(self.model, self.device, self.train_loader, self.optimizer, self.loss_function, epoch,
                      writer, self.log, self.scheduler)
                test(self.model, self.device, self.test_loader, self.loss_function, epoch, writer, self.log)
            except KeyboardInterrupt:
                self.log.info('Training interrupted')
                writer.close()
                break
            writer.close()

