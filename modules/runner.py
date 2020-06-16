import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
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
import numpy as np
from functools import reduce
import models


def torch_model(arch, n_classes, pretrained, log, module_name='torchvision'):
    # Get module with models
    module = torchvision.models if module_name == 'torchvision' else models
    # Get list of available architectures
    # TODO: only resnets supported here now, because of `fc` layer
    available = [k for k, v in module.__dict__.items()
                 if isinstance(v, types.FunctionType) and 'resnet' in k]
    try:
        if module_name == 'torchvision':
            model = getattr(module, arch)(pretrained=pretrained)
            model.fc = nn.Linear(model.fc.in_features, n_classes)
        else:
            model = getattr(module, arch)(n_classes=n_classes)
    except AttributeError as e:
        log.error(f"Architecture {arch} not supported. "
                  f"Select one of the following: {','.join(available)}")
        log.error(e)
        raise
    log.info(f"Created model {module_name}.{arch}(pretrained={str(pretrained)}) with {n_classes} outputs")
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
    meter_loss = Meter('loss')
    meter_corr = Meter('acc')

    tqdm_loader = tqdm(train_loader, desc='train')
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
        tqdm_loader.set_postfix({'loss': meter_loss.avg,
                                 'acc': 100 * meter_corr.avg / len(batch_data.images)})

    # Log in file and tensorboard
    acc = 100.0 * meter_corr.sum / len(train_loader.dataset)
    log.info(
        "Train Epoch: {} [ ({:.0f}%)]\tLoss: {:.6f}".format(epoch, acc, loss.item())
    )
    writer.add_scalar("train_loss", loss.item(), global_step=epoch)
    writer.add_scalar("train_accuracy", acc, global_step=epoch)

    return meter_loss.avg, acc


def test(model, device, test_loader, loss_function, epoch, writer, log):
    model.eval()
    model.to(device)
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for idx, batch_data in enumerate(tqdm(test_loader, desc='valid')):
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
    writer.add_scalar("test_loss", test_loss, global_step=epoch)
    writer.add_scalar(
        "test_accuracy",
        100.0 * correct / len(test_loader.dataset),
        global_step=epoch,
    )

    return test_loss, 100.0 * correct / len(test_loader.dataset)


@dataclass()
class Meter:
    name: str
    history: List
    sum: float = 0
    avg: float = 0
    last: float = 0
    min: float = np.inf
    max: float = -np.inf
    extremum: str = ''
    monitor_min: bool = False

    """Stores all the incremented elements, their sum and average"""
    def __init__(self, name):
        self.name = name
        self.monitor_min = name.endswith('loss')
        self.reset()

    def reset(self):
        self.history = []
        self.sum = 0
        self.avg = 0
        self.last = 0
        self.min = np.inf
        self.max = -np.inf
        self.extremum = ''

    def add(self, value):
        self.last = value
        self.extremum = ''

        if value < self.min:
            self.min = value
            self.extremum = 'min'
        if value > self.max:
            self.max = value
            self.extremum = 'max'

        self.history.append(value)
        self.sum += value
        self.avg = self.sum / len(self.history)

    def is_best(self):
        """Check if the last epoch was the best according to the meter"""
        is_best = (self.monitor_min and self.extremum == 'min') or \
                  (not self.monitor_min and self.extremum == 'max')
        return is_best


class Runner:

    def __init__(self, cfg: DictConfig, log: Logger, train_loader: DataLoader, test_loader: DataLoader):
        self.log = log
        self.cfg = cfg
        self.train_loader = train_loader
        self.test_loader = test_loader

        # Set device
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.log.info(f'Using device={self.device}')

        # Set model
        self.model = torch_model(self.cfg.model.arch, self.cfg.data.classes, self.cfg.model.pretrained,
                                 log, module_name=self.cfg.model.module)
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
        # Paths
        results_root = Path(os.getcwd())
        checkpoint_path = results_root / self.cfg.results.checkpoints.root
        checkpoint_path /= f"{self.cfg.results.checkpoints.name}.pth"
        # Meters
        meters = ['train_loss', 'val_loss', 'train_acc', 'val_acc']
        meters = {m: Meter(m) for m in meters}
        writer = SummaryWriter(results_root / self.cfg.results.checkpoints.tag)
        # Training loop
        for epoch in range(self.cfg.train.epochs):
            try:
                train_loss, train_acc = train(self.model, self.device, self.train_loader,
                                              self.optimizer, self.loss_function, epoch,
                                              writer, self.log, self.scheduler)
                val_loss, val_acc = test(self.model, self.device, self.test_loader,
                                         self.loss_function, epoch, writer, self.log)
                # Meters
                meters['train_loss'].add(train_loss)
                meters['val_loss'].add(val_loss)
                meters['train_acc'].add(train_acc)
                meters['val_acc'].add(val_acc)

                # Checkpoint
                if meters[self.cfg.train.monitor].is_best:
                    self.log.info(f'Save the best model to {checkpoint_path}')
                    torch.save(self.model.state_dict(), checkpoint_path)

            except KeyboardInterrupt:
                self.log.info('Training interrupted')
                break
        writer.close()

