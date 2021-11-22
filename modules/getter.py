from logging import Logger
from types import FunctionType
from typing import Any, Union
from warnings import warn

import torch
import torch.nn as nn
import torchvision

from modules import models
from modules.pytorch_typing import *


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
    warn("Deprecated, use hydra.utils.instantiate() instead", DeprecationWarning)
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
    warn("Deprecated, use hydra.utils.instantiate() instead", DeprecationWarning)
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
    name: str, optimizer: Optimizer, log: Logger, **kwargs: Any
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
    warn("Deprecated, use hydra.utils.instantiate() instead", DeprecationWarning)
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
