"""
Clean typing for PyTorch
https://github.com/catalyst-team/catalyst/blob/master/catalyst/tools/typing.py
"""
from typing import Any, Callable, Dict, Iterator, Union

import torch
from PIL import Image
from torch import nn, optim
from torch.nn.parameter import Parameter
from torch.optim import lr_scheduler
from torch.utils import data

Model = nn.Module
Criterion = nn.Module
Optimizer = optim.Optimizer
Scheduler = lr_scheduler._LRScheduler  # noinspection PyProtectedMember
Dataset = data.Dataset
Device = Union[str, torch.device]
Parameters = Iterator[Parameter]
Transform = Union[Callable[[Image.Image], Image.Image], Callable[..., Dict[str, Any]]]


__all__ = [
    "Model",
    "Criterion",
    "Optimizer",
    "Scheduler",
    "Dataset",
    "Device",
    "Parameters",
    "Transform",
]
