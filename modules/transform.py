from pathlib import Path
from typing import Union

import albumentations as albu
from albumentations import pytorch as albu_pytorch

from pytorch_typing import Transform


def to_tensor_normalize() -> Transform:
    """
    :return: Albumentations transform [imagenet normalization, to tensor]
    """
    base_transform = albu.Compose(
        [
            albu.Normalize(
                [0.4802, 0.4481, 0.3975], [0.2302, 0.2265, 0.2262], max_pixel_value=255
            ),
            albu_pytorch.ToTensorV2(),
        ]
    )
    return base_transform


def load_albu_transform(path: Union[str, Path], data_format: str = "yaml") -> Transform:
    """
    :param path: path to augmentation config
    :param data_format: config format
    :return: Albumentations transform
    """
    albu_transform = albu.load(path, data_format=data_format)
    return albu_transform
