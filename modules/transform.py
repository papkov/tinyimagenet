from pathlib import Path
from typing import Union

from albumentations import Normalize, load, Compose
from albumentations.pytorch.transforms import ToTensorV2

from modules.pytorch_typing import Transform


def to_tensor_normalize() -> Transform:
    """
    :return: Albumentations transform [imagenet normalization, to tensor]
    """
    base_transform = Compose(
        [
            Normalize(
                [0.4802, 0.4481, 0.3975], [0.2302, 0.2265, 0.2262], max_pixel_value=255
            ),
            ToTensorV2(),
        ]
    )
    return base_transform
