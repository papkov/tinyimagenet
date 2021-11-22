import torchvision.transforms as T
from albumentations import Compose, Normalize
from albumentations.pytorch.transforms import ToTensorV2

from modules.pytorch_typing import Transform


def to_tensor_normalize() -> Transform:
    """
    :return: Albumentations transform [imagenet normalization, to tensor]
    """
    base_transform = Compose(
        [
            Normalize(
                [0.4802, 0.4481, 0.3975],
                [0.2302, 0.2265, 0.2262],
                max_pixel_value=255,
            ),
            ToTensorV2(),
        ]
    )
    return base_transform


def to_tensor_normalize_torchvision() -> Transform:
    """
    :return: Torchvision transform [imagenet normalization, to tensor]
    """
    return T.Compose(
        [
            T.ToTensor(),
            T.Normalize(
                [0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225],
            ),
        ]
    )


def transforms_train_torchvision() -> Transform:
    """
    :return: Torchvision default training
    """
    return T.Compose(
        [
            T.Resize(256),
            T.RandomCrop(176),
            T.TrivialAugmentWide(),
            to_tensor_normalize_torchvision(),
        ]
    )


def transforms_valid_torchvision() -> Transform:
    """
    :return: Torchvision default validation
    """
    return T.Compose(
        [
            T.Resize(232),
            T.RandomCrop(224),
            to_tensor_normalize_torchvision(),
        ]
    )
