import os
from pathlib import Path

import logging
import hydra
from omegaconf import DictConfig

import torch
from torchvision import transforms

from dataset import TinyImagenetDataset, DatasetItem
from runner import Runner

import albumentations as albu
from albumentations import pytorch as albu_pytorch


@hydra.main(config_path="../config/config.yaml")
def main(cfg: DictConfig):
    """
    The main training function
    :param cfg: hydra config passed through the decorator
    :return: None
    """
    # Setup logging and show config (hydra takes care of naming)
    log = logging.getLogger(__name__)
    log.debug(f'Config:\n{cfg.pretty()}')

    # Data
    # Specify data paths from config
    data_root = Path(hydra.utils.to_absolute_path(cfg.data.root))
    train_path = data_root / cfg.data.train
    val_path = data_root / cfg.data.val

    # Check if dataset is available
    log.info(f'Looking for dataset in {str(data_root)}')
    if not data_root.exists():
        log.error("Folder not found. Terminating. "
                  "See README.md for data downloading details.")
        return

    # Specify results paths from config
    results_root = Path(os.getcwd())  # hydra handles results folder
    checkpoint_path = results_root / cfg.results.checkpoints.root
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    checkpoint_path /= f"{cfg.results.checkpoints.name}.pth"
    log.info(f"Write checkpoints to {str(checkpoint_path)}")

    # Training
    # Augmentations
    base_transform = albu.Compose(
        [
            albu.Normalize([0.4802, 0.4481, 0.3975],
                           [0.2302, 0.2265, 0.2262]),
            albu_pytorch.ToTensorV2()
        ]
    )
    if 'augmentation' in cfg:
        albu_transform = albu.load(
            hydra.utils.to_absolute_path(cfg.augmentation.root), data_format='yaml'
        )
        log.info(f"Loaded transforms from to {str(cfg.augmentation.root)}")
        log.debug(albu_transform)
        transform = albu.Compose([albu_transform, base_transform])
    else:
        log.info("Augmentations will not be applied")
        transform = base_transform

    # Dataset
    train_dataset = TinyImagenetDataset(train_path, cfg, transform)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        collate_fn=DatasetItem.collate,
        num_workers=cfg.train.num_workers,
    )
    log.info(f"Created training dataset ({len(train_dataset)}) "
             f"and loader ({len(train_loader)})")

    test_dataset = TinyImagenetDataset(val_path, cfg, base_transform)
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=cfg.train.batch_size,
        shuffle=False,
        collate_fn=DatasetItem.collate,
        num_workers=cfg.train.num_workers,
    )
    log.info(f"Created validation dataset ({len(test_dataset)}) "
             f"and loader ({len(test_loader)})")

    runner = Runner(cfg, log, train_loader, test_loader)
    runner.fit()


if __name__ == '__main__':
    main()
