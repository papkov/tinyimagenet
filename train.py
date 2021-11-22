import logging
import os
import sys
from pathlib import Path

import albumentations as albu
import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from modules.dataset import DatasetItem, TinyImagenetDataset
from modules.runner import Runner
from modules.transform import transforms_train_torchvision, transforms_valid_torchvision


@hydra.main(config_path="config", config_name="config.yaml")
def main(cfg: DictConfig) -> None:
    """
    The main training function
    :param cfg: hydra config passed through the decorator
    :return: None
    """

    # Setup logging and show config (hydra takes care of naming)
    log = logging.getLogger(__name__)
    log.info(f"Config:\n{OmegaConf.to_yaml(cfg)}")

    if not hasattr(cfg, "experiment"):
        print(
            "Specify experiment as +experiment=experiment_name. "
            "Check for available experiment configs in config/experiment"
        )
        sys.exit(0)

    # Fix multiprocessing bug
    torch.multiprocessing.set_sharing_strategy("file_system")

    # Data
    # Specify data paths from config
    data_root = Path(hydra.utils.to_absolute_path(cfg.data.root))
    train_path = data_root / cfg.data.train
    val_path = data_root / cfg.data.val

    # Check if dataset is available
    log.info(f"Looking for dataset in {str(data_root)}")
    if not data_root.exists():
        log.error(
            "Folder not found. Terminating. "
            "See README.md for data downloading details."
        )
        return

    # Specify results paths from config
    results_root = Path(os.getcwd())  # hydra handles results folder
    checkpoint_path = results_root / cfg.results.checkpoints.root
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    checkpoint_path /= f"{cfg.results.checkpoints.name}.pth"
    log.info(f"Write checkpoints to {str(checkpoint_path)}")

    # Training
    # Augmentations
    if hasattr(cfg, "augmentation"):
        train_transform = albu.load(
            hydra.utils.to_absolute_path(cfg.augmentation.train), data_format="yaml"
        )
        valid_transform = albu.load(
            hydra.utils.to_absolute_path(cfg.augmentation.valid), data_format="yaml"
        )
        log.info(f"Loaded transforms from:\n{OmegaConf.to_yaml(cfg.augmentation)}")

        log.debug(train_transform)
        log.debug(valid_transform)
    else:
        log.info("Use TrivialAugment by default")
        train_transform = transforms_train_torchvision()
        valid_transform = transforms_valid_torchvision()

    # Dataset
    use_albumentations = hasattr(cfg, "augmentation")
    train_dataset = TinyImagenetDataset(
        train_path, cfg.data, train_transform, use_albumentations
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        collate_fn=DatasetItem.collate,
        num_workers=cfg.train.num_workers,
        drop_last=True,
        pin_memory=True,
    )
    log.info(
        f"Created training dataset ({len(train_dataset)}) "
        f"and loader ({len(train_loader)}): "
        f"batch size {cfg.train.batch_size}, "
        f"num workers {cfg.train.num_workers}"
    )

    valid_dataset = TinyImagenetDataset(
        val_path, cfg.data, valid_transform, use_albumentations
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=cfg.train.batch_size,
        shuffle=False,
        collate_fn=DatasetItem.collate,
        num_workers=cfg.train.num_workers,
        pin_memory=True,
    )
    log.info(
        f"Created validation dataset ({len(valid_dataset)}) "
        f"and loader ({len(valid_loader)}): "
        f"batch size {cfg.train.batch_size}, "
        f"num workers {cfg.train.num_workers}"
    )

    runner = Runner(cfg, log, train_loader, valid_loader)
    runner.fit()


if __name__ == "__main__":
    main()
