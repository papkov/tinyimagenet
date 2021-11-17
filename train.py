import logging
import os
from pathlib import Path

import albumentations as albu
import hydra
import torch
from omegaconf import DictConfig, OmegaConf

from modules.dataset import DatasetItem, TinyImagenetDataset
from modules.runner import Runner
from modules.transform import load_albu_transform, to_tensor_normalize


@hydra.main(config_path="config", config_name="config.yaml")
def main(cfg: DictConfig) -> None:
    """
    The main training function
    :param cfg: hydra config passed through the decorator
    :return: None
    """
    # Setup logging and show config (hydra takes care of naming)
    log = logging.getLogger(__name__)
    print(f"Config:\n{OmegaConf.to_yaml(cfg)}")

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
    base_transform = to_tensor_normalize()
    if "augmentation" in cfg:
        augmentation_root = hydra.utils.to_absolute_path(cfg.augmentation.root)
        albu_transform = load_albu_transform(augmentation_root)
        log.info(f"Loaded transforms from {augmentation_root}")
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
    log.info(
        f"Created training dataset ({len(train_dataset)}) "
        f"and loader ({len(train_loader)}): "
        f"batch size {cfg.train.batch_size}, "
        f"num workers {cfg.train.num_workers}"
    )

    valid_dataset = TinyImagenetDataset(val_path, cfg, base_transform)
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=cfg.train.batch_size,
        shuffle=False,
        collate_fn=DatasetItem.collate,
        num_workers=cfg.train.num_workers,
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
