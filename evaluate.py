import argparse
import logging
from pathlib import Path
from typing import Tuple, Union

import albumentations as albu
import numpy as np
import torch
from hydra.utils import instantiate, to_absolute_path
from omegaconf import OmegaConf
from torch.nn.modules import loss
from torch.utils.data import DataLoader

from modules.dataset import DatasetItem, TinyImagenetDataset
from modules.runner import test
from modules.transform import to_tensor_normalize


def evaluate_model(
    results_root: Union[str, Path], data_part: str = "val", device: str = "cuda"
) -> Tuple[float, float, np.ndarray]:
    """
    The main training function
    :param results_root: path to results folder
    :param data_part: {train, val, test} partition to evaluate model on
    :param device: {cuda, cpu}
    :return: None
    """
    results_root = Path(results_root)
    logging.basicConfig(
        filename=results_root / f"{data_part}.log", level=logging.NOTSET
    )
    # Setup logging and show config ьфлу
    log = logging.getLogger(__name__)
    if not log.handlers:
        log.addHandler(logging.StreamHandler())

    cfg_path = results_root / ".hydra/config.yaml"
    log.info(f"Read config from {cfg_path}")

    cfg = OmegaConf.load(str(cfg_path))
    log.debug(f"Config:\n{cfg.pretty()}")

    # Specify results paths from config
    checkpoint_path = results_root / cfg.results.checkpoints.root
    checkpoint_path /= f"{cfg.results.checkpoints.name}.pth"

    # Data
    # Specify data paths from config
    data_root = Path(cfg.data.root)
    test_path = data_root / data_part

    # Check if dataset is available
    log.info(f"Looking for dataset in {str(data_root)}")
    if not data_root.exists():
        log.error(
            "Folder not found. Terminating. "
            "See README.md for data downloading details."
        )
        raise FileNotFoundError

    valid_transform = to_tensor_normalize()
    if "augmentation" in cfg:
        pre_transform = albu.load(
            to_absolute_path(cfg.augmentation.pre), data_format="yaml"
        )
        post_transform = albu.load(
            to_absolute_path(cfg.augmentation.post), data_format="yaml"
        )
        valid_transform = albu.Compose([pre_transform, post_transform])

    test_dataset = TinyImagenetDataset(test_path, cfg, valid_transform)
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.train.batch_size,
        shuffle=False,
        collate_fn=DatasetItem.collate,
        num_workers=cfg.train.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    log.info(
        f"Created test dataset ({len(test_dataset)}) "
        f"and loader ({len(test_loader)}): "
        f"batch size {cfg.train.batch_size}, "
        f"num workers {cfg.train.num_workers}"
    )

    loss_function = loss.CrossEntropyLoss()
    model = instantiate(cfg.model)
    try:
        model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
    except RuntimeError as e:
        log.error("Failed loading state dict")
        raise e
    except FileNotFoundError as e:
        log.error("Checkpoint not found")
        raise e
    log.info(f"Loaded model from {checkpoint_path}")
    device = torch.device(
        "cuda" if device == "cuda" and torch.cuda.is_available() else "cpu"
    )
    test_loss, test_acc, test_outputs = test(
        model, device, test_loader, loss_function, 0, log
    )
    log.info(f"Loss {test_loss}, acc {test_acc}")
    log.info(f"Outputs:\n{test_outputs.shape}\n{test_outputs[:5, :5]}")
    logging.shutdown()
    return test_loss, test_acc, test_outputs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--results", help="results root", type=str)
    parser.add_argument(
        "-p",
        "--data_part",
        default="val",
        choices=["train", "val", "test"],
        help="data partition to evaluate on",
        type=str,
    )
    parser.add_argument(
        "-d", "--device", default="cuda", choices=["cuda", "cpu"], type=str
    )
    args = parser.parse_args()
    test_loss, test_acc, test_outputs = evaluate_model(
        args.results, args.data_part, args.device
    )
