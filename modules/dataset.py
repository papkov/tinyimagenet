import os
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Union

import pandas as pd
from pandas import DataFrame
from PIL import Image

import torch
from torch.utils.data.dataloader import default_collate
from torch.utils.data.dataset import Dataset
from torchvision import transforms

from omegaconf import DictConfig

Transform = Callable[[Image.Image], Image.Image]


def get_labels_mapping(cfg):
    data_root = Path(cfg.data.root)
    all_folders = [
        dir_name
        for r, d, f in os.walk(data_root / cfg.data.train)
        for dir_name in d
        if dir_name != "images"
    ]
    folders_to_num = {val: index for index, val in enumerate(all_folders)}

    val_labels = pd.read_csv(
        data_root / cfg.data.val / cfg.data.val_labels, sep="\t", header=None, index_col=0)[1].to_dict()

    return folders_to_num, val_labels


@dataclass()
class ItemsBatch:
    images: torch.Tensor
    labels: torch.Tensor
    ids: List[int]
    paths: List[Path]
    items: List["DatasetItem"]


@dataclass()
class DatasetItem:
    image: Union[torch.Tensor, Image.Image]
    label: int
    id: int
    path: Path

    @classmethod
    def collate(cls, items: Sequence["DatasetItem"]) -> ItemsBatch:
        if not isinstance(items, list):
            items = list(items)
        return ItemsBatch(
            images=default_collate([item.image for item in items]),
            labels=default_collate([item.label for item in items]),
            ids=[item.id for item in items],
            paths=[item.path for item in items],
            items=items,
        )


class TinyImagenetDataset(Dataset):
    _transform: Optional[Transform]
    _root: Path
    _df: DataFrame

    def __init__(self, path, cfg: DictConfig, transform: Optional[Transform] = None):
        self._transform = transform
        folders_to_num, val_labels = get_labels_mapping(cfg)

        if not os.path.isdir(path):
            raise NotADirectoryError(f"{path} is not a directory.")
        all_files = [
            os.path.join(r, fyle)
            for r, d, f in os.walk(path)
            for fyle in f
            if ".JPEG" in fyle
        ]
        labels = [
            folders_to_num.get(
                os.path.basename(f).split("_")[0],
                folders_to_num.get(val_labels.get(os.path.basename(f))),
            )
            for f in all_files
        ]
        self._df = pd.DataFrame({"path": all_files, "label": labels})

    def __getitem__(self, index: int) -> DatasetItem:
        path, label = self._df.loc[index, :]
        image = Image.open(path).convert("RGB")
        if self._transform:
            image = self._transform(image)
        return DatasetItem(image=image, label=label, id=index, path=path)

    def __len__(self) -> int:
        return len(self._df)
