"""Dataset factory for evaluation."""

from __future__ import annotations

from typing import Type

from torch.utils.data import Dataset

from .dataset_config import get_dataset_path
from .eval_booster import BoosterDataset


_DATASETS: dict[str, Type[Dataset]] = {
    "e_booster": BoosterDataset,
}


def create_dataset(dataset_name: str, split: str = "val", **kwargs) -> Dataset:
    """Create a dataset instance by name."""
    if dataset_name not in _DATASETS:
        raise ValueError(f"Unsupported dataset: {dataset_name}. Available: {list(_DATASETS.keys())}")
    dataset_root = get_dataset_path(dataset_name, split)
    return _DATASETS[dataset_name](root=dataset_root, **kwargs)
