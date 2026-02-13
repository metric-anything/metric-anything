"""Dataset paths for evaluation."""

from __future__ import annotations

from typing import Dict


DATASET_CONFIGS: Dict[str, Dict[str, str]] = {
    "e_booster": {
        "val": "dataset/raw_data/eval_booster/train",
    },
}


def get_dataset_path(dataset_name: str, split: str = "val") -> str:
    """Return dataset root for a given split."""
    if dataset_name not in DATASET_CONFIGS:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    if split not in DATASET_CONFIGS[dataset_name]:
        raise ValueError(f"Split '{split}' not found for dataset '{dataset_name}'")
    return DATASET_CONFIGS[dataset_name][split]


def list_datasets() -> list[str]:
    """List available dataset keys."""
    return list(DATASET_CONFIGS.keys())
