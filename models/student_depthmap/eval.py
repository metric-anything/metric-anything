#!/usr/bin/env python3
"""Evaluation script for MetricAnything DepthMap."""

from __future__ import annotations

import argparse
import json
import logging
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from depth_model import MetricAnythingDepthMap
from dataset.dataset_factory import create_dataset

LOGGER = logging.getLogger(__name__)

def eval_depth(pred, target):
    assert pred.shape == target.shape

    thresh = torch.max((target / pred), (pred / target))

    d1 = torch.sum(thresh < 1.25).float() / len(thresh)
    d2 = torch.sum(thresh < 1.25 ** 2).float() / len(thresh)
    d3 = torch.sum(thresh < 1.25 ** 3).float() / len(thresh)

    diff = pred - target
    diff_log = torch.log(pred) - torch.log(target)

    abs_rel = torch.mean(torch.abs(diff) / target)
    sq_rel = torch.mean(torch.pow(diff, 2) / target)

    rmse = torch.sqrt(torch.mean(torch.pow(diff, 2)))
    mae = torch.mean(torch.abs(diff))
    rmse_log = torch.sqrt(torch.mean(torch.pow(diff_log , 2)))

    log10 = torch.mean(torch.abs(torch.log10(pred) - torch.log10(target)))
    silog = torch.sqrt(torch.pow(diff_log, 2).mean() - 0.5 * torch.pow(diff_log.mean(), 2))

    return {'d1': d1.item(), 'd2': d2.item(), 'd3': d3.item(), 'abs_rel': abs_rel.item(), 'sq_rel': sq_rel.item(), 
            'rmse': rmse.item(), 'mae': mae.item(), 'rmse_log': rmse_log.item(), 'log10':log10.item(), 'silog':silog.item()}


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _parse_device(value: str | None) -> str | int:
    if value is None:
        return "cuda" if torch.cuda.is_available() else "cpu"
    return int(value) if value.isdigit() else value


@torch.no_grad()
def evaluate(args: argparse.Namespace) -> dict[str, float]:
    """Run evaluation and return aggregated metrics."""
    set_seed(args.seed)

    device_value = _parse_device(args.device)
    device = torch.device(f"cuda:{device_value}") if isinstance(device_value, int) else torch.device(device_value)

    model = MetricAnythingDepthMap.from_pretrained(
        args.pretrained,
        model_kwargs={"device": device_value},
        filename='student_depthmap.pt',
    )
    model.eval()

    dataset = create_dataset(
        args.dataset,
        split="val",
        target_size=model.img_size,
    )
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    pred_values = []
    gt_values = []

    for batch in tqdm(loader, desc="Eval"):
        images = batch["image"].to(device)
        f_px = batch["f_px"].to(device, dtype=images.dtype)
        orig_size = batch["orig_size"]
        orig_h, orig_w = int(orig_size[0, 0].item()), int(orig_size[0, 1].item())

        canonical_inverse = model(images)
        inverse_depth = canonical_inverse * orig_w / f_px.view(-1, 1, 1, 1)
        pred_inverse = F.interpolate(
            inverse_depth,
            size=(orig_h, orig_w),
            mode="bilinear",
            align_corners=False,
        )
        pred_depth = 1.0 / torch.clamp(pred_inverse, min=1e-3, max=1e3)
        pred_depth = pred_depth.squeeze(1)

        gt_depth = batch["depth"].to(device)
        valid_mask = batch["valid_mask"].to(device)

        pred_valid = pred_depth[valid_mask]
        gt_valid = gt_depth[valid_mask]
        if pred_valid.numel() == 0:
            continue

        pred_values.append(pred_valid.cpu())
        gt_values.append(gt_valid.cpu())

    if not pred_values:
        raise RuntimeError("No valid pixels found for evaluation.")

    pred_all = torch.cat(pred_values, dim=0)
    gt_all = torch.cat(gt_values, dim=0)
    return eval_depth(pred_all, gt_all)


def main() -> None:
    parser = argparse.ArgumentParser(description="MetricAnything DepthMap evaluation (Booster)")
    parser.add_argument("--pretrained", type=str, default="yjh001/metricanything_student_depthmap", help="Checkpoint path or Hub repo id.")
    parser.add_argument("--dataset", type=str, default="e_booster", help="Dataset name.")
    parser.add_argument("--output_dir", type=str, default="./eval_output", help="Output directory.")
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader workers.")
    parser.add_argument("--device", type=str, default=None, help="Device string or CUDA index.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    metrics = evaluate(args)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.json"
    with metrics_path.open("w") as f:
        json.dump(metrics, f, indent=2)

    LOGGER.info("Evaluation metrics:")
    for key, value in metrics.items():
        LOGGER.info("  %s: %.4f", key, value)
    LOGGER.info("Saved metrics to %s", metrics_path)


if __name__ == "__main__":
    main()
