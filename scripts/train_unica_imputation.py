#!/usr/bin/env python
"""
Fine-tune the UniCA covariate adapter (wrapping a frozen MOMENT backbone)
to reconstruct masked targets (OT) using the remaining columns as covariates.

The training follows the masking setup from the MOMENT imputation tutorial:
for each sliding window we randomly hide a fraction of the OT timesteps and
optimize UniCA to reconstruct only the masked positions while the covariates
remain fully observed.
"""

import argparse
import random
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from momentfm.utils.masking import Masking

from data.imputation_splitter import load_split_overrides, load_split_series
from models.adapter.unica.module import UniCA
from models.wrapper.fm.moment_wrapper import MomentWrapper


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Directory containing the pretrained MOMENT checkpoint.",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        required=True,
        help="CSV file with the imputation dataset (e.g., data/datasets/imputation/ETTh1.csv).",
    )
    parser.add_argument(
        "--context-length",
        type=int,
        default=512,
        help="Window size (must match MOMENT context length).",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=8,
        help="Stride between consecutive sliding windows (default: 8).",
    )
    parser.add_argument(
        "--mask-ratios",
        type=float,
        nargs="+",
        default=[0.125, 0.25, 0.375, 0.5],
        help="Fractions (or percentages >1) of timesteps to hide.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size for training.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Learning rate for UniCA.",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.0,
        help="Weight decay for the optimizer.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=13,
        help="Random seed.",
    )
    parser.add_argument(
        "--standardize",
        dest="standardize",
        action="store_true",
        help="Apply column-wise StandardScaler normalization.",
    )
    parser.add_argument(
        "--no-standardize",
        dest="standardize",
        action="store_false",
        help="Skip normalization.",
    )
    parser.set_defaults(standardize=True)
    parser.add_argument(
        "--target-column",
        type=str,
        default="OT",
        help="Target column to mask.",
    )
    parser.add_argument(
        "--output-checkpoint",
        type=Path,
        required=True,
        help="Path to store the fine-tuned UniCA state dict.",
    )
    parser.add_argument(
        "--max-windows",
        type=int,
        default=None,
        help="Optional cap on the number of sliding windows.",
    )
    parser.add_argument(
        "--split-config",
        type=Path,
        default=None,
        help="Optional JSON file specifying train/val/test lengths per dataset.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Torch device override.",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def sliding_windows(values: np.ndarray, window: int, stride: int) -> np.ndarray:
    seq_len = values.shape[0]
    if seq_len < window:
        return np.empty((0, window, values.shape[1]), dtype=np.float32)
    windows: List[np.ndarray] = []
    for start in range(0, seq_len - window + 1, stride):
        windows.append(values[start:start + window])
    return np.stack(windows).astype(np.float32)


class OTImputationDataset(Dataset):
    def __init__(self, targets: np.ndarray, covariates: np.ndarray):
        self.targets = targets.astype(np.float32)
        self.covariates = covariates.astype(np.float32)

    def __len__(self) -> int:
        return self.targets.shape[0]

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        return self.targets[idx], self.covariates[idx]


def convert_mask_ratios(ratios: List[float]) -> List[float]:
    converted = []
    for ratio in ratios:
        value = ratio / 100.0 if ratio > 1 else ratio
        if not 0 < value < 1:
            raise ValueError(f"Mask ratio {ratio} must be in (0, 1) or a percentage.")
        converted.append(value)
    return converted


def build_unica_inputs(
        batch_size: int,
        context_length: int,
        prediction_length: int,
        num_covariates: int,
        device: torch.device,
) -> Tuple[torch.Tensor, ...]:
    zero_real = torch.zeros(batch_size, 0, device=device)
    zero_cat = torch.zeros(batch_size, 0, dtype=torch.long, device=device)
    feat_dynamic_real = torch.zeros(batch_size, context_length + prediction_length, 0, device=device)
    feat_dynamic_cat = torch.zeros(batch_size, context_length + prediction_length, 0, dtype=torch.long, device=device)
    past_feat_dyn_cat = torch.zeros(batch_size, context_length, 0, dtype=torch.long, device=device)
    obs_feat_dynamic_real = torch.zeros(batch_size, context_length + prediction_length, 0, device=device)
    obs_feat_dynamic_cat = torch.zeros(batch_size, context_length + prediction_length, 0, dtype=torch.float32, device=device)
    past_obs_feat_dyn_cat = torch.zeros(batch_size, context_length, 0, dtype=torch.float32, device=device)
    item_index = torch.zeros(batch_size, 1, dtype=torch.long, device=device)
    ones_cov_mask = torch.ones(batch_size, context_length, num_covariates, device=device)
    return (
        zero_real,
        zero_cat,
        feat_dynamic_real,
        feat_dynamic_cat,
        past_feat_dyn_cat,
        obs_feat_dynamic_real,
        obs_feat_dynamic_cat,
        past_obs_feat_dyn_cat,
        item_index,
        ones_cov_mask,
    )


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    stride = args.stride or args.context_length
    mask_ratios = convert_mask_ratios(args.mask_ratios)

    split_overrides = load_split_overrides(args.split_config)
    train_series, columns, _ = load_split_series(
        csv_path=args.dataset,
        split="train",
        seq_len=args.context_length,
        standardize=args.standardize,
        overrides=split_overrides,
    )

    columns_list = columns.tolist() if isinstance(columns, np.ndarray) else list(columns)
    if args.target_column not in columns_list:
        raise ValueError(f"Target column {args.target_column} not found in {args.dataset.name}.")
    target_idx = columns_list.index(args.target_column)

    windows = sliding_windows(train_series, args.context_length, stride)
    if windows.size == 0:
        raise RuntimeError("Dataset shorter than context length.")
    if args.max_windows is not None and windows.shape[0] > args.max_windows:
        windows = windows[: args.max_windows]

    target_windows = windows[:, :, target_idx]
    covariates = np.delete(windows, target_idx, axis=2)
    num_covariates = covariates.shape[-1]

    dataset = OTImputationDataset(target_windows, covariates)
    if len(dataset) < args.batch_size:
        print(f"[WARN] Dataset has only {len(dataset)} windows which is smaller than batch size {args.batch_size}. "
              "Consider reducing --batch-size or --stride to obtain more training samples.")
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=0,
    )

    wrapper = MomentWrapper(
        model_id_or_path=args.model_path,
        prediction_length=1,
        device=str(device),
        moment_task="imputation",
    )
    patch_len = int(wrapper.model.config.patch_len)
    patch_stride = int(wrapper.model.config.patch_stride_len)
    mask_generators = {
        ratio: Masking(mask_ratio=ratio, patch_len=patch_len, stride=patch_stride)
        for ratio in mask_ratios
    }

    past_dims = [1] * num_covariates
    adapter_config = {
        "context_length": args.context_length,
        "prediction_length": 1,
        "d_feat_static_real": [],
        "c_feat_static_cat": [],
        "d_feat_dynamic_real": [],
        "c_feat_dynamic_cat": [],
        "d_past_feat_dynamic_real": past_dims,
        "c_past_feat_dynamic_cat": [],
        "with_future": False,
        "with_past": True,
        "with_gate": True,
        "future_with_gate": False,
        "use_satellite": False,
        "use_text": False,
    }
    unica = UniCA(
        model_wrapper=wrapper,
        **adapter_config,
    ).to(device)
    unica.train()

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, unica.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    prediction_length = 1
    for epoch in range(1, args.epochs + 1):
        epoch_loss = 0.0
        total_steps = 0
        for batch_targets, batch_cov in loader:
            batch_targets = batch_targets.to(device=device, dtype=torch.float32)
            batch_cov = batch_cov.to(device=device, dtype=torch.float32)
            batch_size, seq_len = batch_targets.shape

            ratio = random.choice(mask_ratios)
            mask_generator = mask_generators[ratio]
            context = batch_targets.unsqueeze(1)  # [B, 1, T]
            input_mask = torch.ones(batch_size, seq_len, dtype=torch.long, device=device)
            observed_mask = mask_generator.generate_mask(x=context, input_mask=input_mask).to(device=device).float()
            full_obs = observed_mask.sum(dim=1) == seq_len
            if full_obs.any():
                for idx in torch.where(full_obs)[0]:
                    drop_idx = torch.randint(0, seq_len, (1,), device=device)
                    observed_mask[idx, drop_idx] = 0.0
            corrupted = batch_targets * observed_mask

            (
                feat_static_real,
                feat_static_cat,
                feat_dynamic_real,
                feat_dynamic_cat,
                past_feat_dynamic_cat,
                observed_feat_dynamic_real,
                observed_feat_dynamic_cat,
                past_observed_feat_dynamic_cat,
                item_index,
                cov_mask,
            ) = build_unica_inputs(
                batch_size=batch_size,
                context_length=args.context_length,
                prediction_length=prediction_length,
                num_covariates=num_covariates,
                device=device,
            )

            recon_dict = unica.impute(
                past_target=corrupted,
                past_observed_values=observed_mask,
                item_index=item_index,
                feat_static_real=feat_static_real,
                feat_static_cat=feat_static_cat,
                feat_dynamic_real=feat_dynamic_real,
                feat_dynamic_cat=feat_dynamic_cat,
                past_feat_dynamic_real=batch_cov if num_covariates > 0 else None,
                past_feat_dynamic_cat=past_feat_dynamic_cat,
                observed_feat_dynamic_real=observed_feat_dynamic_real,
                observed_feat_dynamic_cat=observed_feat_dynamic_cat,
                past_observed_feat_dynamic_real=cov_mask if num_covariates > 0 else None,
                past_observed_feat_dynamic_cat=past_observed_feat_dynamic_cat,
            )
            reconstruction = recon_dict["reconstruction"]
            missing = 1.0 - observed_mask
            denom = torch.clamp(missing.sum(), min=1.0)
            loss = torch.sum((reconstruction - batch_targets) ** 2 * missing) / denom

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            total_steps += 1

        avg_loss = epoch_loss / max(total_steps, 1)
        print(f"Epoch {epoch}/{args.epochs} - Avg masked MSE: {avg_loss:.6f}")

    args.output_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"state_dict": unica.state_dict(), "train_args": vars(args), "adapter_config": adapter_config},
        args.output_checkpoint,
    )
    print(f"Saved UniCA checkpoint to {args.output_checkpoint}")


if __name__ == "__main__":
    main()
