import json
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


DEFAULT_SPLITS: Dict[str, Dict[str, int]] = {}
FALLBACK_RATIO = (0.6, 0.2, 0.2)

def _normalize_name(name: str) -> str:
    return Path(name).stem.lower()


def load_split_overrides(path: Optional[Path]) -> Dict[str, Dict[str, int]]:
    if path is None:
        return {}
    data = json.loads(Path(path).read_text())
    return {_normalize_name(key): value for key, value in data.items()}


def _coerce_lengths(lengths: Dict[str, int], total_len: int) -> Tuple[int, int, int]:
    train = max(0, min(int(lengths.get("train", 0)), total_len))
    remaining = total_len - train
    val = max(0, min(int(lengths.get("val", 0)), remaining))
    remaining -= val
    test = max(0, min(int(lengths.get("test", 0)), remaining))
    if train == 0 or test == 0:
        raise ValueError("Split configuration must allocate positive lengths for train and test splits.")
    return train, val, test


def get_split_lengths(
        dataset_name: str,
        total_len: int,
        overrides: Optional[Dict[str, Dict[str, int]]] = None,
) -> Tuple[int, int, int]:
    overrides = overrides or {}
    key = _normalize_name(dataset_name)
    if key in overrides:
        return _coerce_lengths(overrides[key], total_len)

    train = int(total_len * FALLBACK_RATIO[0])
    val = int(total_len * FALLBACK_RATIO[1])
    test = total_len - train - val
    if train <= 0 or test <= 0:
        raise ValueError(f"Unable to derive splits for dataset {dataset_name}. Please provide a split config.")
    return train, val, test


def _slice_for_split(
        split: str,
        train_len: int,
        val_len: int,
        test_len: int,
        seq_len: int,
        total_len: int,
) -> slice:
    if split == "train":
        return slice(0, min(train_len, total_len))
    if split == "val":
        start = max(train_len - seq_len, 0)
        end = min(train_len + val_len, total_len)
        return slice(start, end)
    if split == "test":
        start = max(train_len + val_len - seq_len, 0)
        end = min(start + test_len + seq_len, total_len)
        return slice(start, end)
    raise ValueError(f"Unknown split: {split}")


def load_split_series(
        csv_path: Path,
        split: str,
        seq_len: int,
        standardize: bool,
        overrides: Optional[Dict[str, Dict[str, int]]] = None,
) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int, int]]:
    df = pd.read_csv(csv_path)
    columns = np.array([col for col in df.columns if col.lower() != "date"])
    values = (
        df[columns]
        .infer_objects(copy=False)
        .interpolate(method="linear")
        .to_numpy(dtype=np.float32)
    )
    dataset_name = Path(csv_path).stem
    train_len, val_len, test_len = get_split_lengths(dataset_name, values.shape[0], overrides)

    if standardize:
        scaler = StandardScaler()
        scaler.fit(values[:train_len])
        values = scaler.transform(values)

    segment_slice = _slice_for_split(split, train_len, val_len, test_len, seq_len, values.shape[0])
    segment = values[segment_slice]
    if segment.shape[0] < seq_len:
        raise ValueError(
            f"Split '{split}' for dataset {dataset_name} is shorter than the required context length {seq_len}."
        )
    return segment, columns, (train_len, val_len, test_len)
