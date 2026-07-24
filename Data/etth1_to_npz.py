#!/usr/bin/env python3
"""Convert the official ETTh1 training prefix into Diffusion-TS NPZ files.

Only the first 8,640 hourly observations are used.  For each prediction
horizon, this utility first builds every stride-spaced window of length
``lookback + horizon``.  It then reserves the half-open sample-index interval
``[floor(0.70 * N), floor(0.85 * N))`` for generator validation and uses all
remaining samples for generator training.

The output is deliberately kept in raw physical scale.  Diffusion-TS fits its
normalizer from the training NPZ, and its existing ``NPZDataset`` only requires
the ``data`` key; the additional fields written here make the split auditable.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd


TRAIN_POINTS = 8640
DEFAULT_CSV = Path("/data/jinyuli/.darts/datasets/ETTh1.csv")
DEFAULT_OUTPUT_DIR = Path("Data/datasets/etth1")
FEATURE_COLS = ("HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT")
DEFAULT_HORIZONS = (96, 192, 336, 720)


def _validate_options(
    lookback: int,
    horizons: Sequence[int],
    val_start_ratio: float,
    val_end_ratio: float,
    stride: int,
) -> None:
    if lookback <= 0:
        raise ValueError(f"--lookback must be > 0, got {lookback}.")
    if not horizons:
        raise ValueError("--horizons must contain at least one prediction horizon.")
    if any(horizon <= 0 for horizon in horizons):
        raise ValueError(f"Every horizon must be > 0, got {list(horizons)}.")
    if len(set(horizons)) != len(horizons):
        raise ValueError(f"--horizons must not contain duplicates, got {list(horizons)}.")
    if stride <= 0:
        raise ValueError(f"--stride must be > 0, got {stride}.")
    if not (
        math.isfinite(val_start_ratio)
        and math.isfinite(val_end_ratio)
        and 0.0 <= val_start_ratio < val_end_ratio <= 1.0
    ):
        raise ValueError(
            "Validation sample ratios must satisfy "
            f"0 <= start < end <= 1, got {val_start_ratio} and {val_end_ratio}."
        )
    too_long = [
        horizon for horizon in horizons if lookback + horizon > TRAIN_POINTS
    ]
    if too_long:
        raise ValueError(
            "lookback + horizon cannot exceed the 8,640-point ETTh1 training "
            f"prefix; invalid horizons: {too_long}."
        )


def _load_training_prefix(csv_path: Path) -> Tuple[pd.DatetimeIndex, np.ndarray]:
    """Load and validate exactly the official 8,640-row ETTh1 train prefix."""
    csv_path = Path(csv_path)
    if not csv_path.is_file():
        raise FileNotFoundError(f"ETTh1 CSV not found: {csv_path}")

    frame = pd.read_csv(csv_path, nrows=TRAIN_POINTS)
    required = ["date"] + list(FEATURE_COLS)
    missing_columns = [column for column in required if column not in frame.columns]
    if missing_columns:
        raise ValueError(
            f"CSV {csv_path} is missing required columns: {missing_columns}."
        )
    if len(frame) < TRAIN_POINTS:
        raise ValueError(
            f"CSV {csv_path} must contain at least {TRAIN_POINTS} data rows; "
            f"found {len(frame)}."
        )

    dates = pd.to_datetime(frame["date"], errors="coerce")
    invalid_date_rows = np.flatnonzero(dates.isna().to_numpy())
    if invalid_date_rows.size:
        raise ValueError(
            "The first 8,640 rows contain missing or invalid timestamps; "
            f"first offending row: {int(invalid_date_rows[0])}."
        )
    date_index = pd.DatetimeIndex(dates)
    duplicate_rows = np.flatnonzero(date_index.duplicated(keep=False))
    if duplicate_rows.size:
        raise ValueError(
            "The first 8,640 rows contain duplicate timestamps; "
            f"first offending row: {int(duplicate_rows[0])}, "
            f"timestamp: {date_index[duplicate_rows[0]]}."
        )

    deltas = date_index[1:] - date_index[:-1]
    non_hourly = np.flatnonzero(deltas != pd.Timedelta(hours=1))
    if non_hourly.size:
        left = int(non_hourly[0])
        raise ValueError(
            "The first 8,640 rows must be strictly ordered and hourly-contiguous; "
            f"rows {left} and {left + 1} differ by {deltas[left]}."
        )

    numeric = frame.loc[:, FEATURE_COLS].apply(pd.to_numeric, errors="coerce")
    missing_values = numeric.isna().to_numpy()
    if missing_values.any():
        row, column = np.argwhere(missing_values)[0]
        raise ValueError(
            "The first 8,640 rows contain missing or non-numeric feature values; "
            f"first offending cell: row {int(row)}, column {FEATURE_COLS[int(column)]}."
        )

    values = numeric.to_numpy(dtype=np.float32, copy=True)
    non_finite = ~np.isfinite(values)
    if non_finite.any():
        row, column = np.argwhere(non_finite)[0]
        raise ValueError(
            "The first 8,640 rows contain non-finite feature values; "
            f"first offending cell: row {int(row)}, column {FEATURE_COLS[int(column)]}."
        )
    return date_index, values


def _num_windows(num_points: int, seq_len: int, stride: int) -> int:
    if seq_len > num_points:
        return 0
    return (num_points - seq_len) // stride + 1


def _split_sample_indices(
    num_samples: int, val_start_ratio: float, val_end_ratio: float
) -> Tuple[np.ndarray, np.ndarray, int, int]:
    """Return ordered train/val sample IDs and the half-open val boundaries."""
    val_start = math.floor(val_start_ratio * num_samples)
    val_end = math.floor(val_end_ratio * num_samples)
    val_indices = np.arange(val_start, val_end, dtype=np.int64)
    train_indices = np.concatenate(
        (
            np.arange(0, val_start, dtype=np.int64),
            np.arange(val_end, num_samples, dtype=np.int64),
        )
    )
    if val_indices.size == 0:
        raise ValueError(
            "The selected validation ratios produce an empty validation split "
            f"for N={num_samples}."
        )
    if train_indices.size == 0:
        raise ValueError(
            "The selected validation ratios produce an empty training split "
            f"for N={num_samples}."
        )
    return train_indices, val_indices, val_start, val_end


def _window_view(values: np.ndarray, seq_len: int, stride: int) -> np.ndarray:
    """Return a read-only ``(N, seq_len, D)`` view over ``values``."""
    windows = np.lib.stride_tricks.sliding_window_view(
        values, window_shape=seq_len, axis=0
    )
    # NumPy places the windowed axis last: (N_all, D, seq_len).
    return np.moveaxis(windows[::stride], -1, 1)


def _output_paths(
    output_dir: Path, lookback: int, horizon: int
) -> Tuple[Path, Path, Path]:
    seq_len = lookback + horizon
    directory = Path(output_dir) / f"T{seq_len}"
    base = f"etth1_T{seq_len}_p{horizon}"
    return (
        directory / f"{base}_train.npz",
        directory / f"{base}_val.npz",
        directory / f"{base}_meta.json",
    )


def _build_metadata(
    csv_path: Path,
    dates: pd.DatetimeIndex,
    lookback: int,
    horizon: int,
    stride: int,
    val_start_ratio: float,
    val_end_ratio: float,
    num_samples: int,
    val_start: int,
    val_end: int,
    train_count: int,
    val_count: int,
) -> Dict[str, object]:
    seq_len = lookback + horizon
    return {
        "dataset": "ETTh1",
        "source_csv": str(Path(csv_path).resolve()),
        "source_row_range": [0, TRAIN_POINTS],
        "source_first_timestamp": dates[0].isoformat(),
        "source_last_timestamp": dates[-1].isoformat(),
        "source_frequency": "1h",
        "feature_cols": list(FEATURE_COLS),
        "dtype": "float32",
        "raw_physical_scale": True,
        "seq_len": seq_len,
        "lookback": lookback,
        "pred_len": horizon,
        "target_delay": 0,
        "layout": "aligned",
        "stride": stride,
        "N": num_samples,
        "total_samples": num_samples,
        "val_sample_start_ratio": val_start_ratio,
        "val_sample_end_ratio": val_end_ratio,
        "val_sample_start": val_start,
        "val_sample_end_exclusive": val_end,
        "train_sample_ranges": [[0, val_start], [val_end, num_samples]],
        "train_samples": train_count,
        "val_samples": val_count,
        "sample_index_semantics": "ordinal_in_full_sliding_window_set",
        "window_start_row_formula": "sample_index * stride",
    }


def _save_split(
    path: Path,
    data: np.ndarray,
    sample_indices: np.ndarray,
    metadata: Dict[str, object],
    split: str,
) -> None:
    split_metadata = dict(metadata)
    split_metadata["split"] = split
    split_metadata["split_samples"] = int(data.shape[0])
    np.savez_compressed(
        path,
        data=np.asarray(data, dtype=np.float32),
        sample_indices=np.asarray(sample_indices, dtype=np.int64),
        window_start_indices=np.asarray(
            sample_indices * int(metadata["stride"]), dtype=np.int64
        ),
        feature_cols=np.asarray(FEATURE_COLS),
        seq_len=np.asarray(metadata["seq_len"], dtype=np.int64),
        lookback=np.asarray(metadata["lookback"], dtype=np.int64),
        pred_len=np.asarray(metadata["pred_len"], dtype=np.int64),
        target_delay=np.asarray(0, dtype=np.int64),
        layout=np.asarray("aligned"),
        stride=np.asarray(metadata["stride"], dtype=np.int64),
        split=np.asarray(split),
        meta=np.asarray(json.dumps(split_metadata, sort_keys=True)),
    )


def convert(
    csv_path: Path,
    output_dir: Path,
    lookback: int = 336,
    horizons: Sequence[int] = DEFAULT_HORIZONS,
    val_start_ratio: float = 0.70,
    val_end_ratio: float = 0.85,
    stride: int = 1,
    overwrite: bool = False,
) -> List[Tuple[Path, Path, Path]]:
    """Convert ``csv_path`` and return train, val, metadata paths per horizon."""
    horizons = tuple(int(horizon) for horizon in horizons)
    _validate_options(
        lookback, horizons, val_start_ratio, val_end_ratio, stride
    )
    csv_path = Path(csv_path)
    output_dir = Path(output_dir)
    output_paths = [
        _output_paths(output_dir, lookback, horizon) for horizon in horizons
    ]

    if not overwrite:
        existing = [
            path
            for horizon_paths in output_paths
            for path in horizon_paths
            if path.exists()
        ]
        if existing:
            formatted = "\n".join(f"  {path}" for path in existing)
            raise FileExistsError(
                "Refusing to overwrite existing ETTh1 output(s). "
                "Pass --overwrite to replace them:\n"
                f"{formatted}"
            )

    dates, values = _load_training_prefix(csv_path)
    created: List[Tuple[Path, Path, Path]] = []
    for horizon, (train_path, val_path, meta_path) in zip(
        horizons, output_paths
    ):
        seq_len = lookback + horizon
        num_samples = _num_windows(TRAIN_POINTS, seq_len, stride)
        train_indices, val_indices, val_start, val_end = _split_sample_indices(
            num_samples, val_start_ratio, val_end_ratio
        )
        all_windows = _window_view(values, seq_len, stride)
        if all_windows.shape[0] != num_samples:
            raise RuntimeError(
                "Internal window-count mismatch: "
                f"expected {num_samples}, got {all_windows.shape[0]}."
            )
        train_data = all_windows[train_indices]
        val_data = all_windows[val_indices]

        metadata = _build_metadata(
            csv_path=csv_path,
            dates=dates,
            lookback=lookback,
            horizon=horizon,
            stride=stride,
            val_start_ratio=val_start_ratio,
            val_end_ratio=val_end_ratio,
            num_samples=num_samples,
            val_start=val_start,
            val_end=val_end,
            train_count=len(train_indices),
            val_count=len(val_indices),
        )
        train_path.parent.mkdir(parents=True, exist_ok=True)
        _save_split(train_path, train_data, train_indices, metadata, "train")
        _save_split(val_path, val_data, val_indices, metadata, "val")
        with meta_path.open("w", encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=2, sort_keys=True)
            handle.write("\n")

        print(
            f"[saved] H={horizon} T={seq_len} N={num_samples}: "
            f"train={train_data.shape} val={val_data.shape}"
        )
        print(f"        {train_path}")
        print(f"        {val_path}")
        print(f"        {meta_path}")
        created.append((train_path, val_path, meta_path))
    return created


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert the first 8,640 ETTh1 rows into auditable Diffusion-TS "
            "train/val NPZ files."
        )
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=DEFAULT_CSV,
        help=f"ETTh1 CSV path (default: {DEFAULT_CSV}).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output root; files are grouped under T<T> (default: {DEFAULT_OUTPUT_DIR}).",
    )
    parser.add_argument(
        "--lookback",
        type=int,
        default=336,
        help="Forecast lookback length (default: 336).",
    )
    parser.add_argument(
        "--horizons",
        type=int,
        nargs="+",
        default=list(DEFAULT_HORIZONS),
        help="Prediction horizons (default: 96 192 336 720).",
    )
    parser.add_argument(
        "--val-sample-start-ratio",
        type=float,
        default=0.70,
        help="Inclusive validation boundary on the sample axis (default: 0.70).",
    )
    parser.add_argument(
        "--val-sample-end-ratio",
        type=float,
        default=0.85,
        help="Exclusive validation boundary on the sample axis (default: 0.85).",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Sliding-window start stride (default: 1).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing train, val, and metadata files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    convert(
        csv_path=args.csv,
        output_dir=args.output_dir,
        lookback=args.lookback,
        horizons=args.horizons,
        val_start_ratio=args.val_sample_start_ratio,
        val_end_ratio=args.val_sample_end_ratio,
        stride=args.stride,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
