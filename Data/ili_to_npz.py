#!/usr/bin/env python3
"""Convert a raw ILI CSV into train / val npz files for Diffusion-TS.

This is a *standalone* utility (it does not import the Sonnet package). It
mirrors the onset/peak/offset season logic of
``Sonnet/sonnet/data/iliDataloader.py`` so that the generative model's
train/val split stays consistent with - but strictly separate from - Sonnet's
prediction train/val/test split.

For a real Sonnet test period ``Y / Y+1`` (``--test-start-year Y``):

  * gen-train uses the four seasons ``[Y-1, Y-2, Y-3, Y-4]``;
  * the time points that Sonnet would use as the *validation* set for its real
    test period - ``onset(Y-1)``, ``peak(Y-2)``, ``offset(Y-3)`` - are EXCLUDED
    (they would otherwise leak Sonnet's val into the generative train set);
  * the generative *val* set is the Sonnet-style val for a shadow test period
    ``Y-2``: ``onset(Y-3)``, ``peak(Y-4)``, ``offset(Y-5)``.

Samples are sliced within contiguous daily runs only (never crossing a date
gap, a season boundary, or an excluded/val window boundary). Output arrays are
raw (unnormalised) float32 of shape ``N x T x D`` stored under key ``data``;
normalisation is the framework's job so the scaler is fit on train only.
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------- #
# Region conventions
# --------------------------------------------------------------------------- #
DEFAULT_CSV = {
    "eng": "datasets/ILI/raw/eng_ILI 1.csv",
    "us2": "datasets/ILI/raw/us2_ILI.csv",
    "us9": "datasets/ILI/raw/us9_ILI.csv",
    "us10": "datasets/ILI/raw/us10_ILI.csv",
}
# (baseline file, row label) for onset/offset thresholds.
BASELINE_ROW = {
    "eng": ("ENG_Baseline.csv", "ENG"),
    "us2": ("US_Baseline.csv", "Region2"),
    "us9": ("US_Baseline.csv", "Region9"),
    "us10": ("US_Baseline.csv", "Region10"),
}


def _season_bounds(region: str, start_year: int) -> Tuple[pd.Timestamp, pd.Timestamp]:
    """Season boundaries (mirrors Sonnet ``_season_bounds``)."""
    if region.lower() == "eng":
        start = pd.Timestamp(year=start_year, month=9, day=1)
    else:  # US regions start in August
        start = pd.Timestamp(year=start_year, month=8, day=1)
    end = start + pd.DateOffset(years=1) - pd.Timedelta(days=1)
    return start, end


def _season_period(start_year: int) -> str:
    return f"{start_year}/{start_year + 1}"


# --------------------------------------------------------------------------- #
# Thresholds + onset/peak/offset point selection (mirrors Sonnet)
# --------------------------------------------------------------------------- #
def _lookup_threshold(
    baseline_dir: str, region: str, period: str
) -> Optional[float]:
    fname, row_name = BASELINE_ROW[region]
    path = os.path.join(baseline_dir, fname)
    table = pd.read_csv(path, index_col=0)
    if row_name not in table.index or period not in table.columns:
        return None
    value = table.loc[row_name, period]
    if pd.isna(value):
        return None
    return float(value)


def _select_season_point(
    season: pd.DataFrame, target_column: str, event: str, threshold: Optional[float]
) -> pd.Timestamp:
    """Onset/peak/offset point for a season (mirrors Sonnet logic)."""
    target = season[target_column]
    if event == "peak":
        return target.idxmax()
    if threshold is None:
        # Sonnet would raise; we fall back to the peak so the converter is robust
        # to missing baselines for very old seasons.
        return target.idxmax()
    above = target > threshold
    if event == "onset":
        for i in range(0, max(len(above) - 13, 0)):
            if bool(above.iloc[i : i + 14].all()):
                return above.index[i]
        return target.idxmax()
    if event == "offset":  # "outset" in Sonnet
        for i in range(len(above) - 1, 12, -1):
            if bool(above.iloc[i - 13 : i + 1].all()):
                return above.index[i]
        return target.idxmax()
    raise ValueError(f"Unknown validation event: {event}")


def _window_around_point(
    point: pd.Timestamp,
    season_start: pd.Timestamp,
    season_end: pd.Timestamp,
    days: int,
) -> Tuple[pd.Timestamp, pd.Timestamp]:
    """60-day (default) window around a point, clamped to the season."""
    before = days // 2 - 1
    start = point - pd.Timedelta(days=before)
    end = point + pd.Timedelta(days=days - before - 1)
    if start < season_start:
        start = season_start
        end = start + pd.Timedelta(days=days - 1)
    if end > season_end:
        end = season_end
        start = end - pd.Timedelta(days=days - 1)
    return start, end


def _event_window(
    data: pd.DataFrame,
    region: str,
    target_column: str,
    baseline_dir: str,
    season_start_year: int,
    event: str,
    days: int,
) -> Optional[Dict]:
    start, end = _season_bounds(region, season_start_year)
    season = data.loc[start:end]
    if season.empty or season[target_column].dropna().empty:
        return None
    period = _season_period(season_start_year)
    threshold = None
    if event in {"onset", "offset"}:
        threshold = _lookup_threshold(baseline_dir, region, period)
    point = _select_season_point(season, target_column, event, threshold)
    w_start, w_end = _window_around_point(point, start, end, days)
    return {
        "event": event,
        "period": period,
        "threshold": threshold,
        "point": point.strftime("%Y-%m-%d"),
        "start": w_start.strftime("%Y-%m-%d"),
        "end": w_end.strftime("%Y-%m-%d"),
        "_start": w_start,
        "_end": w_end,
    }


# --------------------------------------------------------------------------- #
# Feature selection (reads Sonnet's precomputed cache)
# --------------------------------------------------------------------------- #
def _load_selected_features(
    feature_selection_dir: str,
    region: str,
    fs_start_year: int,
    fs_end_year: int,
    tau: Optional[float],
    corr_threshold: Optional[float],
) -> List[str]:
    """Load the selected feature columns from the cache of period
    ``fs_start_year/fs_end_year``. This period is the *feature-selection
    reference* period, which may differ from the real test period so that the
    same tau feature set can be reused across different test periods."""
    season = f"{fs_start_year}_{fs_end_year}"
    season_dir = os.path.join(feature_selection_dir, region, season)
    if tau is not None:
        # Match Sonnet's file naming, e.g. tau_0.05.txt / tau_0.3.txt.
        tau_str = ("%g" % float(tau))
        path = os.path.join(season_dir, f"tau_{tau_str}.txt")
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Feature list not found: {path}. Available files: "
                f"{sorted(os.listdir(season_dir)) if os.path.isdir(season_dir) else 'NONE'}"
            )
        with open(path, "r", encoding="utf-8") as handle:
            return [line.strip() for line in handle if line.strip()]
    # Fall back to correlations.csv + a raw correlation threshold.
    corr_path = os.path.join(season_dir, "correlations.csv")
    table = pd.read_csv(corr_path)
    if corr_threshold is None:
        return list(table["column"].values)
    keep = table[table["abs_corr"] >= float(corr_threshold)]
    return list(keep["column"].values)


# --------------------------------------------------------------------------- #
# Windowing - mirrors Sonnet CustomILIDataset._build_valid_start_indices.
# A window of length T = lookback + pred is included when (a) the FULL T-window
# is daily-contiguous and (b) only the predict TAIL [start+lookback : start+T]
# lands entirely in the target mask; the lookback prefix extends backward and is
# NOT required to be in the target mask. Optional ``forbid_mask`` drops windows
# whose full T-span intersects forbidden dates (the train-vs-val/excluded guard).
# --------------------------------------------------------------------------- #
def _collect_windows(
    values: np.ndarray,
    index: pd.DatetimeIndex,
    target_mask: np.ndarray,
    seq_len: int,
    pred_len: int,
    stride: int,
    forbid_mask: np.ndarray = None,
) -> List[np.ndarray]:
    """``values`` is ``(len(index), D)``. Returns a list of ``(seq_len, D)`` windows."""
    one_day = pd.Timedelta(days=1)
    n = len(index)
    lookback = seq_len - pred_len
    # daily-step flag between consecutive rows (with present values on both ends)
    step_ok = np.zeros(n, dtype=bool)
    deltas = (index[1:] - index[:-1]) == one_day
    step_ok[1:] = deltas
    samples: List[np.ndarray] = []
    max_start = n - seq_len
    for start in range(0, max_start + 1, stride):
        end = start + seq_len  # exclusive
        # full T-window must be daily-contiguous (steps start+1 .. end-1 all == 1 day)
        if not step_ok[start + 1 : end].all():
            continue
        window = values[start:end]
        if np.isnan(window).any():
            continue
        # predict tail must lie entirely in the target mask
        if not target_mask[start + lookback : end].all():
            continue
        # guard: full T-span must avoid forbidden dates
        if forbid_mask is not None and forbid_mask[start:end].any():
            continue
        samples.append(window)
    return samples


def _mask_for_ranges(
    index: pd.DatetimeIndex, ranges: List[Tuple[pd.Timestamp, pd.Timestamp]]
) -> np.ndarray:
    mask = np.zeros(len(index), dtype=bool)
    for start, end in ranges:
        mask |= (index >= start) & (index <= end)
    return mask


# --------------------------------------------------------------------------- #
# Main conversion
# --------------------------------------------------------------------------- #
def convert(args: argparse.Namespace) -> None:
    region = args.region
    target_column = args.target_column
    Y = args.test_start_year
    test_end_year = Y + 1

    project_root = args.project_root
    csv_path = args.csv or os.path.join(project_root, DEFAULT_CSV[region])
    baseline_dir = (
        args.threshold_dir
        if os.path.isabs(args.threshold_dir)
        else os.path.join(project_root, args.threshold_dir)
    )
    fs_dir = (
        args.feature_selection_dir
        if os.path.isabs(args.feature_selection_dir)
        else os.path.join(project_root, args.feature_selection_dir)
    )

    # --- feature columns -------------------------------------------------- #
    # Feature-selection reference period: defaults to the real test period Y,
    # but can be pinned to a different period so the tau feature set stays
    # consistent across test periods (their per-period tau lists differ).
    ref_start = args.reference_year if args.reference_year is not None else Y
    ref_end = ref_start + 1
    selected = _load_selected_features(
        fs_dir, region, ref_start, ref_end, args.tau, args.corr_threshold
    )
    feature_cols = list(selected) + [target_column]  # target last
    ref_note = (
        f" [reference period {ref_start}/{ref_end}]"
        if ref_start != Y else " [test period's own feature set]"
    )
    print(f"[features] D = {len(feature_cols)} ({len(selected)} trends + target)"
          f"{ref_note}")

    # --- load only the columns we need ------------------------------------ #
    usecols = ["date"] + feature_cols
    df = pd.read_csv(csv_path, usecols=lambda c: c in set(usecols))
    missing = [c for c in usecols if c not in df.columns]
    if missing:
        raise ValueError(f"CSV {csv_path} is missing columns: {missing[:10]} ...")
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").set_index("date")
    df = df[feature_cols]  # enforce column order (target last)
    index = df.index
    values = df.values.astype(np.float64)

    # --- seasons & windows ------------------------------------------------ #
    train_season_years = [Y - 1, Y - 2, Y - 3, Y - 4]
    train_ranges = [_season_bounds(region, y) for y in train_season_years]

    days = args.window_days
    excluded_specs = [(Y - 1, "onset"), (Y - 2, "peak"), (Y - 3, "offset")]
    val_specs = [(Y - 3, "onset"), (Y - 4, "peak"), (Y - 5, "offset")]

    excluded = [
        w for (y, e) in excluded_specs
        if (w := _event_window(df, region, target_column, baseline_dir, y, e, days))
    ]
    val_windows = [
        w for (y, e) in val_specs
        if (w := _event_window(df, region, target_column, baseline_dir, y, e, days))
    ]

    print("[excluded windows] (Sonnet val for real test period "
          f"{Y}/{test_end_year}):")
    for w in excluded:
        print(f"   {w['event']:6s} {w['period']}  {w['start']} .. {w['end']}")
    print("[gen-val windows] (Sonnet val for shadow test period "
          f"{Y-2}/{Y-1}):")
    for w in val_windows:
        print(f"   {w['event']:6s} {w['period']}  {w['start']} .. {w['end']}")

    # --- build masks ------------------------------------------------------ #
    season_mask = _mask_for_ranges(index, train_ranges)
    excluded_mask = _mask_for_ranges(index, [(w["_start"], w["_end"]) for w in excluded])
    genval_mask = _mask_for_ranges(index, [(w["_start"], w["_end"]) for w in val_windows])

    # ``--seq-len`` is the lookback/context length; the stored window is the
    # full T = lookback + pred, so the saved NPZ time dimension is
    # ``seq_len + pred_len`` (NOT seq_len with pred carved out of it).
    lookback, pred_len, stride = args.seq_len, args.pred_len, args.stride
    total_len = lookback + pred_len
    if lookback <= 0 or pred_len <= 0:
        raise ValueError(f"--seq-len ({lookback}) and --pred-len ({pred_len}) must both be > 0.")
    print(f"[window] T(seq_length)={total_len}  lookback(seq_len)={lookback}  pred_len={pred_len}  stride={stride}")

    # Sonnet rule: only the predict tail must land in the target region; the
    # lookback prefix extends backward freely.
    #  * train: tail in the 4 seasons, full T-span avoids gen-val AND excluded
    #           (the guard - protects val integrity, since the whole window is a
    #            generative training example).
    #  * val:   tail in a gen-val window, full T-span avoids excluded windows
    #           (lookback may reuse train-season dates, which is acceptable).
    train_samples = _collect_windows(
        values, index, season_mask, total_len, pred_len, stride,
        forbid_mask=(excluded_mask | genval_mask),
    )
    val_samples = _collect_windows(
        values, index, genval_mask, total_len, pred_len, stride,
        forbid_mask=excluded_mask,
    )

    if not train_samples:
        raise RuntimeError("No train samples produced - check seq_len/pred_len vs segment lengths.")
    if not val_samples:
        raise RuntimeError("No val samples produced - check seq_len/pred_len vs window length.")

    train_arr = np.stack(train_samples).astype(np.float32)
    val_arr = np.stack(val_samples).astype(np.float32)
    print(f"[shapes] train {train_arr.shape}  val {val_arr.shape}")

    # --- save ------------------------------------------------------------- #
    os.makedirs(args.output_dir, exist_ok=True)
    base = args.name or (
        f"ili_{region}_{Y}_{test_end_year}_T{total_len}_p{pred_len}"
        + (f"_ref{ref_start}_{ref_end}" if ref_start != Y else "")
    )
    meta = dict(
        region=region,
        test_period=f"{Y}/{test_end_year}",
        shadow_val_period=f"{Y-2}/{Y-1}",
        seq_len=total_len,
        lookback=lookback,
        pred_len=pred_len,
        stride=stride,
        window_days=days,
        tau=args.tau,
        corr_threshold=args.corr_threshold,
        feature_reference_period=f"{ref_start}/{ref_end}",
        train_seasons=[_season_period(y) for y in train_season_years],
        excluded_windows=[{k: w[k] for k in ("event", "period", "start", "end")} for w in excluded],
        val_windows=[{k: w[k] for k in ("event", "period", "start", "end")} for w in val_windows],
    )
    for split, arr in (("train", train_arr), ("val", val_arr)):
        out = os.path.join(args.output_dir, f"{base}_{split}.npz")
        np.savez_compressed(
            out,
            data=arr,
            feature_cols=np.array(feature_cols),
            seq_len=np.array(total_len),
            stride=np.array(stride),
            region=np.array(region),
            meta=np.array(json.dumps(meta)),
        )
        print(f"[saved] {out}  ({arr.shape})")
    with open(os.path.join(args.output_dir, f"{base}_meta.json"), "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="ILI csv -> train/val npz for Diffusion-TS")
    p.add_argument("--region", required=True, choices=sorted(DEFAULT_CSV.keys()))
    p.add_argument("--csv", default=None, help="Raw ILI csv (defaults per region).")
    p.add_argument("--test-start-year", type=int, default=2015,
                   help="Real Sonnet test period anchor Y (test period Y/Y+1). "
                        "Drives the gen-train/val seasons and windows.")
    p.add_argument("--reference-year", type=int, default=None,
                   help="Feature-selection reference period anchor R: features are read "
                        "from feature_selection/<region>/<R>_<R+1>/ instead of the test "
                        "period's own cache. Defaults to --test-start-year. Pin this to "
                        "fix one tau feature set across different test periods (per-period "
                        "tau lists differ, e.g. 2015_2016 vs 2016_2017 at tau=0.5).")
    p.add_argument("--seq-len", type=int, required=True,
                   help="Lookback/context length (= the lookback prefix). The stored window "
                        "is the full T = seq_len + pred_len, so the saved NPZ time dimension "
                        "is seq_len + pred_len.")
    p.add_argument("--pred-len", type=int, required=True,
                   help="Predict tail length P; only this tail must fall in the target window. "
                        "Appended after the seq_len lookback (T = seq_len + P).")
    p.add_argument("--stride", type=int, default=1, help="Window-start stride (default 1).")
    p.add_argument("--window-days", type=int, default=60,
                   help="onset/peak/offset window length in days (Sonnet uses 60).")
    p.add_argument("--target-column", default="rate")
    p.add_argument("--tau", type=float, default=0.3,
                   help="Correlation tau -> reads tau_<tau>.txt from the cache.")
    p.add_argument("--corr-threshold", type=float, default=None,
                   help="Alternative to --tau: filter correlations.csv by abs_corr.")
    p.add_argument("--feature-selection-dir", default="datasets/ILI/feature_selection")
    p.add_argument("--threshold-dir", default="datasets/ILI/threshold")
    p.add_argument("--project-root", default="/data/jinyuli/Projects/Sonnet",
                   help="Root used to resolve default csv / threshold / feature dirs.")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--name", default=None, help="Output basename (default ili_<region>_<Y>_<Y+1>_T<T>).")
    args = p.parse_args()
    if args.tau is not None and args.corr_threshold is not None:
        # explicit corr-threshold wins; disable tau-file path
        args.tau = None
    return args


if __name__ == "__main__":
    convert(parse_args())
