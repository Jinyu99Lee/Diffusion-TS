import os
import pickle

import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset

from Models.interpretable_diffusion.model_utils import (
    normalize_to_neg_one_to_one,
    unnormalize_to_zero_to_one,
)

SCALER_FNAME = "scaler.pkl"


# --------------------------------------------------------------------------- #
# Sample-axis split helpers (ported from timeVAE/src/data_utils.py so weather
# npz can be split in-framework with the same protocol).
# --------------------------------------------------------------------------- #
def _full_train_recent_blocks_valid_indices(num_samples: int) -> np.ndarray:
    """Validation indices from three recent-year 488-sample blocks.

    Mirrors timeVAE: each block spans 122 days at 4 samples/day.
    """
    expected_block_size = 488
    min_required_samples = 4384
    valid_slices = (
        slice(-488, None),
        slice(-1460 - 976, -1460 - 488),
        slice(-2920 - 1464, -2920 - 976),
    )
    if num_samples < min_required_samples:
        raise ValueError(
            "split_method='full_train_recent_blocks' requires at least "
            f"{min_required_samples} samples, got {num_samples}."
        )
    sample_indices = np.arange(num_samples)
    valid_indices = [sample_indices[s] for s in valid_slices]
    for idx, block in enumerate(valid_indices):
        if block.shape[0] != expected_block_size:
            raise ValueError(
                "Recent-block validation split produced an unexpected block size "
                f"for block {idx}: expected {expected_block_size}, got {block.shape[0]}."
            )
    return np.concatenate(valid_indices, axis=0)


def _split_data(data: np.ndarray, valid_perc: float, seed: int, split_method: str):
    """Return (train_data, valid_data) on the sample axis. Training is never
    shuffled here (the DataLoader shuffles); validation order is preserved."""
    if split_method == "full_train_recent_blocks":
        valid_idx = _full_train_recent_blocks_valid_indices(data.shape[0])
        train_mask = np.ones(data.shape[0], dtype=bool)
        train_mask[valid_idx] = False
        return data[train_mask].copy(), data[valid_idx].copy()
    if split_method != "tail_holdout":
        raise ValueError(
            f"Unknown split_method={split_method!r}. Expected 'tail_holdout' "
            "or 'full_train_recent_blocks'."
        )
    n = data.shape[0]
    n_train = int(n * (1 - valid_perc))
    return data[:n_train].copy(), data[n_train:].copy()


def _load_npz(path: str) -> np.ndarray:
    arr = np.load(path, allow_pickle=True)["data"]
    return np.asarray(arr, dtype=np.float64)


class NPZDataset(Dataset):
    """Pre-windowed ``N x T x D`` npz dataset for Diffusion-TS.

    Two input modes:

    * **explicit split** - provide ``train_path`` and ``val_path`` (used for ILI,
      where the standalone converter already produced the gen-train/gen-val npz);
    * **single npz split** - provide ``data_path`` plus ``split_method`` /
      ``valid_perc`` (used for weather, mirroring timeVAE's
      ``full_train_recent_blocks``).

    The ``MinMaxScaler`` is always fit on the **train** split only and applied to
    both splits (then optionally mapped to ``[-1, 1]``), so the validation set
    never leaks into normalisation. The fitted scaler is pickled to
    ``output_dir`` so final generation can inverse-transform.
    """

    def __init__(
        self,
        name="npz",
        train_path=None,
        val_path=None,
        data_path=None,
        split_method="tail_holdout",
        valid_perc=0.1,
        period="train",
        neg_one_to_one=True,
        seed=123,
        output_dir="./OUTPUT",
        save_scaler=True,
        **kwargs,
    ):
        super().__init__()
        assert period in ("train", "val"), "period must be 'train' or 'val'."
        self.name = name
        self.period = period
        self.auto_norm = neg_one_to_one

        if train_path is not None:
            train_raw = _load_npz(train_path)
            if val_path is not None:
                val_raw = _load_npz(val_path)
            else:  # split the train npz itself
                train_raw, val_raw = _split_data(train_raw, valid_perc, seed, split_method)
        elif data_path is not None:
            full = _load_npz(data_path)
            train_raw, val_raw = _split_data(full, valid_perc, seed, split_method)
        else:
            raise ValueError("Provide either train_path (+optional val_path) or data_path.")

        assert train_raw.ndim == 3, f"expected N x T x D, got {train_raw.shape}"
        self.window = int(train_raw.shape[1])
        self.var_num = int(train_raw.shape[2])
        assert val_raw.shape[1:] == train_raw.shape[1:], "train/val T,D mismatch"

        # Fit scaler on TRAIN only.
        self.scaler = MinMaxScaler()
        self.scaler.fit(train_raw.reshape(-1, self.var_num))
        if save_scaler and output_dir is not None:
            os.makedirs(output_dir, exist_ok=True)
            with open(os.path.join(output_dir, SCALER_FNAME), "wb") as fh:
                pickle.dump(self.scaler, fh)

        self.train_size = train_raw.shape[0]
        self.val_size = val_raw.shape[0]
        raw = train_raw if period == "train" else val_raw
        self.samples = self._normalize(raw)
        self.sample_num = self.samples.shape[0]

    # --- normalisation ---------------------------------------------------- #
    def _normalize(self, arr: np.ndarray) -> np.ndarray:
        d = self.scaler.transform(arr.reshape(-1, self.var_num))
        if self.auto_norm:
            d = normalize_to_neg_one_to_one(d)
        return d.reshape(-1, self.window, self.var_num)

    def unnormalize(self, arr: np.ndarray) -> np.ndarray:
        d = arr.reshape(-1, self.var_num)
        if self.auto_norm:
            d = unnormalize_to_zero_to_one(d)
        d = self.scaler.inverse_transform(d)
        return d.reshape(-1, self.window, self.var_num)

    # --- Dataset API ------------------------------------------------------ #
    def __getitem__(self, ind):
        x = self.samples[ind, :, :]
        return torch.from_numpy(x).float()

    def __len__(self):
        return self.sample_num
