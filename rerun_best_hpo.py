#!/usr/bin/env python3
"""Load the best HPO config + weights and generate the final synthetic npz.

Reads ``best_run.json`` produced by ``hpo_grid_search.py``, rebuilds the model
from that run's config, loads its best checkpoint (EMA weights selected by the
lowest validation loss), and generates a synthetic set with the **same number
of samples as the generative training set**. Output is inverse-transformed back
to the original data scale and saved as ``N_train x T x D`` under key ``data``.
"""
from __future__ import annotations

import argparse
import json
import os
from types import SimpleNamespace

import numpy as np
import torch

from engine.solver import Trainer
from Data.build_dataloader import build_dataloader
from Utils.io_utils import load_yaml_config, merge_opts_to_config, instantiate_from_config, seed_everything

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))

# Window-geometry keys the converters (ili_to_npz.py / ili_iaaft_to_npz.py)
# embed in the training npz meta. Sonnet's synthetic loader rejects delta
# ('target_shifted') windows unless best_synth.npz carries them too.
GEOMETRY_META_KEYS = ("layout", "target_delay", "lookback", "pred_len")


def train_npz_geometry_meta(best: dict) -> dict:
    """Read the geometry meta out of the training npz named in best_run.json."""
    train_npz = best.get("train_npz")
    if not train_npz:
        return {}
    path = train_npz if os.path.isabs(train_npz) else os.path.join(REPO_ROOT, train_npz)
    if not os.path.exists(path):
        print(f"[warn] train npz not found ({path}); best_synth meta will lack "
              f"{'/'.join(GEOMETRY_META_KEYS)}")
        return {}
    with np.load(path, allow_pickle=True) as payload:
        if "meta" not in payload.files:
            return {}
        try:
            meta = json.loads(str(payload["meta"]))
        except (json.JSONDecodeError, TypeError):
            print(f"[warn] cannot parse meta in {path}; ignoring")
            return {}
    return {k: meta[k] for k in GEOMETRY_META_KEYS if k in meta}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate final synthetic npz from best HPO run.")
    p.add_argument("--hpo-dir", default=None, help="Experiment dir containing best_run.json.")
    p.add_argument("--best-json", default=None, help="Path to best_run.json (overrides --hpo-dir).")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument("--num-samples", default="train",
                   help="'train' (= gen-train set size) or an integer count.")
    p.add_argument("--size-every", type=int, default=2001)
    p.add_argument("--output", default=None, help="Output npz path (default <hpo-dir>/best_synth.npz).")
    args = p.parse_args()
    if args.best_json is None:
        if args.hpo_dir is None:
            p.error("Provide --hpo-dir or --best-json.")
        args.best_json = os.path.join(args.hpo_dir, "best_run.json")
    return args


def main():
    args = parse_args()
    seed_everything(args.seed)
    torch.cuda.set_device(args.gpu)

    with open(args.best_json) as f:
        best = json.load(f)

    config = load_yaml_config(best["base_config"])
    config = merge_opts_to_config(config, best["opts"])

    save_dir = os.path.join(os.path.dirname(args.best_json), best["best_run_id"] + "_gen")
    os.makedirs(save_dir, exist_ok=True)
    fake_args = SimpleNamespace(name=best["best_run_id"], save_dir=save_dir)

    # Train dataset gives us the fitted scaler and N_train.
    dl_info = build_dataloader(config, fake_args)
    dataset = dl_info["dataset"]
    n_train = len(dataset)
    if str(args.num_samples) == "train":
        num = n_train
    else:
        num = int(args.num_samples)

    model = instantiate_from_config(config["model"]).cuda()
    trainer = Trainer(config=config, args=fake_args, model=model, dataloader=dl_info, logger=None)
    trainer.load("best")

    print(f"[gen] generating {num} samples of shape ({dataset.window}, {dataset.var_num}) "
          f"(gen-train size = {n_train})")
    samples = trainer.sample(num=num, size_every=args.size_every,
                             shape=[dataset.window, dataset.var_num])
    samples = samples[:num]
    # Inverse [-1,1] -> [0,1] -> original scale via the train-fit scaler.
    samples = dataset.unnormalize(samples)

    out = args.output or os.path.join(os.path.dirname(args.best_json), "best_synth.npz")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    geometry = train_npz_geometry_meta(best)
    if geometry:
        print(f"[gen] geometry meta from train npz: {geometry}")
    np.savez_compressed(out, data=samples.astype(np.float32),
                        meta=np.array(json.dumps({
                            "best_run_id": best["best_run_id"],
                            "best_val_loss": best["best_val_loss"],
                            "n_train": n_train,
                            "num_generated": int(samples.shape[0]),
                            "seq_length": best["seq_length"],
                            "feature_size": best["feature_size"],
                            **geometry,
                        })))
    print(f"[gen] saved {out}  shape={samples.shape}")


if __name__ == "__main__":
    main()
