#!/usr/bin/env python3
"""Grid-search HPO driver for Diffusion-TS (analogous to timeVAE's
``src/hpo_grid_search.py``).

For each combination of candidate hyper-parameters it launches ``main.py`` as a
subprocess to train one model, then reads that run's ``val_metrics.json`` (the
diffusion model's own validation loss - NO histogram loss). The run with the
lowest ``best_val_loss`` is written to ``best_run.json`` so it can later be
re-loaded by ``rerun_best_hpo.py`` to generate the final synthetic npz.

Dataset inputs:
  * Pre-split: --train-npz <train.npz> [--val-npz <val.npz>]
  * Single NPZ: --data-npz <full.npz> [--split-method <method>]

``seq_length`` (T) and ``feature_size`` (D) are inferred from the npz and
injected automatically, so the chosen base config need not match the data.
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np


REPO_ROOT = Path(__file__).resolve().parent

# (cli flag, config dotted-path, type) for the swept model/solver params.
SWEEP_SPECS = [
    ("d_model", "model.params.d_model", int),
    ("n_layer_enc", "model.params.n_layer_enc", int),
    ("n_layer_dec", "model.params.n_layer_dec", int),
    ("n_heads", "model.params.n_heads", int),
    ("mlp_hidden_times", "model.params.mlp_hidden_times", int),
    ("timesteps", "model.params.timesteps", int),
    ("sampling_timesteps", "model.params.sampling_timesteps", int),
    ("loss_type", "model.params.loss_type", str),
    ("beta_schedule", "model.params.beta_schedule", str),
    ("attn_pd", "model.params.attn_pd", float),
    ("resid_pd", "model.params.resid_pd", float),
    ("base_lr", "solver.base_lr", float),
    ("batch_size", "dataloader.batch_size", int),
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Grid-search HPO for Diffusion-TS")
    p.add_argument("--base-config", required=True, help="Base YAML configuration to clone.")
    # dataset inputs (one of the two modes)
    p.add_argument("--train-npz", default=None, help="Explicit training NPZ for pre-split input mode.")
    p.add_argument("--val-npz", default=None, help="Explicit validation NPZ used with --train-npz.")
    p.add_argument("--data-npz", default=None, help="Single NPZ to split into training and validation in-framework.")
    p.add_argument("--split-method", default="full_train_recent_blocks",
                   choices=("full_train_recent_blocks", "tail_holdout"),
                   help="Split method used with --data-npz.")
    p.add_argument("--valid-perc", type=float, default=0.1,
                   help="Validation fraction used with --data-npz.")
    # swept hyper-parameters (each accepts a list)
    for flag, _, typ in SWEEP_SPECS:
        p.add_argument(f"--{flag.replace('_', '-')}", dest=flag, type=typ, nargs="+", default=None)
    # fixed training controls
    p.add_argument("--max-epochs", type=int, default=12000)
    p.add_argument("--save-cycle", type=int, default=1200)
    p.add_argument("--val-num-repeats", type=int, default=3)
    p.add_argument("--seed", type=int, default=12345)
    # execution
    p.add_argument("--gpu-slots", default="0:1",
                   help="Concurrent jobs per GPU, e.g. '0:2,1:2' = 2 jobs each on GPU 0 and 1.")
    p.add_argument("--output-root", default="./outputs/hpo")
    p.add_argument("--experiment-name", default=None)
    p.add_argument("--dry-run", action="store_true", help="List jobs without running them.")
    args = p.parse_args()
    if args.data_npz is None and args.train_npz is None:
        p.error("Provide either --data-npz or --train-npz [--val-npz].")
    return args


def infer_t_d(args: argparse.Namespace) -> tuple[int, int]:
    path = args.train_npz or args.data_npz
    data = np.load(path, allow_pickle=True)["data"]
    return int(data.shape[1]), int(data.shape[2])


def parse_gpu_slots(spec: str) -> List[str]:
    """'0:2,1:1' -> ['0','0','1'] (one entry per concurrent slot)."""
    slots: List[str] = []
    for part in spec.split(","):
        gpu, _, n = part.partition(":")
        slots.extend([gpu.strip()] * (int(n) if n else 1))
    return slots


def build_jobs(args: argparse.Namespace, T: int, D: int) -> List[Dict[str, Any]]:
    grids = []
    active_specs = []
    for flag, dotted, typ in SWEEP_SPECS:
        values = getattr(args, flag)
        if values is None:
            continue
        active_specs.append((flag, dotted, typ))
        grids.append(values)
    jobs = []
    for idx, combo in enumerate(itertools.product(*grids)):
        overrides = {dotted: val for (_, dotted, _), val in zip(active_specs, combo)}
        tag = "_".join(f"{flag}{val}" for (flag, _, _), val in zip(active_specs, combo))
        run_id = f"run_{idx:05d}_{tag}" if tag else f"run_{idx:05d}"
        jobs.append({"run_id": run_id, "overrides": overrides})
    if not jobs:  # no grids -> single baseline run
        jobs.append({"run_id": "run_00000_baseline", "overrides": {}})
    return jobs


def make_opts(args, T, D, job, run_dir) -> List[str]:
    ckpt_root = os.path.join(run_dir, "ckpt")
    opts: Dict[str, Any] = {
        "model.params.seq_length": T,
        "model.params.feature_size": D,
        "solver.max_epochs": args.max_epochs,
        "solver.save_cycle": args.save_cycle,
        "solver.val_num_repeats": args.val_num_repeats,
        "solver.results_folder": ckpt_root,
    }
    # dataset paths
    if args.train_npz is not None:
        opts["dataloader.train_dataset.params.train_path"] = args.train_npz
        opts["dataloader.val_dataset.params.train_path"] = args.train_npz
        if args.val_npz is not None:
            opts["dataloader.train_dataset.params.val_path"] = args.val_npz
            opts["dataloader.val_dataset.params.val_path"] = args.val_npz
    else:
        for ds in ("train_dataset", "val_dataset"):
            opts[f"dataloader.{ds}.params.data_path"] = args.data_npz
            opts[f"dataloader.{ds}.params.split_method"] = args.split_method
            opts[f"dataloader.{ds}.params.valid_perc"] = args.valid_perc
    opts.update(job["overrides"])
    flat: List[str] = []
    for k, v in opts.items():
        flat.extend([k, str(v)])
    return flat, ckpt_root + f"_{T}"


def launch(args, job, T, D, gpu) -> Dict[str, Any]:
    run_dir = os.path.join(args.exp_dir, job["run_id"])
    os.makedirs(run_dir, exist_ok=True)
    opts, ckpt_dir = make_opts(args, T, D, job, run_dir)
    cmd = [
        sys.executable, str(REPO_ROOT / "main.py"),
        "--name", job["run_id"],
        "--config_file", args.base_config,
        "--output", run_dir,
        "--gpu", str(gpu),
        "--seed", str(args.seed),
        "--train",
    ] + opts
    log_path = os.path.join(run_dir, "train.log")
    job.update({"run_dir": run_dir, "ckpt_dir": ckpt_dir, "cmd": cmd,
                "log_path": log_path, "gpu": gpu, "opts": opts})
    log_file = open(log_path, "w")
    proc = subprocess.Popen(cmd, stdout=log_file, stderr=subprocess.STDOUT, cwd=str(REPO_ROOT))
    job["_proc"] = proc
    job["_log_file"] = log_file
    return job


def collect_result(job) -> Dict[str, Any]:
    val_path = os.path.join(job["ckpt_dir"], "val_metrics.json")
    best_val = float("inf")
    best_milestone = -1
    status = "ok"
    if job["_proc"].returncode != 0:
        status = f"failed(rc={job['_proc'].returncode})"
    if os.path.exists(val_path):
        with open(val_path) as f:
            m = json.load(f)
        best_val = float(m.get("best_val_loss", float("inf")))
        best_milestone = int(m.get("best_val_milestone", -1))
    else:
        status = status if status != "ok" else "no_val_metrics"
    return {
        "run_id": job["run_id"], "status": status, "gpu": job["gpu"],
        "best_val_loss": best_val, "best_val_milestone": best_milestone,
        "ckpt_dir": job["ckpt_dir"], "run_dir": job["run_dir"],
        "base_config": job_base_config, "opts": job["opts"],
        **{k: v for k, v in job["overrides"].items()},
    }


def main():
    args = parse_args()
    global job_base_config
    job_base_config = os.path.abspath(args.base_config)
    args.base_config = job_base_config

    T, D = infer_t_d(args)
    exp_name = args.experiment_name or datetime.now().strftime("%Y%m%d_%H%M%S")
    args.exp_dir = os.path.join(args.output_root, exp_name)
    os.makedirs(args.exp_dir, exist_ok=True)

    jobs = build_jobs(args, T, D)
    print(f"[hpo] T={T} D={D}  |  {len(jobs)} job(s)  ->  {args.exp_dir}")
    for j in jobs:
        print("   ", j["run_id"])
    if args.dry_run:
        return

    slots = parse_gpu_slots(args.gpu_slots)
    print(f"[hpo] gpu slots: {slots}")

    pending = list(jobs)
    running: List[Dict[str, Any]] = []
    free_slots = list(slots)
    results: List[Dict[str, Any]] = []

    while pending or running:
        while pending and free_slots:
            gpu = free_slots.pop(0)
            job = pending.pop(0)
            print(f"[launch] {job['run_id']} on gpu {gpu}")
            launch(args, job, T, D, gpu)
            running.append(job)
        still_running = []
        for job in running:
            if job["_proc"].poll() is None:
                still_running.append(job)
            else:
                job["_log_file"].close()
                res = collect_result(job)
                results.append(res)
                free_slots.append(job["gpu"])
                print(f"[done] {res['run_id']} status={res['status']} "
                      f"best_val_loss={res['best_val_loss']:.6f}")
        running = still_running
        if running:
            time.sleep(2)

    # write results table + best
    results_csv = os.path.join(args.exp_dir, "results.csv")
    fields = sorted({k for r in results for k in r.keys() if k != "opts"})
    with open(results_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in results:
            w.writerow(r)

    valid = [r for r in results if np.isfinite(r["best_val_loss"])]
    if not valid:
        print("[hpo] no successful runs with val metrics.")
        return
    best = min(valid, key=lambda r: r["best_val_loss"])
    best_payload = {
        "best_run_id": best["run_id"],
        "best_val_loss": best["best_val_loss"],
        "best_val_milestone": best["best_val_milestone"],
        "ckpt_dir": best["ckpt_dir"],
        "run_dir": best["run_dir"],
        "base_config": best["base_config"],
        "opts": best["opts"],
        "seq_length": T,
        "feature_size": D,
        "train_npz": args.train_npz,
        "val_npz": args.val_npz,
        "data_npz": args.data_npz,
        "split_method": args.split_method,
        "valid_perc": args.valid_perc,
    }
    best_path = os.path.join(args.exp_dir, "best_run.json")
    with open(best_path, "w") as f:
        json.dump(best_payload, f, indent=2)
    print(f"[hpo] best = {best['run_id']}  val_loss={best['best_val_loss']:.6f}")
    print(f"[hpo] wrote {best_path} and {results_csv}")


if __name__ == "__main__":
    main()
