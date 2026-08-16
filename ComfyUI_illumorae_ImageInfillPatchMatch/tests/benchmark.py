"""Benchmark runner.

Runs a method/parameter sweep over a synthetic dataset and writes:
 - results.csv          (one row per (method_cfg x sample))
 - summary.csv          (per-method aggregate)
 - previews/            (side-by-side PNGs of gt | corrupted | result)

Usage:
    python -m tests.benchmark --sweep techniques
    python -m tests.benchmark --sweep grid_small --size 80 --samples 1
    python -m tests.benchmark --sweep fix_verify --save-previews
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time
import traceback
from collections import defaultdict
from typing import Dict, List

import cv2
import numpy as np

# Allow `python tests/benchmark.py` direct execution
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PARENT = os.path.dirname(_THIS_DIR)
for p in (_PARENT, _THIS_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

from tests.dataset import build_dataset, save_sample_preview
from tests.methods import make_method_registry
from tests.metrics import compute_all
from tests.sweeps import SWEEPS


def _cfg_key(method: str, overrides: Dict) -> str:
    if not overrides:
        return method
    parts = [f"{k}={v}" for k, v in sorted(overrides.items())]
    return method + "[" + ",".join(parts) + "]"


def _save_preview(path: str, gt, corrupted, result, mask):
    gt_u8 = (np.clip(gt, 0, 1) * 255).astype(np.uint8)
    cor_u8 = (np.clip(corrupted, 0, 1) * 255).astype(np.uint8)
    res_u8 = (np.clip(result, 0, 1) * 255).astype(np.uint8)
    mask_rgb = np.stack([(mask * 255).astype(np.uint8)] * 3, axis=-1)
    row = np.concatenate([gt_u8, cor_u8, mask_rgb, res_u8], axis=1)
    row_bgr = cv2.cvtColor(row, cv2.COLOR_RGB2BGR)
    cv2.imwrite(path, row_bgr)


def run_benchmark(
    sweep_name: str,
    out_dir: str,
    size: int = 80,
    samples_per_combo: int = 1,
    gt_kinds: List[str] = None,
    mask_kinds: List[str] = None,
    coverages: List[float] = None,
    save_previews: bool = False,
    seed: int = 0,
) -> None:
    if sweep_name not in SWEEPS:
        raise SystemExit(f"Unknown sweep '{sweep_name}'. "
                         f"Options: {list(SWEEPS)}")
    sweep = SWEEPS[sweep_name]()
    registry = make_method_registry()

    dataset = build_dataset(
        size=size,
        gt_kinds=gt_kinds,
        mask_kinds=mask_kinds,
        coverages=coverages,
        samples_per_combo=samples_per_combo,
        seed=seed,
    )

    os.makedirs(out_dir, exist_ok=True)
    previews_dir = os.path.join(out_dir, "previews")
    if save_previews:
        os.makedirs(previews_dir, exist_ok=True)
        # save one preview of inputs
        gt_dir = os.path.join(out_dir, "dataset")
        for s in dataset[:4]:
            save_sample_preview(s, gt_dir)

    results_path = os.path.join(out_dir, "results.csv")
    summary_path = os.path.join(out_dir, "summary.csv")

    rows: List[Dict] = []
    total = len(sweep) * len(dataset)
    count = 0
    t_start = time.time()

    for method_name, overrides in sweep:
        if method_name not in registry:
            print(f"[skip] unknown method: {method_name}")
            continue
        method = registry[method_name]
        cfg_key = _cfg_key(method_name, overrides)
        for sample in dataset:
            count += 1
            params = dict(method.params)
            params.update(overrides)
            t0 = time.perf_counter()
            try:
                # Feed corrupted image so methods cannot cheat using target-region content
                pred = method.fn(sample.corrupted, sample.mask, **params)
            except Exception as e:  # pragma: no cover
                traceback.print_exc()
                print(f"[error] {cfg_key} on {sample.name}: {e}")
                continue
            elapsed = time.perf_counter() - t0
            pred = np.clip(pred.astype(np.float32), 0, 1)
            metrics = compute_all(pred, sample.gt, sample.mask)
            row = {
                "method_cfg": cfg_key,
                "method": method_name,
                "sample": sample.name,
                "gt_kind": sample.gt_kind,
                "mask_kind": sample.mask_kind,
                "coverage": sample.coverage,
                "runtime_s": round(elapsed, 4),
                **{k: round(v, 6) for k, v in metrics.items()},
            }
            rows.append(row)
            if save_previews:
                subdir = os.path.join(previews_dir, cfg_key.replace("/", "_"))
                os.makedirs(subdir, exist_ok=True)
                _save_preview(
                    os.path.join(subdir, f"{sample.name}.png"),
                    sample.gt, sample.corrupted, pred, sample.mask,
                )
            if count % 5 == 0 or count == total:
                avg = (time.time() - t_start) / count
                eta = avg * (total - count)
                print(f"[{count}/{total}] {cfg_key} on {sample.name} "
                      f"psnr={metrics['psnr']:.2f} "
                      f"ssim={metrics['ssim']:.3f} "
                      f"mse={metrics['mse']:.4f} "
                      f"t={elapsed:.2f}s   eta={eta:.0f}s")

    # Write results
    if not rows:
        print("No rows produced.")
        return
    fieldnames = list(rows[0].keys())
    with open(results_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {results_path} ({len(rows)} rows)")

    # Aggregate
    groups: Dict[str, List[Dict]] = defaultdict(list)
    for r in rows:
        groups[r["method_cfg"]].append(r)

    summary = []
    for cfg, rs in groups.items():
        summary.append({
            "method_cfg": cfg,
            "n": len(rs),
            "psnr_mean": round(float(np.mean([r["psnr"] for r in rs])), 3),
            "ssim_mean": round(float(np.mean([r["ssim"] for r in rs])), 4),
            "mse_mean": round(float(np.mean([r["mse"] for r in rs])), 5),
            "mae_mean": round(float(np.mean([r["mae"] for r in rs])), 5),
            "runtime_s_mean": round(float(np.mean([r["runtime_s"] for r in rs])), 3),
        })
    summary.sort(key=lambda r: -r["psnr_mean"])
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        w.writeheader()
        w.writerows(summary)
    print(f"Wrote {summary_path} ({len(summary)} rows)")

    # Print top-N to stdout
    print("\n=== Ranking (by mean PSNR) ===")
    print(f"{'rank':<5}{'method_cfg':<50}{'psnr':>7}{'ssim':>8}{'mse':>9}{'t(s)':>8}")
    for i, s in enumerate(summary, 1):
        print(f"{i:<5}{s['method_cfg'][:48]:<50}"
              f"{s['psnr_mean']:>7.2f}{s['ssim_mean']:>8.3f}"
              f"{s['mse_mean']:>9.4f}{s['runtime_s_mean']:>8.2f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sweep", required=True, choices=list(SWEEPS.keys()))
    ap.add_argument("--out", default=os.path.join(_THIS_DIR, "out"))
    ap.add_argument("--size", type=int, default=80,
                    help="Synthetic image side length (px).")
    ap.add_argument("--samples", type=int, default=1,
                    help="Samples per (gt_kind x mask_kind x coverage).")
    ap.add_argument("--gt", nargs="*", default=None,
                    help="Subset of gt kinds (gradient, checker, noise, stripes, mixed)")
    ap.add_argument("--masks", nargs="*", default=None,
                    help="Subset of mask kinds (rect, ellipse, blob, strip)")
    ap.add_argument("--coverages", nargs="*", type=float, default=None)
    ap.add_argument("--save-previews", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out_dir = os.path.join(args.out, args.sweep)
    run_benchmark(
        sweep_name=args.sweep,
        out_dir=out_dir,
        size=args.size,
        samples_per_combo=args.samples,
        gt_kinds=args.gt,
        mask_kinds=args.masks,
        coverages=args.coverages,
        save_previews=args.save_previews,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
