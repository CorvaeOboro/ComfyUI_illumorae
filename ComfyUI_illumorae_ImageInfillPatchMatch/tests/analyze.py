"""Post-process results.csv / summary.csv to recommend new node defaults.

Usage:
    python -m tests.analyze --results tests/out/grid_small/results.csv
"""
from __future__ import annotations

import argparse
import csv
import os
import re
from collections import defaultdict
from typing import Dict, List, Tuple


def _parse_cfg(method_cfg: str) -> Tuple[str, Dict[str, str]]:
    m = re.match(r"^([^\[]+)(?:\[(.*)\])?$", method_cfg.strip())
    if not m:
        return method_cfg, {}
    name, body = m.group(1), m.group(2) or ""
    params: Dict[str, str] = {}
    if body:
        for kv in body.split(","):
            if "=" in kv:
                k, v = kv.split("=", 1)
                params[k.strip()] = v.strip()
    return name, params


def load_rows(path: str) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def recommend_defaults(rows: List[Dict], method_filter: str = "core_full_fixed") -> Dict:
    """Find the best (patch_size, iterations, search_radius) combo
    by average PSNR, then by SSIM, then by runtime."""
    groups: Dict[str, List[Dict]] = defaultdict(list)
    for r in rows:
        name, params = _parse_cfg(r["method_cfg"])
        if method_filter and name != method_filter:
            continue
        key = tuple(sorted(params.items()))
        groups[str(key)].append((r, params))

    scored = []
    for key, items in groups.items():
        psnrs = [float(r["psnr"]) for r, _ in items]
        ssims = [float(r["ssim"]) for r, _ in items]
        runtimes = [float(r["runtime_s"]) for r, _ in items]
        scored.append({
            "cfg": items[0][1],
            "n": len(items),
            "psnr_mean": sum(psnrs) / len(psnrs),
            "ssim_mean": sum(ssims) / len(ssims),
            "runtime_s_mean": sum(runtimes) / len(runtimes),
        })
    scored.sort(key=lambda r: (-r["psnr_mean"], -r["ssim_mean"], r["runtime_s_mean"]))

    if not scored:
        return {}
    print("\n=== Top 10 configurations ===")
    print(f"{'rank':<5}{'cfg':<50}{'psnr':>7}{'ssim':>8}{'t(s)':>8}")
    for i, s in enumerate(scored[:10], 1):
        cfg_s = ",".join(f"{k}={v}" for k, v in sorted(s["cfg"].items()))
        print(f"{i:<5}{cfg_s[:48]:<50}{s['psnr_mean']:>7.2f}"
              f"{s['ssim_mean']:>8.3f}{s['runtime_s_mean']:>8.2f}")

    best = scored[0]
    print("\n=== Recommended defaults ===")
    for k, v in sorted(best["cfg"].items()):
        print(f"  {k} = {v}")
    print(f"  (avg PSNR {best['psnr_mean']:.2f} dB, "
          f"SSIM {best['ssim_mean']:.3f}, "
          f"runtime {best['runtime_s_mean']:.2f}s)")
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--method", default="core_full_fixed",
                    help="Restrict recommendation to this method name.")
    args = ap.parse_args()
    if not os.path.isfile(args.results):
        raise SystemExit(f"No such file: {args.results}")
    rows = load_rows(args.results)
    recommend_defaults(rows, args.method)


if __name__ == "__main__":
    main()
