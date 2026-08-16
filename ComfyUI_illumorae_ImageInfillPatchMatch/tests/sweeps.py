"""Parameter-sweep configurations.

A sweep is a list of (method_name, params_override_dict).
Used by `benchmark.py --sweep <name>`.
"""
from __future__ import annotations

from itertools import product
from typing import Dict, List, Tuple

Sweep = List[Tuple[str, Dict]]


def sweep_patch_size() -> Sweep:
    out: Sweep = []
    for ps in [3, 5, 7, 9, 11, 13, 17, 21]:
        out.append(("core_full_fixed", {"patch_size": ps}))
    return out


def sweep_iterations() -> Sweep:
    out: Sweep = []
    for it in [1, 2, 3, 5, 8, 12]:
        out.append(("core_full_fixed", {"iterations": it}))
    return out


def sweep_search_radius() -> Sweep:
    out: Sweep = []
    for sr in [10, 20, 40, 60, 100, 160]:
        out.append(("core_full_fixed", {"search_radius": sr}))
    return out


def sweep_full_grid_small() -> Sweep:
    """Coarse joint grid; still small enough to run on 96x96 in minutes."""
    out: Sweep = []
    for ps, it, sr in product(
        [5, 7, 11],
        [2, 4, 6],
        [20, 50, 100],
    ):
        out.append(("core_full_fixed", {
            "patch_size": ps, "iterations": it, "search_radius": sr,
        }))
    return out


def sweep_techniques() -> Sweep:
    """Compare technique wedges at a common reference budget."""
    common = {"patch_size": 7, "iterations": 5, "search_radius": 50}
    names = [
        "baseline_mean",
        "baseline_nearest",
        "baseline_telea_r3",
        "baseline_ns_r3",
        "stock_defaults",
        "core_mimic_stock",
        "core_full_fixed",
        "core_no_random_search",
        "core_no_propagation",
        "core_forward_only",
        "core_multiscale_2",
        "core_multiscale_3",
    ]
    return [(n, dict(common)) for n in names]


def sweep_fix_verify() -> Sweep:
    """Specifically validate: does fixing the backward-propagation bug help?"""
    out: Sweep = []
    common = {"patch_size": 7, "iterations": 5, "search_radius": 50}
    out.append(("core_mimic_stock", dict(common)))
    out.append(("core_full_fixed", dict(common)))
    return out


SWEEPS: Dict[str, callable] = {
    "patch_size": sweep_patch_size,
    "iterations": sweep_iterations,
    "search_radius": sweep_search_radius,
    "grid_small": sweep_full_grid_small,
    "techniques": sweep_techniques,
    "fix_verify": sweep_fix_verify,
}
