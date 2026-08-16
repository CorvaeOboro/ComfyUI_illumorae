"""Regression test for the source-discipline invariant.

The PatchMatch infill algorithm must NEVER consult original pixel
content that lies inside the target region. We verify this by
constructing two images that are identical in the source region but
radically different in the target region, running the algorithm on both
with the same seed, and asserting the infilled target regions match
bit-exact.

If the algorithm is leaking original target content anywhere - in the
patch distance, in the pyramid downsampling, etc. - the two outputs
will disagree inside the target region and this test will fail.

Usage:
    python -m tests.test_mask_discipline
"""
from __future__ import annotations

import os
import sys
import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PARENT = os.path.dirname(_THIS_DIR)
for p in (_PARENT, _THIS_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

from tests.dataset import build_dataset
from tests.methods import patchmatch_core


def _make_variant(sample, variant_kind: str, rng_seed: int) -> np.ndarray:
    """Return a copy of sample.gt whose TARGET pixels have been replaced
    by ``variant_kind``-chosen content. Source pixels are bit-exact to
    sample.gt."""
    img = sample.gt.copy()
    m = sample.mask > 0.5
    if variant_kind == "zeros":
        img[m] = 0.0
    elif variant_kind == "ones":
        img[m] = 1.0
    elif variant_kind == "noise":
        rng = np.random.default_rng(rng_seed)
        img[m] = rng.random((int(m.sum()), 3)).astype(np.float32)
    elif variant_kind == "gt":
        # Leave ground-truth content in the target region - the
        # tempting cheat.
        pass
    else:
        raise ValueError(variant_kind)
    return img


def _run_one(enforce: bool, verbose: bool) -> bool:
    """Run the full variant cross-check with or without source discipline.
    Returns True if every variant pair produces bit-identical output
    inside the target region."""
    dataset = build_dataset(
        size=56,
        gt_kinds=["gradient", "noise", "checker"],
        mask_kinds=["rect", "ellipse", "blob"],
        coverages=[0.15],
        samples_per_combo=1,
        seed=0,
    )
    variants = ["zeros", "ones", "noise", "gt"]
    params = dict(
        patch_size=7,
        iterations=3,
        search_radius=30,
        blend_width=0,
        multiscale_levels=2,
        fix_backward_direction=True,
        reject_target_offsets_on_init=True,
        seed=12345,
        enforce_source_discipline=enforce,
    )

    all_ok = True
    for sample in dataset:
        outputs = {v: patchmatch_core(
            _make_variant(sample, v, rng_seed=77), sample.mask, **params
        ) for v in variants}

        is_target = sample.mask > 0.5
        ref = outputs["zeros"][is_target]
        for v in variants:
            diff = outputs[v][is_target] - ref
            max_abs = float(np.max(np.abs(diff))) if diff.size else 0.0
            mean_abs = float(np.mean(np.abs(diff))) if diff.size else 0.0
            ok = max_abs <= 1e-6
            mark = "OK " if ok else "LEAK"
            if verbose:
                print(f"  [{mark}] sample={sample.name:<28} "
                      f"variant={v:<6} max|delta|={max_abs:.3e} "
                      f"mean|delta|={mean_abs:.3e}")
            if not ok:
                all_ok = False

        # Source region must be bit-exact to the ground truth.
        is_source = ~is_target
        for v in variants:
            src_diff = outputs[v][is_source] - sample.gt[is_source]
            max_src = (float(np.max(np.abs(src_diff)))
                       if src_diff.size else 0.0)
            if max_src > 1e-6:
                if verbose:
                    print(f"  [FAIL] sample={sample.name} variant={v} "
                          f"SOURCE region altered, max|delta|={max_src:.3e}")
                all_ok = False
    return all_ok


def run(verbose: bool = True) -> bool:
    """Two-sided test:
      1. With source discipline enforced -> MUST pass (no leaks).
      2. With discipline disabled         -> SHOULD fail (proving the
         test actually detects leaks when they exist; otherwise a
         false-positive pass would be possible).
    """
    if verbose:
        print("\n[Phase 1] enforce_source_discipline=True  "
              "(expected: PASS / no leaks)")
    strict_ok = _run_one(enforce=True, verbose=verbose)

    if verbose:
        print("\n[Phase 2] enforce_source_discipline=False "
              "(expected: detector FIRES on the legacy leaky path)")
    leaky_ok = _run_one(enforce=False, verbose=verbose)

    # Strict must pass; leaky MUST fail, otherwise the test is not
    # actually verifying anything.
    return strict_ok and (not leaky_ok)


if __name__ == "__main__":
    print("Source-discipline regression test")
    print("=" * 64)
    ok = run(verbose=True)
    print("=" * 64)
    if ok:
        print("PASS: strict path is clean AND detector fires on leaky path.")
        sys.exit(0)
    else:
        print("FAIL: either strict path leaks, or detector cannot catch "
              "a known leaky configuration.")
        sys.exit(1)
