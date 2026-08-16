"""Fill-quality regression test for the exemplar region-fill node.

The source-discipline test (``test_source_discipline.py``) verifies
that the output is invariant to the original target-region content,
but a *bit-exact* match across variants is also satisfied if the
algorithm produces an all-zero output for every variant - which is
useless. This test catches such no-op / degenerate regressions by
asserting that the target region is actually filled with plausible
content derived from the source.

Concretely, on a small synthetic colour-gradient image with a central
rectangular hole we assert all of:

  1. The source region is bit-exact unchanged.
  2. The target region is NOT all zero (no all-black regression).
  3. The target region's per-channel mean is within a tolerance band
     of the source region's per-channel mean (since the fill copies
     source patches, channel statistics should roughly transfer).
  4. The target region is fully filled (no remaining "untouched"
     zero pixels in the interior).

We also run a minimal smoke check on a noise image to confirm the
algorithm does not crash on pure-stochastic content.

This test was added after a regression where the discrete-Laplacian
sign convention in ``_compute_fill_front`` was inverted, which made
the fill-front detector always return an empty set and the loop
abort at step 0 with the working image (target zeroed at entry)
returned unchanged - i.e. the entire target region came out black.
The discipline test alone could not catch this because every variant
also produced a black target region, so all variants matched
bit-exactly.

Usage:
    python -m tests.test_fill_quality
"""
from __future__ import annotations

import os
import sys
import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PARENT = os.path.dirname(_THIS_DIR)
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)

from image_infill_exemplar_regionfill import (  # noqa: E402
    illumoraeImageInfillExemplarRegionFillNode,
)


# Same parameters as the discipline test for consistency.
PARAMS = dict(
    patch_size=9,
    max_steps=0,
    priority_mode="criminisi_2004",
    cheng_omega=0.7,
    cheng_alpha=0.2,
    cheng_beta=0.8,
    use_variance_penalty=True,
    variance_alpha=0.9,
    variance_beta=0.5,
    variance_topk=32,
    enforce_source_discipline=False,
    debug_prints=False,
)


def _run(node, img: np.ndarray, msk: np.ndarray) -> np.ndarray:
    filled, _viz = node._run_single(
        img.astype(np.float32),
        msk.astype(np.float32),
        PARAMS["patch_size"],
        PARAMS["max_steps"],
        PARAMS["priority_mode"],
        PARAMS["cheng_omega"],
        PARAMS["cheng_alpha"],
        PARAMS["cheng_beta"],
        PARAMS["use_variance_penalty"],
        PARAMS["variance_alpha"],
        PARAMS["variance_beta"],
        PARAMS["variance_topk"],
        PARAMS["enforce_source_discipline"],
        PARAMS["debug_prints"],
    )
    return filled.astype(np.float32)


def _make_gradient(H: int = 64, W: int = 64) -> np.ndarray:
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    return np.stack([
        xx / (W - 1),
        yy / (H - 1),
        0.5 * np.ones_like(xx),
    ], axis=-1).astype(np.float32)


def _make_noise(H: int = 64, W: int = 64, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.random((H, W, 3)).astype(np.float32)


def _make_central_rect_mask(H: int, W: int, frac: float = 0.4) -> np.ndarray:
    msk = np.zeros((H, W), dtype=np.float32)
    side = int(min(H, W) * frac)
    y0 = (H - side) // 2
    x0 = (W - side) // 2
    msk[y0:y0 + side, x0:x0 + side] = 1.0
    return msk


def _check_filled(
    name: str,
    img: np.ndarray,
    msk: np.ndarray,
    filled: np.ndarray,
    *,
    mean_tol: float = 0.30,
    verbose: bool = True,
) -> bool:
    """Run the four quality checks. Returns True on PASS."""
    target = msk > 0.5
    source = ~target

    # 1. Source pass-through.
    src_delta = float(np.max(np.abs(filled[source] - img[source])))
    ok_source = src_delta <= 1e-6

    # 2. Not all-zero target.
    target_max = float(filled[target].max())
    ok_nonzero = target_max > 1e-3

    # 3. Channel-mean similarity (loose: copying source patches into
    # target should keep the per-channel mean within a tolerance band
    # of the source mean - exact equality is not expected since the
    # patches selected are biased toward boundary content).
    src_mean = img[source].mean(axis=0)
    tgt_mean = filled[target].mean(axis=0)
    mean_delta = float(np.max(np.abs(tgt_mean - src_mean)))
    ok_mean = mean_delta <= mean_tol

    # 4. No remaining unfilled (all-channels-zero) pixels in target.
    n_black = int((filled[target] == 0).all(axis=-1).sum())
    ok_filled = n_black == 0

    ok = ok_source and ok_nonzero and ok_mean and ok_filled
    mark = "OK  " if ok else "FAIL"
    if verbose:
        print(
            f"  [{mark}] {name:<24} "
            f"src_delta={src_delta:.3e} "
            f"target_max={target_max:.3f} "
            f"|tgt_mean - src_mean|={mean_delta:.3f} "
            f"black_px={n_black}/{int(target.sum())}"
        )
    return ok


def run(verbose: bool = True) -> bool:
    node = illumoraeImageInfillExemplarRegionFillNode()
    if verbose:
        print("\n[Phase 1: fill-quality on synthetic inputs]")

    all_ok = True

    # Case 1: Smooth gradient with a central rectangular hole.
    img1 = _make_gradient(64, 64)
    msk1 = _make_central_rect_mask(64, 64, frac=0.4)
    filled1 = _run(node, img1, msk1)
    all_ok &= _check_filled(
        "gradient + rect", img1, msk1, filled1,
        mean_tol=0.30, verbose=verbose,
    )

    # Case 2: Random colour noise with a central rectangular hole.
    img2 = _make_noise(64, 64, seed=42)
    msk2 = _make_central_rect_mask(64, 64, frac=0.4)
    filled2 = _run(node, img2, msk2)
    all_ok &= _check_filled(
        "noise + rect", img2, msk2, filled2,
        # Noise has high per-channel variance; the channel mean from a
        # subset of patches can drift more than for a smooth gradient.
        mean_tol=0.20, verbose=verbose,
    )

    # Case 3: Larger image, smaller hole - smoke check that the
    # algorithm scales and terminates cleanly.
    img3 = _make_gradient(96, 96)
    msk3 = _make_central_rect_mask(96, 96, frac=0.25)
    filled3 = _run(node, img3, msk3)
    all_ok &= _check_filled(
        "gradient 96 + small rect", img3, msk3, filled3,
        mean_tol=0.30, verbose=verbose,
    )

    # Case 4: Strict-discipline mode also produces non-black output.
    node_strict = illumoraeImageInfillExemplarRegionFillNode()
    PARAMS_STRICT = dict(PARAMS)
    PARAMS_STRICT["enforce_source_discipline"] = True
    filled4, _ = node_strict._run_single(
        img1.astype(np.float32), msk1.astype(np.float32),
        PARAMS_STRICT["patch_size"], PARAMS_STRICT["max_steps"],
        PARAMS_STRICT["priority_mode"],
        PARAMS_STRICT["cheng_omega"],
        PARAMS_STRICT["cheng_alpha"],
        PARAMS_STRICT["cheng_beta"],
        PARAMS_STRICT["use_variance_penalty"],
        PARAMS_STRICT["variance_alpha"],
        PARAMS_STRICT["variance_beta"],
        PARAMS_STRICT["variance_topk"],
        PARAMS_STRICT["enforce_source_discipline"],
        PARAMS_STRICT["debug_prints"],
    )
    all_ok &= _check_filled(
        "gradient + rect (strict)", img1, msk1,
        filled4.astype(np.float32),
        mean_tol=0.30, verbose=verbose,
    )

    return all_ok


if __name__ == "__main__":
    print("Exemplar Region Fill fill-quality regression test")
    print("=" * 70)
    ok = run(verbose=True)
    print("=" * 70)
    if ok:
        print("PASS: target region is filled with plausible content "
              "(no all-black / no-op regression).")
        sys.exit(0)
    else:
        print("FAIL: at least one case produced an all-black or "
              "statistically wrong fill - see line-by-line output.")
        sys.exit(1)
