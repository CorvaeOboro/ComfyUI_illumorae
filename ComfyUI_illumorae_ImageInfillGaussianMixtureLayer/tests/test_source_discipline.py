"""Regression test for the source-discipline invariant of the
Gaussian Mixture Layer (Galerne-Leclaire) infill node.

The node must NEVER consult original pixel content from inside the
target region. We verify this by constructing several images that
are identical in the source region but radically different in the
target region, running the algorithm on each, and asserting the
outputs match bit-exact - both inside the target region (which is
reconstructed from source-derived quantities only) and outside
(which must pass through unchanged).

For the single-Gaussian algorithm the invariant holds by
construction: every quantity (mu, texton, unconditional sample F,
source residual b, CG solution z, correction c) is a function of
``u_S`` only. The final re-stamp ``X_S <- u_S`` additionally
guarantees bit-exact pass-through of source pixels.

The "test the test" phase monkey-patches the node to a leaky
variant that estimates the Gaussian model from the ENTIRE image
(ignoring the mask) and verifies that the detector then fires.

Usage:
    python -m tests.test_source_discipline
"""
from __future__ import annotations

import os
import sys
import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PARENT = os.path.dirname(_THIS_DIR)
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)

from image_infill_gaussian_mixture_layer import (  # noqa: E402
    illumoraeImageInfillGaussianMixtureLayerNode,
)


# --------------------------------------------------------------------------
# Tiny dataset
# --------------------------------------------------------------------------

def _make_gt(kind: str, size: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    if kind == "gradient":
        yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
        img = np.stack([
            xx / (size - 1),
            yy / (size - 1),
            (xx + yy) / (2 * (size - 1)),
        ], axis=-1)
    elif kind == "noise":
        img = rng.random((size, size, 3)).astype(np.float32)
    elif kind == "checker":
        yy, xx = np.mgrid[0:size, 0:size]
        cell = 8
        c = ((xx // cell + yy // cell) % 2).astype(np.float32)
        img = np.stack([c, 1.0 - c, c * 0.5], axis=-1)
    else:
        raise ValueError(kind)
    return img.astype(np.float32)


def _make_mask(kind: str, size: int, coverage: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    m = np.zeros((size, size), dtype=np.float32)
    area = int(size * size * coverage)
    if kind == "rect":
        side = int(np.sqrt(area))
        y0 = (size - side) // 2
        x0 = (size - side) // 2
        m[y0:y0 + side, x0:x0 + side] = 1.0
    elif kind == "ellipse":
        cy, cx = size // 2, size // 2
        ry = int(np.sqrt(area / np.pi))
        rx = ry
        yy, xx = np.mgrid[0:size, 0:size]
        m[((yy - cy) ** 2 / (ry ** 2) + (xx - cx) ** 2 / (rx ** 2)) <= 1] = 1.0
    elif kind == "blob":
        remaining = area
        while remaining > 0:
            r = int(rng.integers(size // 12, size // 6))
            cy = int(rng.integers(r, size - r))
            cx = int(rng.integers(r, size - r))
            yy, xx = np.mgrid[0:size, 0:size]
            disk = ((yy - cy) ** 2 + (xx - cx) ** 2) <= r * r
            m[disk] = 1.0
            remaining -= int(disk.sum())
    else:
        raise ValueError(kind)
    return m


def _build_dataset(size: int = 48, coverage: float = 0.12, seed: int = 0):
    samples = []
    for gt_kind in ("gradient", "noise", "checker"):
        for mask_kind in ("rect", "ellipse", "blob"):
            gt = _make_gt(gt_kind, size, seed=seed)
            msk = _make_mask(mask_kind, size, coverage, seed=seed + 1)
            if msk.sum() == 0:
                continue
            samples.append((f"{gt_kind}_{mask_kind}", gt, msk))
    return samples


def _make_variant(gt: np.ndarray, mask: np.ndarray,
                  variant_kind: str, rng_seed: int) -> np.ndarray:
    img = gt.copy()
    is_target = mask > 0.5
    if variant_kind == "zeros":
        img[is_target] = 0.0
    elif variant_kind == "ones":
        img[is_target] = 1.0
    elif variant_kind == "noise":
        rng = np.random.default_rng(rng_seed)
        img[is_target] = rng.random(
            (int(is_target.sum()), 3)
        ).astype(np.float32)
    elif variant_kind == "gt":
        pass
    else:
        raise ValueError(variant_kind)
    return img


# --------------------------------------------------------------------------
# Core check
# --------------------------------------------------------------------------

PARAMS = dict(
    cg_max_iterations=40,
    cg_tolerance=1e-6,
    add_innovation=True,
    innovation_strength=1.0,
    regularization=1e-3,
    seed=0,
    debug_prints=False,
)

VARIANTS = ("zeros", "ones", "noise", "gt")


def _run_node(node, img_np: np.ndarray, msk_np: np.ndarray) -> np.ndarray:
    """Call ``_run_single`` directly so the test does not depend on
    torch (the torch wrapper is just batching, which we don't need)."""
    filled, _viz = node._run_single(
        img_np.astype(np.float32),
        msk_np.astype(np.float32),
        PARAMS["cg_max_iterations"],
        PARAMS["cg_tolerance"],
        PARAMS["add_innovation"],
        PARAMS["innovation_strength"],
        PARAMS["regularization"],
        PARAMS["seed"],
        PARAMS["debug_prints"],
    )
    return filled.astype(np.float32)


def _run_one(node, verbose: bool, label: str) -> bool:
    dataset = _build_dataset(size=48, coverage=0.12, seed=0)
    if verbose:
        print(f"\n[{label}] running {len(dataset)} samples x "
              f"{len(VARIANTS)} variants")

    all_ok = True
    for name, gt, msk in dataset:
        outputs = {
            v: _run_node(node, _make_variant(gt, msk, v, rng_seed=77), msk)
            for v in VARIANTS
        }
        is_target = msk > 0.5
        is_source = ~is_target

        ref_target = outputs["zeros"][is_target]
        ref_source = outputs["zeros"][is_source]
        for v in VARIANTS:
            d_target = outputs[v][is_target] - ref_target
            max_t = float(np.max(np.abs(d_target))) if d_target.size else 0.0
            mean_t = float(np.mean(np.abs(d_target))) if d_target.size else 0.0
            ok_target = max_t <= 1e-6

            d_source = outputs[v][is_source] - ref_source
            max_s = float(np.max(np.abs(d_source))) if d_source.size else 0.0
            ok_source = max_s <= 1e-6

            mark = "OK  " if (ok_target and ok_source) else "LEAK"
            if verbose:
                print(f"  [{mark}] sample={name:<20} variant={v:<6} "
                      f"target max|delta|={max_t:.3e} "
                      f"mean|delta|={mean_t:.3e}"
                      f"   source max|delta|={max_s:.3e}")
            if not (ok_target and ok_source):
                all_ok = False

        # Sanity: source pass-through is bit-exact (final re-stamp).
        src_vs_gt = float(np.max(np.abs(
            outputs["zeros"][is_source] - gt[is_source]
        )))
        if verbose and src_vs_gt > 1e-6:
            print(f"         (source vs gt: max|delta|={src_vs_gt:.3e})")
            all_ok = False
    return all_ok


# --------------------------------------------------------------------------
# "Test the test": leaky variant must be detected
# --------------------------------------------------------------------------

def _install_leaky_run(node):
    """Replace ``_run_single`` with a variant that estimates
    the Gaussian model (mu, texton) from the ENTIRE image, ignoring
    the source mask. This directly reads u_T (original target pixels)
    so cross-variant outputs must diverge.
    """
    import numpy as _np
    import types

    def _leaky(self, img, msk, cg_max_iterations, cg_tolerance,
               add_innovation, innovation_strength, regularization,
               seed, debug_prints,
               detrend_sigma=0.0, clamp_to_source_gamut=True):
        # ``detrend_sigma`` / ``clamp_to_source_gamut`` are accepted
        # to match the production ``_run_single`` signature
        # but ignored - this stub is intentionally minimalist; it
        # just leaks ``u_T`` into ``mu`` and the texton via the
        # all-ones mask below, which is enough for the leak detector.
        H, W, C = img.shape
        source_mask = (msk < 0.5).astype(_np.float32)
        src_bool = source_mask > 0.5

        # *** LEAKY: use the ENTIRE image to build the Gaussian model
        # (ignores the mask). Original target content is now baked
        # into mu and into the texton spectrum. ***
        all_mask = _np.ones_like(source_mask, dtype=_np.float32)
        mu, t = self._build_texton(img, all_mask)
        T = _np.fft.rfft2(t, axes=(0, 1))
        psd = (T * _np.conj(T)).real.astype(_np.float32)
        # Tikhonov ridge - same scaling rule as production.
        texton_energy_pc = (t * t).sum(axis=(0, 1)).astype(_np.float32)
        ridge = float(regularization) * _np.maximum(
            texton_energy_pc, 1e-12)

        rng = _np.random.default_rng(int(seed) & 0xFFFFFFFF)
        if add_innovation:
            white = rng.standard_normal((H, W)).astype(_np.float32)
            white *= float(innovation_strength)
            W_spec = _np.fft.rfft2(white)
            F = _np.zeros_like(img, dtype=_np.float32)
            for c in range(C):
                conv_c = _np.fft.irfft2(
                    T[..., c] * W_spec, s=(H, W),
                ).astype(_np.float32)
                F[..., c] = mu[c] + conv_c
        else:
            F = _np.broadcast_to(
                mu.reshape(1, 1, C), img.shape
            ).astype(_np.float32).copy()

        u_src = img[src_bool].astype(_np.float32)
        F_src = F[src_bool].astype(_np.float32)
        b = u_src - F_src
        z, _it, _rel = self._conjugate_gradient(
            b, source_mask, psd, ridge,
            max_iter=int(cg_max_iterations),
            tol=float(cg_tolerance),
            debug=False,
        )
        z_full = _np.zeros_like(img, dtype=_np.float32)
        sy, sx = _np.where(src_bool)
        z_full[sy, sx, :] = z
        correction = _np.zeros_like(img, dtype=_np.float32)
        for c in range(C):
            correction[..., c] = self._fft_convolve(
                psd[..., c], z_full[..., c],
            ).astype(_np.float32)
        X = F + correction
        X[src_bool] = img[src_bool]
        return (_np.clip(X, 0.0, 1.0).astype(_np.float32),
                _np.zeros_like(img, dtype=_np.float32))

    node._run_single = types.MethodType(_leaky, node)


def run(verbose: bool = True) -> bool:
    node = illumoraeImageInfillGaussianMixtureLayerNode()

    strict_ok = _run_one(node, verbose,
                         label="Phase 1: strict (expected PASS)")

    node2 = illumoraeImageInfillGaussianMixtureLayerNode()
    _install_leaky_run(node2)
    leaky_ok = _run_one(node2, verbose,
                        label="Phase 2: leaky (expected LEAK)")

    return strict_ok and (not leaky_ok)


if __name__ == "__main__":
    print("Gaussian Mixture Layer source-discipline regression test")
    print("=" * 70)
    ok = run(verbose=True)
    print("=" * 70)
    if ok:
        print("PASS: strict path is clean AND detector fires on leaky path.")
        sys.exit(0)
    else:
        print("FAIL: either strict path leaks or detector failed to catch "
              "a known leaky configuration.")
        sys.exit(1)
