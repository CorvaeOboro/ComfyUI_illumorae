"""Regression test for the source-discipline invariant of the boundary
propagate node.

The Image Infill Boundary Propagate node must NEVER consult original
pixel content that lies inside the target region. We verify this by
constructing several images that are identical in the source region but
radically different in the target region, running the algorithm on each,
and asserting the infilled target regions match bit-exact.

If the algorithm is leaking original target content anywhere - e.g. via
a soft-mask final blend - the outputs will disagree inside the target
region and this test will fail.

The test also performs a "test the test" sanity check by manually
monkey-patching ``propagate_boundary_rgb`` back to the legacy soft-mask
blend and verifying that the detector then fires, proving the test is
actually sensitive to real leaks.

Usage:
    python -m tests.test_source_discipline
"""
from __future__ import annotations

import os
import sys
import numpy as np
import cv2

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PARENT = os.path.dirname(_THIS_DIR)
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)

from image_infill_boundary_propagate import (  # noqa: E402
    illumoraeImageInfillBoundaryPropagateNode,
)


# --------------------------------------------------------------------------
# Tiny dataset
# --------------------------------------------------------------------------

def _make_gt(kind: str, size: int, seed: int) -> np.ndarray:
    """Deterministic ground-truth image in float32 [0, 1]."""
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
    """Binary mask (float32, 0 or 1) where 1 = target region, 0 = source."""
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
        # Union of several random disks summing to approximately ``area``.
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


def _build_dataset(size: int = 64, coverage: float = 0.18, seed: int = 0):
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
        # The tempting cheat: keep original content in the target region.
        pass
    else:
        raise ValueError(variant_kind)
    return img


# --------------------------------------------------------------------------
# Core check
# --------------------------------------------------------------------------

PARAMS = dict(
    mask_mode="white=fill",
    propagate_iterations=60,
    blur_center=True,
    blur_strength=5.0,
    blur_falloff_distance=40,
    feather_amount=8,
    debug_prints=False,
)

VARIANTS = ("zeros", "ones", "noise", "gt")


def _run_one(node, verbose: bool, label: str) -> bool:
    dataset = _build_dataset(size=64, coverage=0.18, seed=0)
    if verbose:
        print(f"\n[{label}] running {len(dataset)} samples x "
              f"{len(VARIANTS)} variants")

    all_ok = True
    for name, gt, msk in dataset:
        outputs = {
            v: node.process_single_numpy(
                _make_variant(gt, msk, v, rng_seed=77).copy(),
                msk.copy(),
                **PARAMS,
            )[0]  # [0] -> filled image (drop viz outputs)
            for v in VARIANTS
        }
        is_target = msk > 0.5
        is_source = ~is_target
        # Cross-variant checks: since only target-region content differs
        # across variants, every pixel of every variant's output - both
        # inside and outside the target region - must match the
        # reference variant bit-exact. Any cross-variant divergence is
        # direct evidence that the algorithm observed the original
        # target content somewhere.
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

        # Additional sanity: source-region output matches ground truth
        # up to the uint8 quantization used inside apply_center_blur
        # (~1/255). This is a roundtrip artifact of the algorithm, not
        # a source-discipline issue - but we note it here for
        # transparency.
        src_vs_gt = float(np.max(np.abs(
            outputs["zeros"][is_source] - gt[is_source]
        )))
        if verbose and src_vs_gt > 1e-6:
            print(f"         (source vs gt: max|delta|={src_vs_gt:.3e} "
                  f"<- uint8 quantization in apply_center_blur)")
    return all_ok


# --------------------------------------------------------------------------
# "Test the test": reproduce the legacy leaky behavior and verify detection
# --------------------------------------------------------------------------

def _install_legacy_leaky_blend(node):
    """Monkey-patch ``propagate_boundary_rgb`` on this node instance to
    use the old soft-mask blend that leaked original target content, so
    we can confirm the detector actually catches a known-leaky
    configuration."""
    def _leaky_propagate_boundary_rgb(image_np, mask_np, iterations):
        num_channels = image_np.shape[2] if image_np.ndim == 3 else 1
        image_uint8 = (image_np * 255).astype(np.uint8)
        mask_binary = (mask_np > 0.5).astype(np.uint8) * 255
        fill_mask = mask_binary.copy()
        original_image = image_uint8.copy()
        result = image_uint8.copy()
        mask_3ch = np.stack([mask_binary] * num_channels, axis=-1)
        result = np.where(mask_3ch > 0, 0, result)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        for _ in range(iterations):
            dilated = cv2.dilate(result, kernel, iterations=1)
            fill_mask_eroded = cv2.erode(fill_mask, kernel, iterations=1)
            update_mask = fill_mask - fill_mask_eroded
            update_mask_nch = np.stack([update_mask] * num_channels, axis=-1)
            result = np.where(update_mask_nch > 0, dilated, result)
            fill_mask = fill_mask_eroded
            if np.sum(fill_mask) == 0:
                break
        # LEGACY soft-mask blend (the bug).
        mask_blend = np.stack([mask_np] * num_channels, axis=-1)
        final_result = (result.astype(np.float32) * mask_blend +
                        original_image.astype(np.float32) * (1.0 - mask_blend))
        return final_result.astype(np.float32) / 255.0

    # Also remove the belt-and-suspenders target-zeroing so the leak is
    # maximally exposed (we want the detector to fire unambiguously).
    def _leaky_process_single_numpy(img, msk, mask_mode,
                                    propagate_iterations, blur_center,
                                    blur_strength, blur_falloff_distance,
                                    feather_amount, debug_prints=False):
        msk = node._resolve_mask(msk, mask_mode, debug_prints=False)
        # NOTE: intentionally NOT zeroing the target here.
        if feather_amount > 0:
            msk_feathered = node.feather_mask(msk, feather_amount)
        else:
            msk_feathered = msk
        mask_binary_viz = ((msk_feathered > 0.5).astype(np.uint8)) * 255
        dist_map = cv2.distanceTransform(mask_binary_viz, cv2.DIST_L2, 5)
        dist_map_norm = (dist_map / dist_map.max()
                         if dist_map.max() > 0 else dist_map)
        blur_weight_map = np.clip(dist_map / max(blur_falloff_distance, 1),
                                  0, 1)
        dist_map_rgb = np.stack([dist_map_norm] * 3,
                                axis=-1).astype(np.float32)
        blur_weight_rgb = np.stack([blur_weight_map] * 3,
                                   axis=-1).astype(np.float32)
        infilled = _leaky_propagate_boundary_rgb(
            img, msk_feathered, propagate_iterations,
        )
        if blur_center and blur_strength > 0:
            infilled = node.apply_center_blur(
                infilled, msk_feathered, blur_strength, blur_falloff_distance
            )
        return (infilled.astype(np.float32), dist_map_rgb, blur_weight_rgb)

    node.process_single_numpy = _leaky_process_single_numpy


def run(verbose: bool = True) -> bool:
    node = illumoraeImageInfillBoundaryPropagateNode()

    # Phase 1 -- strict / fixed path: MUST pass.
    strict_ok = _run_one(node, verbose,
                         label="Phase 1: strict (expected PASS)")

    # Phase 2 -- legacy leaky path: detector MUST fire (i.e. NOT pass).
    node2 = illumoraeImageInfillBoundaryPropagateNode()
    _install_legacy_leaky_blend(node2)
    leaky_ok = _run_one(node2, verbose,
                        label="Phase 2: legacy leaky (expected LEAK)")

    return strict_ok and (not leaky_ok)


if __name__ == "__main__":
    print("Boundary Propagate node source-discipline regression test")
    print("=" * 70)
    ok = run(verbose=True)
    print("=" * 70)
    if ok:
        print("PASS: strict path is clean AND detector fires on legacy path.")
        sys.exit(0)
    else:
        print("FAIL: either strict path leaks or detector failed to catch "
              "a known leaky configuration.")
        sys.exit(1)
