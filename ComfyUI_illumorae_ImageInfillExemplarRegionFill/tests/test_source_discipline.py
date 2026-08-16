"""Regression test for the source-discipline invariant of the
exemplar region-fill node.

The node must NEVER consult original pixel content from inside the
target region. We verify this by constructing several images that
are identical in the source region but radically different in the
target region, running the algorithm on each, and asserting the
outputs match bit-exact across variants - both inside the target
region (which is reconstructed from source content) and outside
(which must pass through unchanged).

Note on discipline modes:
  The node defaults to ``enforce_source_discipline=False`` (loose
  mode, matching Criminisi 2004), in which previously-filled target
  pixels become valid exemplar source for later SSDs. This does NOT
  break the invariant under test here, because filled pixels are
  always copies of original-source content (the working image's
  target region is zeroed at entry, so there is nothing else for
  them to be). Only original-target content leaking into reads
  would cause cross-variant divergence; that is what this test
  detects.

The test also performs a "test the test" sanity check by
monkey-patching ``_run_single`` to a leaky variant that skips the
target-zeroing step, verifying that the detector then fires.

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

from image_infill_exemplar_regionfill import (  # noqa: E402
    illumoraeImageInfillExemplarRegionFillNode,
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
    patch_size=7,
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
    blend_width=0,
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

        # Sanity: the source region must be passed through bit-exact to
        # ground truth (the algorithm only writes inside the original
        # target region; ``blend_width=0`` so no feather either).
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
    """Replace ``_run_single`` with a variant that does NOT zero the
    target region of the working image at entry. SSDs then read
    original target content directly, so cross-variant outputs must
    diverge.
    """
    orig_run = node._run_single.__func__

    def _leaky_run(self, img, msk, *args, **kwargs):
        # Monkey-patch by re-implementing the entry: we copy the body
        # of _run_single but skip the target-zeroing step. The
        # cleanest way is to delegate to the original after pre-staging
        # the image so the original's zeroing becomes a no-op for
        # original-target content - but the original always re-zeros
        # via `img * (msk < 0.5)`. So we patch by directly bypassing.
        from image_infill_exemplar_regionfill import (
            illumoraeImageInfillExemplarRegionFillNode as _Cls,
        )
        # Run under a temporary override of the multiplication step:
        # we patch numpy's multiplication via a small wrapper. Simpler:
        # call the original on the image itself (no zeroing) by
        # constructing a fake mask that is all-source for the entry
        # zeroing, then immediately restore the real mask. But the
        # downstream code reads `msk` for fill-front; so we need:
        #   - work = img (NOT zeroed) -- requires editing the source.
        # Practical approach: monkey-patch the local zeroing by
        # subclassing in-place. We re-implement here.
        import numpy as _np
        import cv2 as _cv2
        H, W = msk.shape
        patch_size = args[0]
        max_steps = args[1] if len(args) > 1 else 0
        priority_mode = args[2] if len(args) > 2 else "criminisi_2004"
        cheng_omega = args[3] if len(args) > 3 else 0.7
        cheng_alpha = args[4] if len(args) > 4 else 0.2
        cheng_beta = args[5] if len(args) > 5 else 0.8
        use_variance_penalty = args[6] if len(args) > 6 else True
        variance_alpha = args[7] if len(args) > 7 else 0.9
        variance_beta = args[8] if len(args) > 8 else 0.5
        variance_topk = args[9] if len(args) > 9 else 64
        enforce_source_discipline = args[10] if len(args) > 10 else False
        debug_prints = args[11] if len(args) > 11 else False
        half = patch_size // 2

        # *** LEAKY: do NOT zero the target region. Original target
        # content is now visible to every SSD. ***
        work = img.astype(_np.float32).copy()
        target_region = (msk > 0.5).astype(_np.uint8)
        source_region = (1 - target_region).astype(_np.uint8)
        original_source_region = source_region.copy()
        confidence = source_region.astype(_np.float32)
        fill_order = _np.full((H, W), -1.0, dtype=_np.float32)
        if max_steps <= 0:
            max_steps = H * W
        step = 0
        while int(target_region.sum()) > 0 and step < max_steps:
            ff_ys, ff_xs = self._compute_fill_front(target_region)
            if ff_ys.size == 0:
                break
            nx_ff, ny_ff = self._compute_normals(
                source_region, ff_ys, ff_xs,
            )
            gx, gy = self._compute_image_gradients(work, source_region)
            conf_ff = self._compute_confidence(
                confidence, ff_ys, ff_xs, patch_size,
            )
            data_ff = (_np.abs(
                gx[ff_ys, ff_xs] * nx_ff + gy[ff_ys, ff_xs] * ny_ff
            ) + 1e-3).astype(_np.float32)
            if priority_mode == "cheng_blend":
                rcp = (1.0 - cheng_omega) * conf_ff + cheng_omega
                priority = (cheng_alpha * rcp
                            + cheng_beta * data_ff).astype(_np.float32)
            else:
                priority = (conf_ff * data_ff).astype(_np.float32)
            idx = int(_np.argmax(priority))
            ty = int(ff_ys[idx])
            tx = int(ff_xs[idx])
            aY = max(ty - half, 0)
            bY = min(ty + half, H - 1)
            aX = max(tx - half, 0)
            bX = min(tx + half, W - 1)
            pH = bY - aY + 1
            pW = bX - aX + 1
            target_patch = work[aY:bY + 1, aX:bX + 1, :]
            valid_mask_2d = source_region[aY:bY + 1, aX:bX + 1].astype(
                _np.float32
            )
            ssd = self._ssd_search(work, target_patch, valid_mask_2d)
            admissible = (original_source_region
                          if enforce_source_discipline else source_region)
            valid_ul = self._build_admissible_ul_mask(
                admissible, pH, pW,
            )
            valid_ul = self._exclude_self_overlap(
                valid_ul, aY, aX, pH, pW,
            )
            if not valid_ul.any():
                break
            best_uly, best_ulx = self._select_best_ul(
                ssd, valid_ul, work, valid_mask_2d, pH, pW,
                use_variance_penalty, variance_alpha, variance_beta,
                variance_topk,
            )
            if best_uly < 0:
                break
            copy_2d = (source_region[aY:bY + 1, aX:bX + 1] == 0)
            if copy_2d.any():
                src_block = work[best_uly:best_uly + pH,
                                 best_ulx:best_ulx + pW, :]
                tgt_block = work[aY:bY + 1, aX:bX + 1, :]
                tgt_block[copy_2d] = src_block[copy_2d]
                work[aY:bY + 1, aX:bX + 1, :] = tgt_block
                src_reg_block = source_region[aY:bY + 1, aX:bX + 1]
                src_reg_block[copy_2d] = 1
                source_region[aY:bY + 1, aX:bX + 1] = src_reg_block
                tgt_reg_block = target_region[aY:bY + 1, aX:bX + 1]
                tgt_reg_block[copy_2d] = 0
                target_region[aY:bY + 1, aX:bX + 1] = tgt_reg_block
                conf_block = confidence[aY:bY + 1, aX:bX + 1]
                conf_block[copy_2d] = float(conf_ff[idx])
                confidence[aY:bY + 1, aX:bX + 1] = conf_block
                fo_block = fill_order[aY:bY + 1, aX:bX + 1]
                fo_block[copy_2d] = float(step)
                fill_order[aY:bY + 1, aX:bX + 1] = fo_block
            step += 1
        if step > 0:
            denom = max(step - 1, 1)
            viz = _np.where(fill_order >= 0,
                            fill_order / float(denom), 0.0)
        else:
            viz = _np.zeros_like(fill_order)
        viz_rgb = _np.stack([viz.astype(_np.float32)] * 3, axis=-1)
        filled = _np.clip(work, 0.0, 1.0).astype(_np.float32)
        return filled, viz_rgb

    # Bind the leaky function as a method on the instance.
    import types
    node._run_single = types.MethodType(_leaky_run, node)


def run(verbose: bool = True) -> bool:
    node = illumoraeImageInfillExemplarRegionFillNode()

    strict_ok = _run_one(node, verbose,
                         label="Phase 1: strict (expected PASS)")

    node2 = illumoraeImageInfillExemplarRegionFillNode()
    _install_leaky_run(node2)
    leaky_ok = _run_one(node2, verbose,
                        label="Phase 2: leaky (expected LEAK)")

    return strict_ok and (not leaky_ok)


if __name__ == "__main__":
    print("Exemplar Region Fill source-discipline regression test")
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
