"""Method registry: stock PatchMatch node, technique-wedge variants, and baselines.

Each method is a callable:
    fn(image_f32_HWC, mask_f32_HW, **params) -> result_f32_HWC

The `patchmatch_core` function re-implements the same algorithm as the
node but exposes toggles so we can A/B test individual techniques
(propagation direction, random search, multi-scale, blending, etc.)
without modifying the production node.
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Callable, Dict, Optional

import cv2
import numpy as np

# Make the node importable without ComfyUI on path
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PARENT = os.path.dirname(_THIS_DIR)
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)

# Torch is required by the node. Guard the import so baselines still run
# if torch is not installed.
try:
    import torch  # type: ignore
    _HAS_TORCH = True
except Exception:
    _HAS_TORCH = False

if _HAS_TORCH:
    from image_infill_patchmatch import illumoraeImageInfillPatchMatchNode


# ---------------------------------------------------------------------------
# Stock node wrapper
# ---------------------------------------------------------------------------

def run_stock_node(
    image: np.ndarray,
    mask: np.ndarray,
    patch_size: int = 7,
    iterations: int = 5,
    search_radius: int = 50,
    blend_width: int = 5,
    multiscale_levels: int = 2,
    seed: int = 0,
    **_ignored,
) -> np.ndarray:
    """Call the production ComfyUI node (without ComfyUI)."""
    if not _HAS_TORCH:
        raise RuntimeError("torch is required for the stock node method")
    node = illumoraeImageInfillPatchMatchNode()
    img_t = torch.from_numpy(image).unsqueeze(0).float()
    msk_t = torch.from_numpy(mask).unsqueeze(0).float()
    out, _ = node.patchmatch_infill(
        img_t, msk_t,
        patch_size=int(patch_size),
        iterations=int(iterations),
        search_radius=int(search_radius),
        blend_width=int(blend_width),
        multiscale_levels=int(multiscale_levels),
        seed=int(seed),
        debug_prints=False,
    )
    return out[0].cpu().numpy().astype(np.float32)


# ---------------------------------------------------------------------------
# Standalone PatchMatch core (used for technique wedges)
# ---------------------------------------------------------------------------

def _patch_distance(
    img: np.ndarray, mask: np.ndarray,
    y1: int, x1: int, y2: int, x2: int, half: int,
) -> float:
    h, w, _ = img.shape
    # clamp patch to bounds with same offsets relative to both centers
    dy_lo = -min(half, y1, y2)
    dy_hi = min(half, h - 1 - y1, h - 1 - y2)
    dx_lo = -min(half, x1, x2)
    dx_hi = min(half, w - 1 - x1, w - 1 - x2)
    if dy_hi < dy_lo or dx_hi < dx_lo:
        return float("inf")
    p1 = img[y1 + dy_lo:y1 + dy_hi + 1, x1 + dx_lo:x1 + dx_hi + 1]
    p2 = img[y2 + dy_lo:y2 + dy_hi + 1, x2 + dx_lo:x2 + dx_hi + 1]
    m1 = mask[y1 + dy_lo:y1 + dy_hi + 1, x1 + dx_lo:x1 + dx_hi + 1]
    m2 = mask[y2 + dy_lo:y2 + dy_hi + 1, x2 + dx_lo:x2 + dx_hi + 1]
    # Source-discipline: only positions in the source region in BOTH
    # patches contribute. The query patch (p1) may overlap the target
    # region; including those positions would leak original target
    # content into the score.
    valid = (m1 < 0.5) & (m2 < 0.5)
    if not valid.any():
        return float("inf")
    diff = (p1 - p2) ** 2
    s = diff.sum(axis=-1) * valid
    return float(s.sum() / (valid.sum() + 1e-8))


def _init_offsets(
    mask: np.ndarray, search_radius: int,
    reject_target_offsets: bool, rng: np.random.Generator,
) -> np.ndarray:
    h, w = mask.shape
    offs = np.zeros((h, w, 2), dtype=np.int32)
    is_target = mask > 0.5
    ys, xs = np.where(is_target)
    for y, x in zip(ys, xs):
        for _try in range(10):
            dy = int(rng.integers(-search_radius, search_radius + 1))
            dx = int(rng.integers(-search_radius, search_radius + 1))
            sy = int(np.clip(y + dy, 0, h - 1))
            sx = int(np.clip(x + dx, 0, w - 1))
            if (not reject_target_offsets) or mask[sy, sx] < 0.5:
                offs[y, x] = (sy - y, sx - x)
                break
        else:
            offs[y, x] = (sy - y, sx - x)
    return offs


def _propagate(
    img: np.ndarray, mask: np.ndarray, offs: np.ndarray,
    half: int, reverse: bool, fix_direction: bool,
) -> np.ndarray:
    h, w = mask.shape
    is_target = mask > 0.5
    out = offs.copy()
    if reverse:
        y_iter = range(h - 1, -1, -1)
        x_iter = range(w - 1, -1, -1)
        # fixed bwd pass: check right / bottom neighbors
        neigh = [(0, 1), (1, 0)] if fix_direction else [(0, -1), (-1, 0)]
    else:
        y_iter = range(h)
        x_iter = range(w)
        neigh = [(0, -1), (-1, 0)]  # left / top

    for y in y_iter:
        for x in x_iter:
            if not is_target[y, x]:
                continue
            cur = out[y, x]
            sy, sx = y + cur[0], x + cur[1]
            best_d = _patch_distance(img, mask, y, x, sy, sx, half)
            best = cur
            for dy, dx in neigh:
                ny, nx = y + dy, x + dx
                if 0 <= ny < h and 0 <= nx < w and is_target[ny, nx]:
                    no = out[ny, nx]
                    sy2, sx2 = y + no[0], x + no[1]
                    if 0 <= sy2 < h and 0 <= sx2 < w:
                        d = _patch_distance(img, mask, y, x, sy2, sx2, half)
                        if d < best_d:
                            best_d = d
                            best = no
            out[y, x] = best
    return out


def _random_search(
    img: np.ndarray, mask: np.ndarray, offs: np.ndarray,
    half: int, search_radius: int, rng: np.random.Generator,
) -> np.ndarray:
    h, w = mask.shape
    is_target = mask > 0.5
    out = offs.copy()
    ys, xs = np.where(is_target)
    for y, x in zip(ys, xs):
        cur = out[y, x]
        sy, sx = y + cur[0], x + cur[1]
        best_d = _patch_distance(img, mask, y, x, sy, sx, half)
        best = cur
        radius = search_radius
        while radius >= 1:
            dy = int(rng.integers(-radius, radius + 1))
            dx = int(rng.integers(-radius, radius + 1))
            sy2 = int(np.clip(y + cur[0] + dy, 0, h - 1))
            sx2 = int(np.clip(x + cur[1] + dx, 0, w - 1))
            if mask[sy2, sx2] < 0.5:
                d = _patch_distance(img, mask, y, x, sy2, sx2, half)
                if d < best_d:
                    best_d = d
                    best = np.array([sy2 - y, sx2 - x], dtype=np.int32)
            radius //= 2
        out[y, x] = best
    return out


def _reconstruct(
    img: np.ndarray, mask: np.ndarray, offs: np.ndarray,
    blend_width: int, do_blend: bool,
) -> np.ndarray:
    h, w, _ = img.shape
    out = img.copy()
    is_target = mask > 0.5
    mask_u8 = (mask * 255).astype(np.uint8)
    # distance from each target pixel to the target boundary (inside
    # the target region)
    dist = cv2.distanceTransform(mask_u8, cv2.DIST_L2, 5)

    # Base for blending near the boundary: nearest source pixel.
    blend_base = None
    if do_blend and blend_width > 0:
        is_source_u8 = 1 - (mask_u8 > 0).astype(np.uint8)
        _, labels = cv2.distanceTransformWithLabels(
            is_source_u8, cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_PIXEL,
        )
        src_ys, src_xs = np.where(is_source_u8 == 1)
        lut = np.zeros((labels.max() + 1, 2), dtype=np.int32)
        lut[labels[src_ys, src_xs], 0] = src_ys
        lut[labels[src_ys, src_xs], 1] = src_xs
        blend_base = img[lut[labels, 0], lut[labels, 1]]

    ys, xs = np.where(is_target)
    for y, x in zip(ys, xs):
        o = offs[y, x]
        sy = int(np.clip(y + o[0], 0, h - 1))
        sx = int(np.clip(x + o[1], 0, w - 1))
        src = img[sy, sx]
        if blend_base is not None and dist[y, x] < blend_width:
            w_src = float(dist[y, x]) / float(blend_width)
            out[y, x] = w_src * src + (1.0 - w_src) * blend_base[y, x]
        else:
            out[y, x] = src
    return out


def _source_aware_pyrdown(
    img: np.ndarray, mask: np.ndarray,
) -> "tuple[np.ndarray, np.ndarray]":
    """Source-aware pyramid downsample. Mirrors the production node
    helper: coarse image pixels are weighted Gaussian averages over
    source pixels only, so no target content contaminates coarse
    source pixels."""
    src_weight = (mask < 0.5).astype(np.float32)
    img_w = img * src_weight[..., None]
    img_blur = cv2.pyrDown(img_w)
    weight_blur = cv2.pyrDown(src_weight)
    denom = np.maximum(weight_blur, 1e-6)[..., None]
    small_img = (img_blur / denom).astype(np.float32)
    small_mask = (weight_blur < 0.5).astype(np.float32)
    return small_img, small_mask


def patchmatch_core(
    image: np.ndarray,
    mask: np.ndarray,
    patch_size: int = 7,
    iterations: int = 5,
    search_radius: int = 50,
    blend_width: int = 0,
    *,
    use_propagation: bool = True,
    use_random_search: bool = True,
    use_bidirectional: bool = True,
    fix_backward_direction: bool = True,
    reject_target_offsets_on_init: bool = True,
    multiscale_levels: int = 1,
    do_blend: bool = False,
    seed: int = 0,
    enforce_source_discipline: bool = True,
    **_ignored,
) -> np.ndarray:
    """Configurable PatchMatch. Setting toggles off replicates specific
    bugs or disables a stage for A/B testing.

    When ``enforce_source_discipline`` is True (default) the working
    image has its target region zeroed at entry and pyramid downsampling
    is source-aware, so no original target content ever influences the
    result. Setting it to False recovers the legacy (leaky) behavior
    for comparison.
    """
    if patch_size % 2 == 0:
        patch_size += 1
    half = patch_size // 2

    if enforce_source_discipline:
        image = image * (mask < 0.5).astype(np.float32)[..., None]

    def _run_level(img_l: np.ndarray, msk_l: np.ndarray,
                   init_offs: Optional[np.ndarray]) -> np.ndarray:
        rng = np.random.default_rng(seed)
        offs = init_offs
        if offs is None:
            offs = _init_offsets(msk_l, search_radius,
                                 reject_target_offsets_on_init, rng)
        for _ in range(iterations):
            if use_propagation:
                offs = _propagate(img_l, msk_l, offs, half,
                                  reverse=False, fix_direction=False)
                if use_bidirectional:
                    offs = _propagate(img_l, msk_l, offs, half,
                                      reverse=True,
                                      fix_direction=fix_backward_direction)
            if use_random_search:
                offs = _random_search(img_l, msk_l, offs, half,
                                      search_radius, rng)
        return offs

    if multiscale_levels <= 1:
        offs = _run_level(image, mask, None)
        return _reconstruct(image, mask, offs, blend_width, do_blend)

    # Coarse-to-fine pyramid (with degeneration guard).
    pyr_imgs = [image]
    pyr_msks = [mask]
    for _ in range(multiscale_levels - 1):
        if enforce_source_discipline:
            small_img, small_msk = _source_aware_pyrdown(pyr_imgs[-1], pyr_msks[-1])
        else:
            small_img = cv2.pyrDown(pyr_imgs[-1])
            small_msk = cv2.pyrDown(pyr_msks[-1])
            small_msk = (small_msk > 0.25).astype(np.float32)
        if (min(small_img.shape[:2]) < patch_size * 2
                or small_msk.sum() == 0
                or small_msk.sum() == small_msk.size):
            break
        pyr_imgs.append(small_img)
        pyr_msks.append(small_msk)
    offs: Optional[np.ndarray] = None
    for lvl in range(len(pyr_imgs) - 1, -1, -1):
        if offs is not None:
            # upscale offsets to this level's resolution
            th, tw = pyr_msks[lvl].shape
            offs = cv2.resize(offs.astype(np.float32), (tw, th),
                              interpolation=cv2.INTER_NEAREST)
            offs = (offs * 2).astype(np.int32)
        offs = _run_level(pyr_imgs[lvl], pyr_msks[lvl], offs)
    return _reconstruct(image, mask, offs, blend_width, do_blend)


# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------

def baseline_cv2_telea(
    image: np.ndarray, mask: np.ndarray,
    inpaint_radius: int = 3, **_ignored,
) -> np.ndarray:
    bgr = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    m = (mask > 0.5).astype(np.uint8) * 255
    out = cv2.inpaint(bgr, m, inpaint_radius, cv2.INPAINT_TELEA)
    return cv2.cvtColor(out, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0


def baseline_cv2_ns(
    image: np.ndarray, mask: np.ndarray,
    inpaint_radius: int = 3, **_ignored,
) -> np.ndarray:
    bgr = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    m = (mask > 0.5).astype(np.uint8) * 255
    out = cv2.inpaint(bgr, m, inpaint_radius, cv2.INPAINT_NS)
    return cv2.cvtColor(out, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0


def baseline_mean_color(
    image: np.ndarray, mask: np.ndarray, **_ignored,
) -> np.ndarray:
    m = mask > 0.5
    if not (~m).any():
        return image.copy()
    mean = image[~m].mean(axis=0)
    out = image.copy()
    out[m] = mean
    return out


def baseline_nearest_valid(
    image: np.ndarray, mask: np.ndarray, **_ignored,
) -> np.ndarray:
    """Copy from the nearest source pixel (distance transform)."""
    m_u8 = (mask > 0.5).astype(np.uint8)
    # For source pixels, distance=0; we want, for each target pixel,
    # the coords of the nearest source pixel.
    _, labels = cv2.distanceTransformWithLabels(
        1 - m_u8, cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_PIXEL,
    )
    h, w = mask.shape
    # Build label -> coords table from source pixels
    src_ys, src_xs = np.where(m_u8 == 0)
    # OpenCV assigns labels to zero-pixels of input starting from 1
    # We need the coord for each label.
    label_to_coord = np.zeros((labels.max() + 1, 2), dtype=np.int32)
    lbl_of_pixel = labels[src_ys, src_xs]
    label_to_coord[lbl_of_pixel, 0] = src_ys
    label_to_coord[lbl_of_pixel, 1] = src_xs
    out = image.copy()
    ys, xs = np.where(m_u8 == 1)
    for y, x in zip(ys, xs):
        lbl = labels[y, x]
        sy, sx = label_to_coord[lbl]
        out[y, x] = image[sy, sx]
    return out


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

@dataclass
class Method:
    name: str
    fn: Callable[..., np.ndarray]
    params: Dict


def make_method_registry() -> Dict[str, Method]:
    reg: Dict[str, Method] = {}

    def add(name: str, fn, **params):
        reg[name] = Method(name, fn, dict(params))

    # --- Baselines ---
    add("baseline_mean", baseline_mean_color)
    add("baseline_nearest", baseline_nearest_valid)
    add("baseline_telea_r3", baseline_cv2_telea, inpaint_radius=3)
    add("baseline_ns_r3", baseline_cv2_ns, inpaint_radius=3)

    # --- Stock node (as shipped defaults) ---
    add("stock_defaults", run_stock_node,
        patch_size=7, iterations=5, search_radius=50, blend_width=5)

    # --- Technique wedges on configurable core ---
    # Full correct version
    add("core_full_fixed",
        patchmatch_core,
        patch_size=7, iterations=5, search_radius=50,
        use_propagation=True, use_random_search=True,
        use_bidirectional=True, fix_backward_direction=True,
        reject_target_offsets_on_init=True, multiscale_levels=1)

    # Mimic stock node's quirks (buggy bwd pass, no avoid-mask init)
    add("core_mimic_stock",
        patchmatch_core,
        patch_size=7, iterations=5, search_radius=50,
        use_propagation=True, use_random_search=True,
        use_bidirectional=True, fix_backward_direction=False,
        reject_target_offsets_on_init=False, multiscale_levels=1)

    # Ablations
    add("core_no_random_search",
        patchmatch_core,
        patch_size=7, iterations=5, search_radius=50,
        use_random_search=False, fix_backward_direction=True)
    add("core_no_propagation",
        patchmatch_core,
        patch_size=7, iterations=5, search_radius=50,
        use_propagation=False, fix_backward_direction=True)
    add("core_forward_only",
        patchmatch_core,
        patch_size=7, iterations=5, search_radius=50,
        use_bidirectional=False, fix_backward_direction=True)

    # Multi-scale
    add("core_multiscale_2",
        patchmatch_core,
        patch_size=7, iterations=4, search_radius=40,
        multiscale_levels=2, fix_backward_direction=True)
    add("core_multiscale_3",
        patchmatch_core,
        patch_size=7, iterations=3, search_radius=30,
        multiscale_levels=3, fix_backward_direction=True)

    return reg
