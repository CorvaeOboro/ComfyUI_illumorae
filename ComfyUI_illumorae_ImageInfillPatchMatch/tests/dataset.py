"""Synthetic dataset + mask generation for PatchMatch benchmarks.

All images are float32 in [0,1] with shape (H, W, 3).
All masks are float32 in {0,1} with shape (H, W); 1 = target region (to fill), 0 = source region.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Ground-truth image generators
# ---------------------------------------------------------------------------

def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def gt_gradient(size: int, seed: int) -> np.ndarray:
    r = _rng(seed)
    h = w = size
    y = np.linspace(0, 1, h, dtype=np.float32)[:, None]
    x = np.linspace(0, 1, w, dtype=np.float32)[None, :]
    c0 = r.uniform(0, 1, 3).astype(np.float32)
    c1 = r.uniform(0, 1, 3).astype(np.float32)
    t = (0.5 * (x + y))[..., None]
    img = c0 * (1 - t) + c1 * t
    return np.clip(img, 0, 1)


def gt_checker(size: int, seed: int) -> np.ndarray:
    r = _rng(seed)
    tile = int(r.integers(8, 24))
    h = w = size
    yy, xx = np.mgrid[0:h, 0:w]
    pattern = ((yy // tile + xx // tile) % 2).astype(np.float32)
    c0 = r.uniform(0, 1, 3).astype(np.float32)
    c1 = r.uniform(0, 1, 3).astype(np.float32)
    img = c0 * (1 - pattern[..., None]) + c1 * pattern[..., None]
    # small noise
    img = img + r.normal(0, 0.02, img.shape).astype(np.float32)
    return np.clip(img, 0, 1)


def gt_noise_texture(size: int, seed: int) -> np.ndarray:
    """Smoothed color noise - emulates 'natural' texture."""
    r = _rng(seed)
    h = w = size
    base = r.uniform(0, 1, (h, w, 3)).astype(np.float32)
    # multi-scale blur + add
    blur1 = cv2.GaussianBlur(base, (0, 0), sigmaX=8)
    blur2 = cv2.GaussianBlur(base, (0, 0), sigmaX=3)
    img = 0.6 * blur1 + 0.3 * blur2 + 0.1 * base
    # normalize per-channel
    for c in range(3):
        ch = img[..., c]
        img[..., c] = (ch - ch.min()) / (ch.max() - ch.min() + 1e-8)
    return np.clip(img, 0, 1)


def gt_stripes(size: int, seed: int) -> np.ndarray:
    r = _rng(seed)
    h = w = size
    freq = float(r.uniform(0.05, 0.2))
    angle = float(r.uniform(0, np.pi))
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    u = np.cos(angle) * xx + np.sin(angle) * yy
    s = 0.5 + 0.5 * np.sin(2 * np.pi * freq * u)
    c0 = r.uniform(0, 1, 3).astype(np.float32)
    c1 = r.uniform(0, 1, 3).astype(np.float32)
    img = c0 * (1 - s[..., None]) + c1 * s[..., None]
    return np.clip(img, 0, 1)


def gt_mixed(size: int, seed: int) -> np.ndarray:
    """Combine a gradient background with textured foreground patches."""
    bg = gt_gradient(size, seed)
    tex = gt_noise_texture(size, seed + 1)
    r = _rng(seed + 2)
    mask = np.zeros((size, size), dtype=np.float32)
    for _ in range(int(r.integers(2, 5))):
        cx, cy = r.integers(0, size, 2)
        rad = int(r.integers(size // 10, size // 4))
        cv2.circle(mask, (int(cx), int(cy)), rad, 1.0, -1)
    mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=3)
    mask = mask[..., None]
    img = bg * (1 - mask) + tex * mask
    return np.clip(img, 0, 1)


GT_GENERATORS: Dict[str, Callable[[int, int], np.ndarray]] = {
    "gradient": gt_gradient,
    "checker": gt_checker,
    "noise": gt_noise_texture,
    "stripes": gt_stripes,
    "mixed": gt_mixed,
}


# ---------------------------------------------------------------------------
# Mask generators
# ---------------------------------------------------------------------------

def mask_rect(size: int, seed: int, coverage: float = 0.15) -> np.ndarray:
    r = _rng(seed)
    m = np.zeros((size, size), dtype=np.float32)
    area = coverage * size * size
    aspect = float(r.uniform(0.5, 2.0))
    h_r = int(np.sqrt(area / aspect))
    w_r = int(np.sqrt(area * aspect))
    h_r = max(4, min(h_r, size - 4))
    w_r = max(4, min(w_r, size - 4))
    y0 = int(r.integers(0, size - h_r))
    x0 = int(r.integers(0, size - w_r))
    m[y0:y0 + h_r, x0:x0 + w_r] = 1.0
    return m


def mask_ellipse(size: int, seed: int, coverage: float = 0.15) -> np.ndarray:
    r = _rng(seed)
    m = np.zeros((size, size), dtype=np.float32)
    area = coverage * size * size
    aspect = float(r.uniform(0.5, 2.0))
    ry = int(np.sqrt(area / (np.pi * aspect)))
    rx = int(np.sqrt(area * aspect / np.pi))
    ry = max(3, min(ry, size // 2 - 2))
    rx = max(3, min(rx, size // 2 - 2))
    cy = int(r.integers(ry + 1, size - ry - 1))
    cx = int(r.integers(rx + 1, size - rx - 1))
    cv2.ellipse(m, (cx, cy), (rx, ry), 0, 0, 360, 1.0, -1)
    return m


def mask_blob(size: int, seed: int, coverage: float = 0.15) -> np.ndarray:
    """Irregular blob: random points + convex-hull-like dilation."""
    r = _rng(seed)
    m = np.zeros((size, size), dtype=np.float32)
    n_pts = int(r.integers(4, 10))
    cx = int(r.integers(size // 4, 3 * size // 4))
    cy = int(r.integers(size // 4, 3 * size // 4))
    radius = int(np.sqrt(coverage * size * size / np.pi) * 1.2)
    pts = []
    for _ in range(n_pts):
        ang = float(r.uniform(0, 2 * np.pi))
        rr = float(r.uniform(0.3, 1.0)) * radius
        pts.append([cx + rr * np.cos(ang), cy + rr * np.sin(ang)])
    pts = np.array(pts, dtype=np.int32)
    cv2.fillPoly(m, [pts], 1.0)
    m = cv2.GaussianBlur(m, (0, 0), sigmaX=2)
    m = (m > 0.3).astype(np.float32)
    return m


def mask_thin_strip(size: int, seed: int, coverage: float = 0.08) -> np.ndarray:
    r = _rng(seed)
    m = np.zeros((size, size), dtype=np.float32)
    thickness = max(3, int(size * 0.04))
    length = int(coverage * size * size / thickness)
    length = min(length, size - 4)
    y0 = int(r.integers(thickness, size - thickness))
    x0 = int(r.integers(2, size - length - 2))
    angle_deg = float(r.uniform(-30, 30))
    x1 = int(x0 + length * np.cos(np.deg2rad(angle_deg)))
    y1 = int(y0 + length * np.sin(np.deg2rad(angle_deg)))
    x1 = int(np.clip(x1, 0, size - 1))
    y1 = int(np.clip(y1, 0, size - 1))
    cv2.line(m, (x0, y0), (x1, y1), 1.0, thickness)
    return m


MASK_GENERATORS: Dict[str, Callable[[int, int, float], np.ndarray]] = {
    "rect": mask_rect,
    "ellipse": mask_ellipse,
    "blob": mask_blob,
    "strip": mask_thin_strip,
}


# ---------------------------------------------------------------------------
# Dataset sample container
# ---------------------------------------------------------------------------

@dataclass
class Sample:
    name: str
    gt: np.ndarray          # (H,W,3) float32 in [0,1]
    mask: np.ndarray        # (H,W) float32 in {0,1}
    gt_kind: str
    mask_kind: str
    coverage: float

    @property
    def corrupted(self) -> np.ndarray:
        """Ground truth with the target region zeroed-out (pure black
        inside the target region).

        PatchMatch must not 'see' ground-truth pixels in the target
        region; downstream the source region is already controlled by
        the mask, but we zero it for safety / visualization.
        """
        m = self.mask[..., None]
        return (self.gt * (1 - m)).astype(np.float32)


def build_dataset(
    size: int = 96,
    gt_kinds: Optional[List[str]] = None,
    mask_kinds: Optional[List[str]] = None,
    coverages: Optional[List[float]] = None,
    samples_per_combo: int = 1,
    seed: int = 0,
) -> List[Sample]:
    gt_kinds = gt_kinds or list(GT_GENERATORS.keys())
    mask_kinds = mask_kinds or list(MASK_GENERATORS.keys())
    coverages = coverages or [0.10, 0.20]

    dataset: List[Sample] = []
    idx = 0
    for gk in gt_kinds:
        for mk in mask_kinds:
            for cov in coverages:
                for s in range(samples_per_combo):
                    local_seed = seed + 1000 * idx + s
                    gt = GT_GENERATORS[gk](size, local_seed)
                    mask = MASK_GENERATORS[mk](size, local_seed + 7, cov)
                    name = f"{gk}_{mk}_cov{int(cov*100):02d}_s{s}"
                    dataset.append(Sample(name, gt, mask, gk, mk, cov))
                    idx += 1
    return dataset


def save_sample_preview(sample: Sample, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    gt_u8 = (sample.gt * 255).astype(np.uint8)
    mask_u8 = (sample.mask * 255).astype(np.uint8)
    corr_u8 = (sample.corrupted * 255).astype(np.uint8)
    cv2.imwrite(os.path.join(out_dir, f"{sample.name}_gt.png"),
                cv2.cvtColor(gt_u8, cv2.COLOR_RGB2BGR))
    cv2.imwrite(os.path.join(out_dir, f"{sample.name}_mask.png"), mask_u8)
    cv2.imwrite(os.path.join(out_dir, f"{sample.name}_corrupted.png"),
                cv2.cvtColor(corr_u8, cv2.COLOR_RGB2BGR))


def load_real_images(
    folder: str, size: int = 96
) -> List[Tuple[str, np.ndarray]]:
    """Optional: load real images (any format OpenCV supports)."""
    out: List[Tuple[str, np.ndarray]] = []
    if not os.path.isdir(folder):
        return out
    for fname in sorted(os.listdir(folder)):
        path = os.path.join(folder, fname)
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
        img = img.astype(np.float32) / 255.0
        out.append((os.path.splitext(fname)[0], img))
    return out
