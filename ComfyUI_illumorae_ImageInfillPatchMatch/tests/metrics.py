"""Image-quality metrics, restricted to the inpainted (target) region.

All inputs are float32 in [0,1], shape (H,W,3); mask shape (H,W).
"""
from __future__ import annotations

from typing import Dict

import numpy as np

try:
    from skimage.metrics import structural_similarity as _sk_ssim  # type: ignore
    _HAS_SKIMAGE = True
except Exception:  # pragma: no cover
    _HAS_SKIMAGE = False


def _masked_pixels(img: np.ndarray, mask: np.ndarray) -> np.ndarray:
    m = mask > 0.5
    if not m.any():
        return img.reshape(-1, img.shape[-1])
    return img[m]


def mse_masked(pred: np.ndarray, gt: np.ndarray, mask: np.ndarray) -> float:
    p = _masked_pixels(pred, mask)
    g = _masked_pixels(gt, mask)
    if p.size == 0:
        return 0.0
    return float(np.mean((p - g) ** 2))


def mae_masked(pred: np.ndarray, gt: np.ndarray, mask: np.ndarray) -> float:
    p = _masked_pixels(pred, mask)
    g = _masked_pixels(gt, mask)
    if p.size == 0:
        return 0.0
    return float(np.mean(np.abs(p - g)))


def psnr_masked(pred: np.ndarray, gt: np.ndarray, mask: np.ndarray) -> float:
    mse = mse_masked(pred, gt, mask)
    if mse <= 1e-12:
        return 99.0
    return float(10.0 * np.log10(1.0 / mse))


def _ssim_fallback(a: np.ndarray, b: np.ndarray) -> float:
    """Very simple global SSIM approximation (no sliding window).

    Used only if scikit-image is unavailable. Produces a value in
    [-1, 1], same sign convention as SSIM.
    """
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    mu_a, mu_b = a.mean(), b.mean()
    va = a.var()
    vb = b.var()
    cov = ((a - mu_a) * (b - mu_b)).mean()
    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    num = (2 * mu_a * mu_b + c1) * (2 * cov + c2)
    den = (mu_a ** 2 + mu_b ** 2 + c1) * (va + vb + c2)
    if den <= 0:
        return 0.0
    return float(num / den)


def ssim_full(pred: np.ndarray, gt: np.ndarray) -> float:
    """SSIM over the whole image (windowed SSIM cannot be cleanly
    restricted to the target region; we approximate with full-image
    SSIM as a secondary metric).
    """
    if _HAS_SKIMAGE:
        try:
            return float(
                _sk_ssim(gt, pred, channel_axis=-1, data_range=1.0)
            )
        except TypeError:
            # older skimage: multichannel kw
            return float(
                _sk_ssim(gt, pred, multichannel=True, data_range=1.0)
            )
    return _ssim_fallback(pred, gt)


def compute_all(
    pred: np.ndarray, gt: np.ndarray, mask: np.ndarray
) -> Dict[str, float]:
    return {
        "mse": mse_masked(pred, gt, mask),
        "mae": mae_masked(pred, gt, mask),
        "psnr": psnr_masked(pred, gt, mask),
        "ssim": ssim_full(pred, gt),
    }
