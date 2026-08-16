"""Diagnostic tests for CLAHE region-uniformity and contrast behavior.

These tests construct synthetic images with distinct regions (flat sky,
textured mid-band, dark shadow) and measure how different CLAHE
parameter configurations affect the per-region L-channel standard
deviation. The goal is to quantify the reported symptom: "normalized
high contrast details across all regions" not being achieved with the
current defaults.

Metrics reported per configuration:
  - Per-region L-channel std (higher = more local contrast amplification)
  - sky/dark std ratio (close to 1.0 = uniform enhancement;
    >> 1.0 = over-amplification in flat regions, the reported symptom)
  - Tile size in pixels (image_size / grid_size)

Usage:
    python -m tests.test_clahe_quality
    pytest tests/test_clahe_quality.py
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


# --------------------------------------------------------------------------
# Synthetic test images
# --------------------------------------------------------------------------

def _make_regioned_image(size: int = 1024, seed: int = 42) -> np.ndarray:
    """A 1024x1024 RGB image with three distinct regions:
    - top third:    flat "sky" (near-uniform light gray)
    - middle third: textured band (sinusoidal + noise)
    - bottom third: flat "shadow" (near-uniform dark gray)
    """
    rng = np.random.default_rng(seed)
    img = np.zeros((size, size, 3), dtype=np.float32)
    h3 = size // 3

    # Sky: light gray with tiny noise
    img[:h3] = [180, 190, 200]
    img[:h3] += rng.normal(0, 2, (h3, size, 3))

    # Mid: sinusoidal texture + moderate noise
    xs = np.linspace(0, 50, size, dtype=np.float32)
    wave = 40 * np.sin(xs)[None, :, None]
    img[h3:2 * h3] = 128 + wave
    img[h3:2 * h3] += rng.normal(0, 8, (h3, size, 3))

    # Dark: dark gray with tiny noise
    img[2 * h3:] = [20, 25, 30]
    img[2 * h3:] += rng.normal(0, 2, (size - 2 * h3, size, 3))

    return np.clip(img, 0, 255).astype(np.uint8)


def _make_gradient_image(size: int = 512) -> np.ndarray:
    """A smooth left-to-right gradient (low local contrast everywhere)."""
    xx = np.linspace(0, 255, size, dtype=np.float32)
    img = np.broadcast_to(xx[None, :, None], (size, size, 3)).copy()
    return img.astype(np.uint8)


def _make_noise_image(size: int = 512, seed: int = 7) -> np.ndarray:
    """Uniform random noise (high local contrast everywhere)."""
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, (size, size, 3), dtype=np.uint8)


# --------------------------------------------------------------------------
# CLAHE application (mirrors the node's LAB-L-channel approach)
# --------------------------------------------------------------------------

def _apply_clahe_rgb(img_uint8: np.ndarray, clip: float, grid: tuple) -> np.ndarray:
    """Apply CLAHE to the L channel of an RGB uint8 image."""
    lab = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    cl = cv2.createCLAHE(clipLimit=clip, tileGridSize=grid)
    l_out = cl.apply(l)
    return cv2.cvtColor(cv2.merge((l_out, a, b)), cv2.COLOR_LAB2RGB)


def _region_stds(l_channel: np.ndarray, size: int) -> dict:
    """Per-region L-channel std for the 3-region test image."""
    h3 = size // 3
    return {
        "sky": float(l_channel[:h3].std()),
        "mid": float(l_channel[h3:2 * h3].std()),
        "dark": float(l_channel[2 * h3:].std()),
    }


# --------------------------------------------------------------------------
# Configurations to compare
# --------------------------------------------------------------------------

CONFIGS = [
    ("12x12 clip=3.0 (current default)", 3.0, (12, 12)),
    ("8x8  clip=2.0 (opencv standard)",  2.0, (8, 8)),
    ("4x4  clip=2.0 (large tiles)",      2.0, (4, 4)),
    ("8x8  clip=3.0",                    3.0, (8, 8)),
    ("16x16 clip=2.0 (small tiles)",     2.0, (16, 16)),
    ("8x8  clip=5.0 (high clip)",        5.0, (8, 8)),
]


# --------------------------------------------------------------------------
# Tests
# --------------------------------------------------------------------------

def test_regioned_image_uniformity():
    """On a 3-region image, measure per-region std amplification for
    each CLAHE configuration. Prints a comparison table. Does not
    hard-fail (diagnostic), but asserts the test image itself has
    the expected region structure.
    """
    size = 1024
    img = _make_regioned_image(size)
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l_in, _, _ = cv2.split(lab)
    input_stds = _region_stds(l_in, size)

    # Sanity: the test image must have distinct regions.
    assert input_stds["sky"] < input_stds["mid"], (
        "Test image sky region should have lower std than mid region"
    )
    assert input_stds["dark"] < input_stds["mid"], (
        "Test image dark region should have lower std than mid region"
    )

    print("\n=== Regioned image (1024x1024, 3 regions) ===")
    print(f"Input  L std: sky={input_stds['sky']:.1f}  "
          f"mid={input_stds['mid']:.1f}  "
          f"dark={input_stds['dark']:.1f}")
    print()
    print(f"{'Config':<38} {'sky':>7} {'mid':>7} {'dark':>7} "
          f"{'sky/dark':>9} {'tile_px':>8}")
    print("-" * 82)

    results = []
    for name, clip, grid in CONFIGS:
        out = _apply_clahe_rgb(img, clip, grid)
        l_out = cv2.cvtColor(out, cv2.COLOR_RGB2LAB)
        l_ch, _, _ = cv2.split(l_out)
        stds = _region_stds(l_ch, size)
        ratio = stds["sky"] / max(stds["dark"], 1e-6)
        tile_px = size // grid[0]
        print(f"{name:<38} {stds['sky']:>7.1f} {stds['mid']:>7.1f} "
              f"{stds['dark']:>7.1f} {ratio:>9.2f} {tile_px:>8}")
        results.append((name, stds, ratio))

    print()
    print("Key: sky/dark ratio close to 1.0 = uniform enhancement")
    print("     sky/dark >> 1.0 = over-amplification in flat regions")


def test_gradient_image_amplification():
    """On a smooth gradient (no local contrast), CLAHE should not
    introduce significant contrast. Measures std amplification.
    """
    size = 512
    img = _make_gradient_image(size)
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l_in, _, _ = cv2.split(lab)
    input_std = float(l_in.std())

    print("\n=== Gradient image (512x512, smooth L-to-R) ===")
    print(f"Input  L std: {input_std:.1f}")
    print()
    print(f"{'Config':<38} {'L std':>7} {'amp':>7}")
    print("-" * 56)

    for name, clip, grid in CONFIGS:
        out = _apply_clahe_rgb(img, clip, grid)
        l_out = cv2.cvtColor(out, cv2.COLOR_RGB2LAB)
        l_ch, _, _ = cv2.split(l_out)
        out_std = float(l_ch.std())
        amp = out_std / max(input_std, 1e-6)
        print(f"{name:<38} {out_std:>7.1f} {amp:>7.2f}x")

    print()
    print("A smooth gradient has no local detail to enhance.")
    print("High amplification here = CLAHE is amplifying quantization "
          "noise / tile boundaries.")


def test_noise_image_amplification():
    """On uniform noise (already max local contrast), CLAHE should
    not amplify further. Measures std change.
    """
    size = 512
    img = _make_noise_image(size)
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l_in, _, _ = cv2.split(lab)
    input_std = float(l_in.std())

    print("\n=== Noise image (512x512, uniform random) ===")
    print(f"Input  L std: {input_std:.1f}")
    print()
    print(f"{'Config':<38} {'L std':>7} {'amp':>7}")
    print("-" * 56)

    for name, clip, grid in CONFIGS:
        out = _apply_clahe_rgb(img, clip, grid)
        l_out = cv2.cvtColor(out, cv2.COLOR_RGB2LAB)
        l_ch, _, _ = cv2.split(l_out)
        out_std = float(l_ch.std())
        amp = out_std / max(input_std, 1e-6)
        print(f"{name:<38} {out_std:>7.1f} {amp:>7.2f}x")

    print()
    print("Noise already fills the histogram; CLAHE should not amplify.")


def test_tile_boundary_artifacts():
    """Check for visible tile-seam artifacts by measuring the L-channel
    gradient at tile boundaries vs. interior. High boundary/interior
    ratio = visible tiling artifacts.
    """
    size = 1024
    img = _make_regioned_image(size)
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l_in, _, _ = cv2.split(lab)

    print("\n=== Tile boundary artifacts (1024x1024 regioned) ===")
    print(f"{'Config':<38} {'bound':>7} {'interior':>9} {'ratio':>7}")
    print("-" * 65)

    for name, clip, grid in CONFIGS:
        out = _apply_clahe_rgb(img, clip, grid)
        l_out = cv2.cvtColor(out, cv2.COLOR_RGB2LAB)
        l_ch, _, _ = cv2.split(l_out)

        # Measure horizontal gradient magnitude
        gx = np.abs(np.diff(l_ch.astype(np.float32), axis=1))
        h, w = gx.shape
        tile_w = size // grid[0]

        # Boundary columns: near tile edges
        bound_cols = set()
        for i in range(1, grid[0]):
            c = i * tile_w
            for delta in range(-2, 3):
                if 0 <= c + delta < w:
                    bound_cols.add(c + delta)
        bound_mask = np.zeros(w, dtype=bool)
        for c in bound_cols:
            bound_mask[c] = True

        bound_grad = float(gx[:, bound_mask].mean())
        interior_grad = float(gx[:, ~bound_mask].mean())
        ratio = bound_grad / max(interior_grad, 1e-6)
        print(f"{name:<38} {bound_grad:>7.1f} {interior_grad:>9.1f} {ratio:>7.2f}")

    print()
    print("ratio close to 1.0 = no visible tile seams")
    print("ratio >> 1.0 = tile boundary artifacts")


if __name__ == "__main__":
    tests = [
        test_regioned_image_uniformity,
        test_gradient_image_amplification,
        test_noise_image_amplification,
        test_tile_boundary_artifacts,
    ]
    for t in tests:
        print(f"\n--- {t.__name__} ---")
        t()
        print()
    print("All diagnostic tests completed.")
