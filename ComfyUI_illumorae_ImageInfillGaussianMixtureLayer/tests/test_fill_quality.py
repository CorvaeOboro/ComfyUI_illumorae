"""Fill-quality / self-diagnosis regression test for the
Gaussian Mixture Layer (Galerne-Leclaire 2017) inpainting node.

Why this exists
---------------
The discipline test (``test_source_discipline.py``) only verifies that
the output is invariant to original target-region content - a property
the all-zero / saturated / colour-shifted output ALSO satisfies. So
the discipline test cannot catch the "wildly bright / off-gamut /
saturated" failure mode that has been observed on real 512x512 images
with ~100x100 holes.

This test catches such failures by running the algorithm on synthetic
inputs whose source gamut and ground-truth target content are known,
and asserting four fill-plausibility properties per case:

  1. **Source pass-through, bit-exact.** ``X[source] == img[source]``.
  2. **Gamut containment.** Target-region pixel values stay within a
     small tolerance of the per-channel ``[source_min, source_max]``
     range. Any single pixel further than ``gamut_tol`` outside the
     source range is a "wild-bright-colour" pixel; we report the
     count and fail when it exceeds ``max_wild_pixels_frac`` of the
     target area.
  3. **Mean fidelity.** ``|mean(X[target]) - mean(img[source])|`` per
     channel is within ``mean_tol``. Galerne-Leclaire's stationary-
     Gaussian model preserves the first moment exactly in the
     conditional mean, so this should hold up to sampling noise.
  4. **Saturation guard.** The fraction of clipped (==0 or ==1)
     target pixels per channel must be small. Heavy clipping is the
     signature of an unbounded correction term that has been
     squashed by the final ``clip(0, 1)``.

The test runs on:
  - a uniform colour patch (most stringent: GT is constant, so any
    Gaussian sample with non-zero variance will violate the loose
    gamut bound, but the mean check is exact);
  - a smooth colour gradient (medium difficulty);
  - a textured image (random RGB noise, the regime Galerne-Leclaire
    is actually designed for - the second-order stats should match).

Note on GMM parameter coverage:
  We exercise the default config used by the ComfyUI entry point
  AND a strict config (regularization=1e-1, large CG budget) to
  separate "model-mismatch" failures from "ill-conditioned solve"
  failures. If the strict path passes but the default fails, the
  default ridge is too small.

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

from image_infill_gaussian_mixture_layer import (  # noqa: E402
    illumoraeImageInfillGaussianMixtureLayerNode,
)


# --------------------------------------------------------------------------
# Synthetic test images
# --------------------------------------------------------------------------

def _make_uniform(H: int, W: int, color=(0.4, 0.5, 0.3)) -> np.ndarray:
    img = np.zeros((H, W, 3), dtype=np.float32)
    img[..., 0] = color[0]
    img[..., 1] = color[1]
    img[..., 2] = color[2]
    return img


def _make_gradient(H: int, W: int) -> np.ndarray:
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    return np.stack([
        xx / (W - 1),
        yy / (H - 1),
        0.5 * np.ones_like(xx),
    ], axis=-1).astype(np.float32)


def _make_natural_multiregion(H: int, W: int, seed: int = 0) -> np.ndarray:
    """Mimic a real-world natural photo: a few flat colour regions
    (sky / grass / building-like) with sharp inter-region edges and
    a small amount of additive Gaussian noise. This is the regime
    where users have reported "wildly bright" GMM fills - the
    stationary-Gaussian model is wrong for this content but the
    *numerics* should still stay in-gamut. Used to detect
    correction-term blowups that show up only on multi-modal
    natural inputs."""
    rng = np.random.default_rng(seed)
    img = np.zeros((H, W, 3), dtype=np.float32)
    # Sky-ish blue on top half.
    img[:H // 2, :, :] = (0.40, 0.60, 0.95)
    # Grass-ish green on bottom half.
    img[H // 2:, :, :] = (0.20, 0.55, 0.20)
    # Building / wall on the left quarter, overrides both.
    img[:, :W // 4, :] = (0.65, 0.50, 0.40)
    # Mild per-pixel sensor noise so source variance is non-zero.
    img = img + 0.02 * rng.standard_normal(img.shape).astype(np.float32)
    return np.clip(img, 0.0, 1.0).astype(np.float32)


def _make_texture(H: int, W: int, seed: int = 0,
                  mu=(0.4, 0.5, 0.3), sigma=0.08) -> np.ndarray:
    """Stationary Gaussian texture - the regime Galerne-Leclaire is
    designed for. Built by smoothing white noise (band-limited) so
    the result has a meaningful spatial autocorrelation, then biased
    to ``mu`` and rescaled to ``sigma`` per channel."""
    rng = np.random.default_rng(seed)
    raw = rng.standard_normal((H, W, 3)).astype(np.float32)
    # Cheap separable smoothing kernel via FFT (Gaussian, sigma=2 px).
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    cy, cx = H // 2, W // 2
    g = np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * 2.0 ** 2))
    g = g / g.sum()
    G = np.fft.fft2(np.fft.ifftshift(g))
    out = np.zeros_like(raw)
    for c in range(3):
        R = np.fft.fft2(raw[..., c])
        out[..., c] = np.real(np.fft.ifft2(R * G))
    out -= out.mean(axis=(0, 1), keepdims=True)
    out /= max(float(out.std()), 1e-6)
    out = mu + sigma * out
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def _make_central_rect_mask(H: int, W: int, hole_h: int,
                            hole_w: int) -> np.ndarray:
    msk = np.zeros((H, W), dtype=np.float32)
    y0 = (H - hole_h) // 2
    x0 = (W - hole_w) // 2
    msk[y0:y0 + hole_h, x0:x0 + hole_w] = 1.0
    return msk


# --------------------------------------------------------------------------
# Quality checks
# --------------------------------------------------------------------------

def _check(
    name: str,
    img: np.ndarray,
    msk: np.ndarray,
    filled: np.ndarray,
    *,
    gamut_tol: float,
    max_wild_frac: float,
    mean_tol: float,
    sat_tol_frac: float,
    verbose: bool,
) -> bool:
    target = msk > 0.5
    source = ~target

    # 1. Source pass-through.
    src_delta = float(np.max(np.abs(filled[source] - img[source])))
    ok_source = src_delta <= 1e-6

    # 2. Gamut containment per channel.
    src_min = img[source].min(axis=0)
    src_max = img[source].max(axis=0)
    tgt = filled[target]
    below = tgt < (src_min - gamut_tol)
    above = tgt > (src_max + gamut_tol)
    wild_per_ch = (below | above).sum(axis=0)
    wild_total_frac = float((below | above).any(axis=-1).mean())
    ok_gamut = wild_total_frac <= max_wild_frac

    # 3. Mean fidelity.
    src_mean = img[source].mean(axis=0)
    tgt_mean = tgt.mean(axis=0)
    mean_delta = float(np.max(np.abs(tgt_mean - src_mean)))
    ok_mean = mean_delta <= mean_tol

    # 4. Saturation guard - *relative* to source. If the source
    # itself has pixels at 0 or 1 (e.g. a bright blue sky that was
    # clipped during image capture), legitimate fills in that
    # area will also reach those endpoints, which is fine. We
    # only fail when the target's at-endpoint fraction exceeds the
    # source's by more than ``sat_tol_frac`` - that extra mass is
    # the unnatural overshoot the algorithm must not introduce.
    src_vals = img[source]
    src_sat = float(
        ((src_vals <= 0.0) | (src_vals >= 1.0)).any(axis=-1).mean()
    )
    tgt_sat = float(((tgt <= 0.0) | (tgt >= 1.0)).any(axis=-1).mean())
    excess_sat = max(0.0, tgt_sat - src_sat)
    ok_sat = excess_sat <= sat_tol_frac

    ok = ok_source and ok_gamut and ok_mean and ok_sat
    mark = "OK  " if ok else "FAIL"
    if verbose:
        print(
            f"  [{mark}] {name:<32} "
            f"src_delta={src_delta:.2e} "
            f"wild={wild_total_frac*100:5.2f}% "
            f"mean_delta={mean_delta:.3f} "
            f"sat={tgt_sat*100:5.2f}% (src_sat={src_sat*100:4.1f}%, excess={excess_sat*100:5.2f}%) "
            f"src_range=[{src_min.min():.2f},{src_max.max():.2f}] "
            f"tgt_range=[{tgt.min():.2f},{tgt.max():.2f}]"
        )
        if not ok:
            reasons = []
            if not ok_source:
                reasons.append(f"source pass-through broken (delta={src_delta:.2e})")
            if not ok_gamut:
                reasons.append(
                    f"out-of-gamut: {wild_total_frac*100:.2f}% > "
                    f"{max_wild_frac*100:.2f}% "
                    f"(per-ch counts={wild_per_ch.tolist()})"
                )
            if not ok_mean:
                reasons.append(
                    f"mean drift {mean_delta:.3f} > {mean_tol:.3f}"
                )
            if not ok_sat:
                reasons.append(
                    f"excess saturation {excess_sat*100:.2f}% > "
                    f"{sat_tol_frac*100:.2f}% "
                    f"(tgt_sat={tgt_sat*100:.2f}%, "
                    f"src_sat={src_sat*100:.2f}%)"
                )
            for r in reasons:
                print(f"         -> {r}")
    return ok


def _run(node, img, msk, **overrides):
    params = dict(
        cg_max_iterations=200,
        cg_tolerance=1e-6,
        add_innovation=True,
        innovation_strength=1.0,
        regularization=1e-3,
        seed=0,
        debug_prints=False,
    )
    params.update(overrides)
    filled, _ = node._run_single(
        img.astype(np.float32),
        msk.astype(np.float32),
        params["cg_max_iterations"],
        params["cg_tolerance"],
        params["add_innovation"],
        params["innovation_strength"],
        params["regularization"],
        params["seed"],
        params["debug_prints"],
    )
    return filled.astype(np.float32)


# --------------------------------------------------------------------------
# Phases
# --------------------------------------------------------------------------

def _run_phase(label, node, cases, *, verbose=True, **overrides) -> bool:
    if verbose:
        print(f"\n[{label}] regularization={overrides.get('regularization', 1e-3)} "
              f"add_innovation={overrides.get('add_innovation', True)}")
    all_ok = True
    for case in cases:
        filled = _run(node, case["img"], case["msk"], **overrides)
        all_ok &= _check(
            case["name"], case["img"], case["msk"], filled,
            gamut_tol=case["gamut_tol"],
            max_wild_frac=case["max_wild_frac"],
            mean_tol=case["mean_tol"],
            sat_tol_frac=case["sat_tol_frac"],
            verbose=verbose,
        )
    return all_ok


def _build_cases():
    cases = []

    # Case 1: Uniform colour, 512x512 image with a 100x100 hole. This
    # is the user-reported failure regime. Source variance is exactly
    # 0, so the unconditional Gaussian sample collapses to mu and the
    # output should be exactly mu inside the hole. ANY deviation > a
    # tiny tolerance signals a numerical instability.
    cases.append(dict(
        name="uniform_512_hole100",
        img=_make_uniform(512, 512, color=(0.4, 0.5, 0.3)),
        msk=_make_central_rect_mask(512, 512, 100, 100),
        gamut_tol=0.02,
        max_wild_frac=0.001,
        mean_tol=0.02,
        sat_tol_frac=0.001,
    ))

    # Case 2: Smooth gradient, 256x256 image with a 64x64 hole.
    # Galerne-Leclaire's stationary-Gaussian model is wrong here
    # (gradient is not stationary), but the gamut should still be
    # respected and there should be no extreme saturation.
    cases.append(dict(
        name="gradient_256_hole64",
        img=_make_gradient(256, 256),
        msk=_make_central_rect_mask(256, 256, 64, 64),
        gamut_tol=0.05,
        max_wild_frac=0.05,
        mean_tol=0.10,
        sat_tol_frac=0.05,
    ))

    # Case 3: Stationary Gaussian texture, 256x256 with a 64x64 hole.
    # This is the algorithm's intended regime; gamut + mean + sat
    # should all be tight.
    cases.append(dict(
        name="texture_256_hole64",
        img=_make_texture(256, 256, seed=1),
        msk=_make_central_rect_mask(256, 256, 64, 64),
        gamut_tol=0.10,
        max_wild_frac=0.10,
        mean_tol=0.05,
        sat_tol_frac=0.05,
    ))

    # Case 4: Realistic 512x512 with 100x100 hole, textured. Direct
    # match to the user's reported failure case.
    cases.append(dict(
        name="texture_512_hole100",
        img=_make_texture(512, 512, seed=2),
        msk=_make_central_rect_mask(512, 512, 100, 100),
        gamut_tol=0.10,
        max_wild_frac=0.10,
        mean_tol=0.05,
        sat_tol_frac=0.05,
    ))

    # Case 5: Natural multi-region image (sky + grass + building) at
    # 512x512 with a 100x100 hole DEAD CENTER (straddles the
    # sky/grass colour boundary). The user's reported "wildly bright
    # blue / splotches of green" failure is exactly this regime.
    # The stationary-Gaussian model is wrong here, but the output
    # must still respect the source gamut and not saturate.
    img_nat = _make_natural_multiregion(512, 512, seed=0)
    cases.append(dict(
        name="natural_512_centerhole100",
        img=img_nat,
        msk=_make_central_rect_mask(512, 512, 100, 100),
        # Multi-modal source has a wide gamut already; we tolerate
        # samples slightly outside it but flag wild excursions.
        gamut_tol=0.10,
        max_wild_frac=0.15,
        # Mean check is loose because two large regions of very
        # different colour are blended in the source mean.
        mean_tol=0.20,
        sat_tol_frac=0.10,
    ))

    # Case 6: Natural multi-region with the hole on the colour
    # boundary between sky and building (most visually demanding).
    msk_corner = np.zeros((512, 512), dtype=np.float32)
    # 100x100 hole straddling the building/sky vertical edge.
    msk_corner[206:306, 78:178] = 1.0
    cases.append(dict(
        name="natural_512_edgehole100",
        img=img_nat,
        msk=msk_corner,
        gamut_tol=0.10,
        max_wild_frac=0.15,
        mean_tol=0.20,
        sat_tol_frac=0.10,
    ))

    return cases


def run(verbose: bool = True) -> bool:
    node = illumoraeImageInfillGaussianMixtureLayerNode()
    cases = _build_cases()

    # Phase A: default settings (regularization=1e-3, innovation on).
    ok_default = _run_phase(
        "Phase A: default settings", node, cases,
        verbose=verbose, regularization=1e-3, add_innovation=True,
    )

    # Phase B: kriging mean only (no innovation) - tests whether the
    # CONDITIONAL MEAN is well-bounded. If A fails but B passes, the
    # innovation amplifies a numerical issue. If both fail, the
    # kriging solve itself is unstable.
    ok_kriging = _run_phase(
        "Phase B: kriging mean (innovation off)", node, cases,
        verbose=verbose, regularization=1e-3, add_innovation=False,
    )

    # Phase C: heavy regularization (1e-1). Should pass everywhere if
    # the only issue is ridge under-scaling.
    ok_strict = _run_phase(
        "Phase C: heavy regularization (1e-1)", node, cases,
        verbose=verbose, regularization=1e-1, add_innovation=True,
    )

    return ok_default and ok_kriging and ok_strict


if __name__ == "__main__":
    print("Gaussian Mixture Layer fill-quality regression test")
    print("=" * 78)
    ok = run(verbose=True)
    print("=" * 78)
    if ok:
        print("PASS: all phases produce in-gamut, mean-faithful, "
              "non-saturated fills.")
        sys.exit(0)
    else:
        print("FAIL: at least one case produced out-of-gamut / "
              "saturated / mean-shifted output. See per-line details.")
        sys.exit(1)
