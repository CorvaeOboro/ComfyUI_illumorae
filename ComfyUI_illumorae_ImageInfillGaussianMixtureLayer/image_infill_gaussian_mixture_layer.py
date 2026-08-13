"""
Image Infill Gaussian Mixture Layer
An approximate attempt at implementing the Gaussian texture conditional simulation,
referencing Galerne-Leclaire 2017 algorithm.
Stationary-Gaussian texture inpainting via FFT moving-average + CG kriging.

# References:
#   Galerne, B. and Leclaire, A. (2017). "Texture inpainting using
#     efficient Gaussian conditional simulation." SIAM J. Imaging
#     Sciences 10(3), 1446-1474.
#   Galerne, B. and Leclaire, A. (2017). "An Algorithm for Gaussian
#     Texture Inpainting." Image Processing On Line (IPOL) 7, 262-277.
#
# Background:
#   Le Ravalec, M. et al. (2000). "The FFT moving average (FFT-MA)
#     generator: an efficient numerical method for generating and
#     conditioning Gaussian simulations." Mathematical Geology 32(6),
#     701-723.

STATUS:: working
TITLE::Image Infill Gaussian Mixture Layer
DESCRIPTIONSHORT::Gaussian texture inpainting by FFT-based conditional simulation (Galerne-Leclaire 2017); best for homogeneous microtextures.
VERSION::20260812
IMAGE::comfyui_illumorae_image_infill_gaussian_mixture_layer.png
GROUP::Image
"""
#region IMPORTS
from __future__ import annotations

import numpy as np
import cv2
from typing import Tuple, TYPE_CHECKING

# torch is optional dependency used only by the ComfyUI image entrypoint
# guarded import for testing in basic environments without torch
try:
    import torch  # type: ignore
    _HAS_TORCH = True
except Exception:
    torch = None  # type: ignore
    _HAS_TORCH = False

if TYPE_CHECKING:
    import torch  # noqa: F401
#endregion

"""
# Algorithm summary (single-Gaussian case):
#   Model the image "u" as a realization of a stationary Gaussian random
#   field with mean vector ``mu`` (one scalar per channel, estimated as
#   the source-region mean) and spot-noise texton
#       t[p] = (u[p] - mu) * M[p] / sqrt(|S|)   for p in S
#       t[p] = 0                                 otherwise
#   (shape: H, W, C; ``M`` is the source indicator, ``|S|`` the source
#   pixel count). The field has covariance
#       Sigma = t (*) flip(t)      (circular convolution)
#   whose spectrum is ``|fft(t)|^2`` and is therefore diagonalized by
#   the DFT.
#
#   INPAINTING by conditional simulation proceeds as:
#     (a) Draw W ~ N(0, I_{H*W}) a single-channel white noise field.
#     (b) Unconditional sample F = mu + t (*) W (per channel, via FFT).
#     (c) Residual at source:   b = u_S - F_S.
#     (d) Solve Sigma_SS z = b for each channel via conjugate gradient,
#         where each matvec Sigma v is computed by scattering v into
#         an H*W image at S, FFT-convolving with the autocorrelation
#         r = t (*) flip(t) (precomputed spectrum = |fft(t)|^2), then
#         gathering back at S.
#     (e) Correction: c = Sigma_.S z  (same FFT-conv with r), full-image.
#     (f) Output X = F + c; re-stamp X_S <- u_S to eliminate residual
#         CG error so source pixels pass through bit-exact.
#
#   Complexity: O(K * HW * log(HW)) where K is the number of CG
#   iterations (typically < 100). Memory: O(HW).
#
#   Source-discipline: every quantity (mu, t, F, b, z, c) is a
#   function of u_S only. Target-region pixels of the original image
#   are never read. Step (f) is a re-stamp, not a blend, so source
#   pixels are preserved bit-exact.
"""


class illumoraeImageInfillGaussianMixtureLayerNode:
    """Gaussian-field texture inpainting (Galerne-Leclaire 2017).

    Terminology (consistent with the rest of this repository):
      - **target region** ``Omega`` : pixels to inpaint (mask == 1).
      - **source region** ``Phi``   : known exemplar pixels (mask == 0).

    Pipeline per image:
      1. Estimate channel-wise mean ``mu`` from source pixels.
      2. Build a centered, zero-extended spot-noise texton ``t`` from
         the source pixels; its DFT gives the Gaussian spectrum.
      3. Sample unconditional Gaussian ``F = mu + t (*) W`` via FFT.
      4. Conjugate-gradient solve ``Sigma_SS z = u_S - F_S``
         per channel, using FFT-based masked convolution for the
         matvec operator.
      5. Full-image correction ``c = Sigma_.S z``. Output
         ``X = F + c`` with source pixels re-stamped bit-exact.

    This algorithm produces a plausible *sample* from the Gaussian
    field conditioned on the source pixels.
    Suited to homogeneous microtextures (grass, sand, fabric,
    stone). For images with strong structure (edges, geometry), run
    the exemplar-region-fill or PatchMatch nodes first; this node
    complements them by filling pure-texture residuals.

    Source-discipline:
      Every internal quantity depends only on ``u_S`` (source pixels);
      ``u_T`` (original target pixels) is never read. The final
      re-stamp guarantees ``X_S = u_S`` bit-exact (no CG residual).
    """

    #region INIT
    def __init__(self):
        pass
    #endregion

    #region C-DETREND
    @staticmethod
    def _auto_trend_sigma(source_mask: np.ndarray) -> float:
        """Heuristic detrending sigma based on hole geometry.

        The normalised-convolution trend (``_compute_smooth_trend``)
        needs a Gaussian wide enough that the furthest target pixel
        still receives appreciable weight from some source pixel -
        otherwise the denominator vanishes and the trend collapses
        to the global mean there, which is precisely the regime
        that triggers multi-modal Galerne-Leclaire blow-up. We set

            sigma = 1.5 * max_{p in target} dist(p, source)

        so the weight at the furthest target is
        ``exp(-(1/1.5)^2 / 2) ~= 0.8`` - plenty. Clamped to
        ``[8, 0.5 * min(H, W)]`` to avoid pathological tiny/huge
        kernels.
        """
        H, W = source_mask.shape
        target_u8 = (source_mask < 0.5).astype(np.uint8)
        n_tgt = int(target_u8.sum())
        if n_tgt == 0 or n_tgt == H * W:
            return 8.0
        dist = cv2.distanceTransform(target_u8, cv2.DIST_L2, 3)
        max_dist = float(dist.max())
        sigma = max(8.0, 1.5 * max_dist)
        sigma = min(sigma, 0.5 * float(min(H, W)))
        return float(sigma)

    @staticmethod
    def _compute_smooth_trend(
        image: np.ndarray,
        source_mask: np.ndarray,
        sigma: float,
    ) -> np.ndarray:
        """Normalised-convolution (Knutsson 1993) trend field ``m``.

        Defined everywhere on the image as

            m(x) = (G_sigma * (u * M))(x) / (G_sigma * M)(x)

        where ``G_sigma`` is a Gaussian kernel of standard deviation
        ``sigma``. At source pixels ``m`` is a locally weighted
        average of ``u``; at target pixels it is a smooth
        extrapolation of nearby source averages. This is the
        preprocessing step used by Galerne-Leclaire's IPOL reference
        code to handle *non-stationary* natural images: the
        stationary-Gaussian field model is fit to the **residual**
        ``u - m`` (which is zero-mean and approximately stationary on
        the source) instead of to ``u`` directly. Without this step,
        a multi-modal source (sky + grass + wall) yields a texton
        with huge low-frequency energy, the kriging correction
        overshoots, and the clipped output looks bright / saturated.

        Source-dependent: ``u`` is read only via ``u * M``,
        so target-region values are not used.
        """
        src_f = source_mask.astype(np.float32)
        n_src = float(src_f.sum())
        if n_src <= 0.0:
            return np.zeros_like(image, dtype=np.float32)
        sigma_eff = max(1.0, float(sigma))
        # Use an odd kernel sized to +/- 3 sigma (captures > 99.7%
        # of the Gaussian mass; cv2 default "auto" can round short).
        ksize = int(2 * round(3.0 * sigma_eff) + 1)
        if ksize % 2 == 0:
            ksize += 1
        masked = image.astype(np.float32) * src_f[..., None]
        num = cv2.GaussianBlur(
            masked, (ksize, ksize), sigma_eff, sigma_eff,
            borderType=cv2.BORDER_REFLECT101,
        )
        den = cv2.GaussianBlur(
            src_f, (ksize, ksize), sigma_eff, sigma_eff,
            borderType=cv2.BORDER_REFLECT101,
        )
        den_safe = np.maximum(den, 1e-6)
        m = num / den_safe[..., None]
        # Global-mean fallback where normalization is small
        global_mu = (masked.reshape(-1, image.shape[-1]).sum(axis=0)
                     / n_src).astype(np.float32)
        bad = (den < 1e-5)
        if bad.any():
            m[bad] = global_mu
        return m.astype(np.float32)
    #endregion

    #region C-TEXTON
    @staticmethod
    def _build_texton(
        image: np.ndarray,
        source_mask: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Build the centered spot-noise texton and channel means.

        Given an image ``u`` (H, W, C) and a binary source mask ``M``
        (H, W) with entries 1 on source / 0 on target, returns
        ``(mu, t)`` where

            mu[c] = mean_{p in S} u[p, c]
            t[p, c] = (u[p, c] - mu[c]) * M[p] / sqrt(|S|)

        Normalizing by ``sqrt(|S|)`` ensures that the spectrum
        ``|fft(t)|^2`` is an unbiased estimate of the source's PSD:
        variance of the synthesized field matches the empirical
        variance of the source sample as the image size grows.

        Strictly source-dependent: ``u`` is read only at positions
        where ``M == 1``. Target-region values of ``u`` are silently
        zeroed by the ``* M`` factor and never used.
        """
        H, W = source_mask.shape
        n_src = float(source_mask.sum())
        if n_src <= 0.0:
            mu = np.zeros(image.shape[-1], dtype=np.float32)
            t = np.zeros_like(image, dtype=np.float32)
            return mu, t
        src_f = source_mask.astype(np.float32)
        # Per-channel source-only mean. Use a masked sum to avoid reading
        # target pixels (multiply-by-indicator ensures target values do
        # not contribute even if non-zero in the input buffer).
        masked = image.astype(np.float32) * src_f[..., None]
        mu = masked.reshape(-1, image.shape[-1]).sum(axis=0) / n_src
        mu = mu.astype(np.float32)
        # Centered, zero-extended, normalized texton.
        t = (image.astype(np.float32) - mu) * src_f[..., None]
        t = t / np.sqrt(n_src, dtype=np.float32)
        return mu, t.astype(np.float32)
    #endregion

    #region C-FFT
    @staticmethod
    def _fft_convolve(
        spectrum: np.ndarray,
        field: np.ndarray,
    ) -> np.ndarray:
        """Circular convolution in the Fourier domain.

        ``spectrum`` is a precomputed DFT of a real kernel (shape
        ``(H, W)`` or ``(H, W, C)``); ``field`` is a real spatial
        array with matching shape. Returns the real part of the
        circular convolution.
        """
        F = np.fft.rfft2(field, axes=(0, 1))
        return np.fft.irfft2(spectrum * F, s=field.shape[:2], axes=(0, 1))
    #endregion

    #region C-KRIGE
    def _apply_sigma(
        self,
        v_source: np.ndarray,
        source_mask: np.ndarray,
        psd: np.ndarray,
        ridge: np.ndarray,
    ) -> np.ndarray:
        """Apply the masked autocovariance operator ``Sigma_SS + ridge*I``.

        For each channel, the Gaussian-field covariance is the
        autocorrelation of the texton, which is diagonalized in the
        DFT basis with spectrum ``psd = |fft(t)|^2``. The masked
        operator acts on source-only vectors ``v`` as:

            ((Sigma_SS + r I) v)[p in S] =
                ( IFFT( psd * FFT(extend(v)) ) )[p]  +  r * v[p]

        where ``extend(v)`` scatters ``v`` to a full-image buffer at
        source positions (zero elsewhere). Implemented channel-wise:
        ``v_source`` has shape ``(N_S, C)``, ``psd`` has shape
        ``(H, W//2+1, C)``, ``ridge`` has shape ``(C,)``.

        The ridge term ``r * v`` is **Tikhonov regularization**: the
        bare ``Sigma_SS`` has compact-support / band-limited spectrum
        and is rank-deficient, so CG without ridge amplifies
        null-space components without bound and produces saturated,
        color-shifted output. ``r`` is auto-scaled per channel by the
        PSD peak ``max_k |fft(t_c)|^2`` (see the ridge-scaling comment
        in ``_run_single``), so the CG condition number is
        ``1 / regularization`` regardless of how concentrated the
        spectrum is.
        """
        H, W = source_mask.shape
        C = v_source.shape[-1]
        full = np.zeros((H, W, C), dtype=np.float32)
        # Scatter source vector to full field.
        sy, sx = np.where(source_mask > 0.5)
        full[sy, sx, :] = v_source.astype(np.float32)
        # FFT-convolution with the PSD (Sigma_SS v).
        conv = self._fft_convolve(psd, full).astype(np.float32)
        # Tikhonov ridge: + ridge[c] * v_source[c] per channel.
        return (conv[sy, sx, :]
                + ridge.reshape(1, C).astype(np.float32)
                * v_source.astype(np.float32))

    def _conjugate_gradient(
        self,
        b_source: np.ndarray,
        source_mask: np.ndarray,
        psd: np.ndarray,
        ridge: np.ndarray,
        max_iter: int,
        tol: float,
        debug: bool,
    ) -> Tuple[np.ndarray, int, float]:
        """Solve ``Sigma_SS z = b`` by conjugate gradient, per channel.

        ``Sigma_SS`` is symmetric positive semi-definite. CG
        converges in at most ``N_S`` iterations; in practice
        roughly 20-100 iterations suffice.

        Returns ``(z, iterations_used, final_residual_norm)``.
        """
        x = np.zeros_like(b_source, dtype=np.float32)
        r = b_source.astype(np.float32).copy()
        p = r.copy()
        # Per-channel residual-norm tracking, but CG runs jointly for
        # all channels with independent state via vectorized dot
        # products.
        rs_old = np.einsum('ij,ij->j', r, r).astype(np.float64)
        b_norm = np.sqrt(np.einsum('ij,ij->j', b_source, b_source)
                         .astype(np.float64))
        b_norm_safe = np.maximum(b_norm, 1e-30)
        tol2 = float(tol) ** 2
        last_rel = float("inf")
        it = 0
        for it in range(1, max_iter + 1):
            Ap = self._apply_sigma(p, source_mask, psd, ridge)
            pAp = np.einsum('ij,ij->j', p, Ap).astype(np.float64)
            pAp_safe = np.where(np.abs(pAp) < 1e-30, 1e-30, pAp)
            alpha = (rs_old / pAp_safe).astype(np.float32)
            x = x + alpha[None, :] * p
            r = r - alpha[None, :] * Ap
            rs_new = np.einsum('ij,ij->j', r, r).astype(np.float64)
            rel = float(np.sqrt(rs_new.max()) / b_norm_safe.max())
            last_rel = rel
            if rel < tol:
                break
            # Guard: if any channel's rs_old is numerically 0, beta
            # for that channel is 0 and the update there is effectively a no-op.
            rs_old_safe = np.where(rs_old < 1e-30, 1e-30, rs_old)
            beta = (rs_new / rs_old_safe).astype(np.float32)
            p = r + beta[None, :] * p
            rs_old = rs_new
            if debug and (it == 1 or it % 20 == 0):
                self._debug_print(
                    True,
                    f"    CG iter {it}: rel_res={rel:.3e}",
                )
        return x, it, last_rel
    #endregion

    #region C-DRIVER
    def _run_single(
        self,
        img: np.ndarray,
        msk: np.ndarray,
        cg_max_iterations: int,
        cg_tolerance: float,
        add_innovation: bool,
        innovation_strength: float,
        regularization: float,
        seed: int,
        debug_prints: bool,
        detrend_sigma: float = 0.0,
        clamp_to_source_gamut: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Stationary-Gaussian inpainting (Galerne-Leclaire 2017)
        with local-mean detrending and a source-gamut safety clamp.

        ``img``: ``(H, W, C)`` float32 in ``[0, 1]``.
        ``msk``: ``(H, W)`` float32, 1 = target, 0 = source.

        Returns ``(filled, texton_viz)`` both float32 in ``[0, 1]``.

        Pipeline (indices match the ``[step N]`` debug prints):

          0. **Detrend.** Build a smooth trend ``m`` by normalised
             convolution of ``u * M`` with a Gaussian (sigma from
             hole geometry if ``detrend_sigma <= 0``). Work on the
             zero-mean residual ``r = u - m`` from here on. This is
             the preprocessing that lets Galerne-Leclaire handle
             non-stationary natural-image content (sky + grass +
             wall) without the CG correction overshooting into a
             clipped, saturated output. Pass ``detrend_sigma < 0``
             to disable.
          1-6. **FFT-MA + CG kriging** on the residual, as
             in Galerne-Leclaire 2017.
          7. **Re-stamp + add trend back.** ``X = m + (F + c)``,
             then overwrite ``X[source] = img[source]`` bit-exact.
          8. **Source-gamut clamp** (optional). Clamp the target
             region to the per-channel ``[src_min, src_max]`` of the
             source pixels, guarding against residual model-
             mismatch producing out-of-gamut colours. Source pixels
             are not touched.

        Extensive ``debug_prints=True`` logging at every step makes
        divergence from Galerne-Leclaire's arithmetic directly visible:
        residual mean must be ~0 after detrend, PSD DC must be ~0,
        F scale must match texton energy, CG must converge, final
        output must lie in the source gamut.
        """
        H, W, C = img.shape
        source_mask = (msk < 0.5).astype(np.float32)
        src_bool = source_mask > 0.5
        n_src = int(src_bool.sum())
        n_tgt = int((~src_bool).sum())
        self._debug_print(
            debug_prints,
            f"start: image={H}x{W}x{C} source_px={n_src} target_px={n_tgt} "
            f"innovation={add_innovation} seed={seed}",
        )
        if n_src == 0 or n_tgt == 0:
            return (np.clip(img, 0.0, 1.0).astype(np.float32),
                    np.zeros((H, W, 3), dtype=np.float32))

        # ----- [step 0] detrending -------------------------------
        if detrend_sigma < 0.0:
            m = np.zeros((H, W, C), dtype=np.float32)
            used_sigma = 0.0
            self._debug_print(
                debug_prints,
                f"[step 0] detrend DISABLED (detrend_sigma < 0); pure "
                f"Galerne-Leclaire 2017 on raw u.",
            )
        else:
            used_sigma = (float(detrend_sigma) if detrend_sigma > 0.0
                          else self._auto_trend_sigma(source_mask))
            m = self._compute_smooth_trend(img, source_mask, used_sigma)
            self._debug_print(
                debug_prints,
                f"[step 0] detrend sigma={used_sigma:.1f} "
                f"m stats: {self._stats_str(m)}",
            )
        img_resid = (img.astype(np.float32) - m).astype(np.float32)
        if debug_prints:
            self._debug_print(
                True,
                f"[step 0] residual u-m (source-only) = "
                f"{self._stats_str(img_resid * source_mask[..., None])}",
            )

        # ----- [step 1] stationary-Gaussian model on residual ----
        # ``mu_resid`` should be ~ 0 per channel when detrending is
        # active - we log any non-zero value (indicating that
        # the trend is incorrectly biased, e.g. from a buggy
        # normalization) is visible.
        mu, t = self._build_texton(img_resid, source_mask)
        texton_energy_per_channel = (t * t).sum(axis=(0, 1)).astype(np.float32)
        self._debug_print(
            debug_prints,
            f"[step 1] mu_resid={np.round(mu, 4).tolist()} "
            f"texton_energy_per_ch={np.round(texton_energy_per_channel, 5).tolist()} "
            f"(= source-residual variance per channel)",
        )

        # ----- [step 2] PSD |FFT(t)|^2 ---------------------------
        T = np.fft.rfft2(t, axes=(0, 1))
        psd = (T * np.conj(T)).real.astype(np.float32)
        psd_dc = psd[0, 0].astype(np.float64)
        psd_peak = psd.reshape(-1, C).max(axis=0).astype(np.float64)
        self._debug_print(
            debug_prints,
            f"[step 2] PSD DC (expect ~0)={np.round(psd_dc, 8).tolist()} "
            f"PSD peak={np.round(psd_peak, 5).tolist()}",
        )

        # Tikhonov ridge per channel, scaled by the **spectral max**
        # (PSD peak) NOT by texton energy.
        #
        # the kriging operator Sigma_SS is diagonalised in the
        # DFT basis with eigenvalues given by the PSD ``|T|^2``. The
        # CG condition number is approximately
        # ``lambda_max / (lambda_min + ridge) = psd_peak / ridge``
        # (since lambda_min -> 0 for any finite-support texton).
        # Scaling ridge by ``texton_energy`` (the *mean* PSD value,
        # not its peak) gives condition numbers like
        # ``psd_peak / (r * psd_mean) = r^{-1} * peak/mean``, which
        # is 1e7 for multi-modal source where peak >> mean (sharp
        # colour regions produce a strong low-frequency mode). CG
        # then diverges, ``z`` blows up to O(500), the correction
        # term saturates, and the output reaches +/- 2 before the
        # final clip - trying to fix the "bright saturated" symptom.
        #
        # Scaling by ``psd_peak`` instead makes ``kappa = 1/regularization``
        # regardless of how concentrated the spectrum is, so CG
        # converges in O(50) iterations on any input.
        psd_peak_per_ch = psd.reshape(-1, C).max(axis=0).astype(np.float32)
        ridge = (float(regularization)
                 * np.maximum(psd_peak_per_ch, 1e-12))
        self._debug_print(
            debug_prints,
            f"[step 2] ridge per channel = "
            f"{np.round(ridge, 6).tolist()} "
            f"(= {float(regularization):.1e} * psd_peak); "
            f"texton_energy/psd_peak ratio = "
            f"{np.round(texton_energy_per_channel / np.maximum(psd_peak_per_ch, 1e-12), 4).tolist()} "
            f"(small ratio = concentrated spectrum, needs this scaling)",
        )

        # ----- [step 3] unconditional sample F = mu + t (*) W ----
        rng = np.random.default_rng(int(seed) & 0xFFFFFFFF)
        if add_innovation:
            white = rng.standard_normal((H, W)).astype(np.float32)
            white *= float(innovation_strength)
            W_spec = np.fft.rfft2(white)
            F = np.zeros_like(img, dtype=np.float32)
            for c in range(C):
                conv_c = np.fft.irfft2(
                    T[..., c] * W_spec, s=(H, W),
                ).astype(np.float32)
                F[..., c] = mu[c] + conv_c
            self._debug_print(
                debug_prints,
                f"[step 3] F (FFT-MA sample, innovation={innovation_strength:.2f}): "
                f"{self._stats_str(F)}",
            )
        else:
            F = np.broadcast_to(mu.reshape(1, 1, C),
                                img.shape).astype(np.float32).copy()
            self._debug_print(
                debug_prints,
                f"[step 3] F = mu_resid (kriging-mean mode, innovation OFF): "
                f"{self._stats_str(F)}",
            )

        # ----- [step 4] residual at source -----------------------
        u_resid_src = img_resid[src_bool].astype(np.float32)
        F_src = F[src_bool].astype(np.float32)
        b = u_resid_src - F_src
        if debug_prints:
            b_mean = b.mean(axis=0)
            b_std = b.std(axis=0)
            self._debug_print(
                True,
                f"[step 4] b = u_resid_S - F_S: "
                f"mean={np.round(b_mean, 4).tolist()} "
                f"std={np.round(b_std, 4).tolist()} "
                f"(expect b.mean~0, b.std~sqrt(2*texton_energy))",
            )

        # ----- [step 5] CG solve (Sigma_SS + r I) z = b ----------
        z, cg_iters, cg_rel = self._conjugate_gradient(
            b, source_mask, psd, ridge,
            max_iter=int(cg_max_iterations),
            tol=float(cg_tolerance),
            debug=debug_prints,
        )
        self._debug_print(
            debug_prints,
            f"[step 5] CG done: iters={cg_iters} rel_res={cg_rel:.3e} "
            f"z stats: mean={np.round(z.mean(axis=0), 4).tolist()} "
            f"max|z|={np.round(np.abs(z).max(axis=0), 4).tolist()}",
        )

        # ----- [step 6] correction c = Sigma_.S z ----------------
        z_full = np.zeros_like(img, dtype=np.float32)
        sy, sx = np.where(src_bool)
        z_full[sy, sx, :] = z
        correction = np.zeros_like(img, dtype=np.float32)
        for c in range(C):
            correction[..., c] = self._fft_convolve(
                psd[..., c], z_full[..., c],
            ).astype(np.float32)
        self._debug_print(
            debug_prints,
            f"[step 6] correction c=Sigma.S*z: {self._stats_str(correction)}",
        )

        # ----- [step 7] assemble output --------------------------
        X_resid = F + correction
        X = m + X_resid
        self._debug_print(
            debug_prints,
            f"[step 7] X_resid=F+c: {self._stats_str(X_resid)}",
        )
        self._debug_print(
            debug_prints,
            f"[step 7] X=m+X_resid (pre-restamp, pre-clamp): "
            f"{self._stats_str(X)}",
        )

        # Re-stamp source bit-exact (eliminates CG residual error).
        X[src_bool] = img[src_bool]

        # Diagnostic: target-region gamut / saturation fractions BEFORE any clamping / clipping.
        if debug_prints and n_tgt > 0:
            tgt = X[~src_bool]
            src_vals = img[src_bool]
            src_min_c = src_vals.min(axis=0)
            src_max_c = src_vals.max(axis=0)
            below = (tgt < src_min_c[None, :] - 1e-6)
            above = (tgt > src_max_c[None, :] + 1e-6)
            oog_any = (below | above).any(axis=-1).mean()
            sat_low = (tgt <= 0.0).any(axis=-1).mean()
            sat_high = (tgt >= 1.0).any(axis=-1).mean()
            self._debug_print(
                True,
                f"[step 7] target diagnostics BEFORE clamp/clip: "
                f"out-of-gamut={100*oog_any:.2f}% "
                f"sat<=0={100*sat_low:.2f}% sat>=1={100*sat_high:.2f}% "
                f"src_gamut_per_ch=[{src_min_c.tolist()}..{src_max_c.tolist()}] "
                f"tgt_range=[{tgt.min():.3f},{tgt.max():.3f}]",
            )

        # ----- [step 8] source-gamut safety clamp ----------------
        if clamp_to_source_gamut and n_src > 0 and n_tgt > 0:
            src_vals = img[src_bool]
            src_min_c = src_vals.min(axis=0).astype(np.float32)
            src_max_c = src_vals.max(axis=0).astype(np.float32)
            tgt_mask3 = np.broadcast_to(
                (~src_bool)[..., None], X.shape
            )
            X_tgt_clamped = np.clip(
                X, src_min_c.reshape(1, 1, C), src_max_c.reshape(1, 1, C),
            )
            n_clamped_per_ch = np.zeros(C, dtype=np.int64)
            for c in range(C):
                tgt_slice = X[..., c][~src_bool]
                clamped = (
                    (tgt_slice < src_min_c[c]) | (tgt_slice > src_max_c[c])
                )
                n_clamped_per_ch[c] = int(clamped.sum())
            X = np.where(tgt_mask3, X_tgt_clamped, X)
            self._debug_print(
                debug_prints,
                f"[step 8] source-gamut clamp applied: "
                f"clamped_per_ch={n_clamped_per_ch.tolist()} / "
                f"{n_tgt} target pixels",
            )
        elif debug_prints:
            self._debug_print(
                True, "[step 8] source-gamut clamp DISABLED",
            )

        X_final = np.clip(X, 0.0, 1.0).astype(np.float32)
        self._debug_print(
            debug_prints,
            f"[step 8] X_final (post-clip): {self._stats_str(X_final)}",
        )

        texton_viz = self._texton_visualization(img, source_mask)
        return X_final, texton_viz
    #endregion

    #region DEBUG
    @staticmethod
    def _debug_print(debug: bool, msg: str) -> None:
        # Consistent ``[ ImageInfillGaussianMixtureLayer ]`` prefix on
        # every line so debug output is easy to filter from a noisy
        # ComfyUI console.
        if debug:
            print(f"[ ImageInfillGaussianMixtureLayer ] {msg}")

    @staticmethod
    def _stats_str(arr: np.ndarray) -> str:
        """Compact per-channel ``[min, max, mean, std]`` debug summary.

        Used throughout ``_run_single`` to make every stage's
        numerical state visible when ``debug_prints=True``, so any
        divergence from Galerne-Leclaire's intended arithmetic
        (non-zero residual mean, PSD DC leak, F out of scale, CG
        non-convergence, trend NaN, etc.) is diagnosable from one
        ComfyUI console run.
        """
        a = np.asarray(arr, dtype=np.float32)
        if a.ndim == 3:
            per_ch = [
                f"ch{c}=[{a[..., c].min():+.3f},{a[..., c].max():+.3f},"
                f"mu={a[..., c].mean():+.3f},sd={a[..., c].std():.3f}]"
                for c in range(a.shape[-1])
            ]
            return " ".join(per_ch)
        return (f"[{a.min():+.3f},{a.max():+.3f},mu={a.mean():+.3f},"
                f"sd={a.std():.3f}]")
    #endregion

    #region VIZ
    @staticmethod
    def _texton_visualization(
        img: np.ndarray,
        source_mask: np.ndarray,
    ) -> np.ndarray:
        """Visualize the centered source content (proxy for the
        texton), normalized to ``[0, 1]`` for display.

        Source pixels show ``(u - mu)`` rescaled; target pixels are
        zeroed. Useful for debugging the Gaussian-model fit.
        """
        src_f = source_mask.astype(np.float32)
        n_src = float(src_f.sum())
        if n_src <= 0:
            return np.zeros_like(img, dtype=np.float32)
        mu = (img.astype(np.float32) * src_f[..., None]
              ).reshape(-1, img.shape[-1]).sum(axis=0) / n_src
        centered = (img - mu) * src_f[..., None]
        # Rescale to [0, 1] for visualization, preserving per-channel relative magnitudes.
        m = np.abs(centered).max()
        if m > 0:
            vis = 0.5 + 0.5 * centered / m
        else:
            vis = np.full_like(img, 0.5, dtype=np.float32)
        return vis.astype(np.float32)
    #endregion

    #region UI
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "mask": ("MASK",),
            },
            "optional": {
                "cg_max_iterations": ("INT", {
                    "default": 100, "min": 1, "max": 2000, "step": 1,
                    "display": "number",
                }),
                "cg_tolerance": ("FLOAT", {
                    "default": 1e-6, "min": 1e-12, "max": 1e-1,
                    "step": 1e-6, "display": "number",
                }),
                "add_innovation": ("BOOLEAN", {"default": True}),
                "innovation_strength": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 4.0, "step": 0.05,
                    "display": "number",
                }),
                "regularization": ("FLOAT", {
                    "default": 1e-3, "min": 1e-8, "max": 1.0,
                    "step": 1e-4, "display": "number",
                }),
                "seed": ("INT", {
                    "default": 0, "min": 0, "max": 0xFFFFFFFF, "step": 1,
                }),
                "detrend_sigma": ("FLOAT", {
                    "default": 0.0, "min": -1.0, "max": 256.0,
                    "step": 1.0, "display": "number",
                }),
                "clamp_to_source_gamut": ("BOOLEAN", {"default": True}),
                "debug_prints": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE")
    RETURN_NAMES = ("infilled_image", "texton_viz")
    FUNCTION = "gaussian_mixture_infill"
    CATEGORY = "illumorae"
    OUTPUT_NODE = False
    DESCRIPTION = (
        "Gaussian-field texture inpainting via FFT-based conditional "
        "simulation (Galerne-Leclaire 2017). Samples from a stationary "
        "Gaussian model fit on source pixels; preserves texture "
        "statistics and is best for homogeneous microtextures."
    )
    #endregion

    #region ENTRY
    def gaussian_mixture_infill(
        self,
        image: torch.Tensor,
        mask: torch.Tensor,
        cg_max_iterations: int = 100,
        cg_tolerance: float = 1e-6,
        add_innovation: bool = True,
        innovation_strength: float = 1.0,
        regularization: float = 1e-3,
        seed: int = 0,
        debug_prints: bool = False,
        detrend_sigma: float = 0.0,
        clamp_to_source_gamut: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """ComfyUI entry point.

        Args:
            image: ``(B, H, W, C)`` float tensor in ``[0, 1]``.
            mask:  ``(B, H, W)`` float tensor, 1 = target, 0 = source.
            cg_max_iterations: Maximum CG iterations for the kriging
                solve. Typical convergence is 20-100. Larger values
                reduce residual error at a linear cost.
            cg_tolerance: Relative residual norm at which CG stops
                early. Output is always re-stamped at source pixels
                so CG residual never leaks into the source-region
                output.
            add_innovation: If True, add the stochastic FFT-MA
                unconditional sample (Galerne-Leclaire conditional
                simulation). If False, skip the innovation and return
                the conditional mean (kriging / BLUP); the target
                region is then smoother and deterministic.
            innovation_strength: Scalar multiplier on the white-noise
                innovation. ``1.0`` is the model-faithful value; lower
                values dampen texture; larger values over-synthesize.
            regularization: Tikhonov ridge for the kriging system,
                expressed as a fraction of the per-channel PSD peak
                ``max_k |fft(t_c)|^2``. The default ``1e-3`` sets the
                CG condition number to ``1/regularization = 1000``,
                giving convergence in roughly 20-100 iterations.
                Increase (e.g., 1e-2 .. 1e-1) if you see saturated /
                color-shifted patches; decrease (e.g., 1e-4) if the
                fill is over-smoothed and you trust the source
                statistics. ``Sigma_SS`` is rank-deficient by
                construction so this term must be > 0.
            seed: RNG seed for the innovation. Batch element ``b``
                uses ``seed + b`` to give distinct samples per batch.
            debug_prints: Periodic progress prints (CG residual, etc).
            detrend_sigma: Local-mean detrending Gaussian sigma (in
                pixels). ``0.0`` (default) = auto-select from hole
                geometry via ``_auto_trend_sigma`` - recommended for
                all natural-image use. ``> 0`` uses the given sigma.
                ``< 0`` disables detrending for pure Galerne-Leclaire
                2017 behaviour (only use for homogeneous microtextures
                such as grass / fabric / sand). Without detrending,
                multi-modal source content (sky + grass + wall) causes
                the kriging correction to overshoot, producing
                wildly saturated / off-gamut fills after the final
                clip to ``[0, 1]``.
            clamp_to_source_gamut: Final safety clamp that restricts
                target-region output to each channel's ``[src_min,
                src_max]`` range (source pixels are untouched). ON by
                default; set False only to see the unclamped model
                output for diagnosis.

        Returns:
            ``(infilled_image, texton_viz)``, both ``(B, H, W, 3)``
            float32 in ``[0, 1]``. ``texton_viz`` shows the centered
            source content (proxy for the texton) for debugging.
        """
        if not _HAS_TORCH:
            raise RuntimeError(
                "torch is required for the ComfyUI entry point "
                "gaussian_mixture_infill; install torch or call "
                "_run_single directly on numpy arrays."
            )
        self._debug_print(debug_prints, f"image={tuple(image.shape)} "
                          f"mask={tuple(mask.shape)}")

        batch_size = image.shape[0]
        results = []
        viz_outs = []
        for b in range(batch_size):
            img_np = image[b].detach().cpu().numpy().astype(np.float32)
            msk_np = mask[b].detach().cpu().numpy().astype(np.float32)
            if (msk_np > 0.5).sum() == 0:
                filled = img_np.copy()
                viz = np.zeros(img_np.shape, dtype=np.float32)
            else:
                filled, viz = self._run_single(
                    img_np, msk_np,
                    cg_max_iterations=int(cg_max_iterations),
                    cg_tolerance=float(cg_tolerance),
                    add_innovation=bool(add_innovation),
                    innovation_strength=float(innovation_strength),
                    regularization=float(regularization),
                    seed=int(seed) + b,
                    debug_prints=bool(debug_prints),
                    detrend_sigma=float(detrend_sigma),
                    clamp_to_source_gamut=bool(clamp_to_source_gamut),
                )
            results.append(torch.from_numpy(filled).float())
            viz_outs.append(torch.from_numpy(viz).float())
        out_img = torch.stack(results, dim=0)
        out_viz = torch.stack(viz_outs, dim=0)
        self._debug_print(debug_prints, f"output={tuple(out_img.shape)}")
        return (out_img, out_viz)
    #endregion


# ComfyUI node registration
NODE_CLASS_MAPPINGS = {"illumoraeImageInfillGaussianMixtureLayerNode":illumoraeImageInfillGaussianMixtureLayerNode}

NODE_DISPLAY_NAME_MAPPINGS = {"illumoraeImageInfillGaussianMixtureLayerNode":"Image Infill Gaussian Mixture Layer"}
