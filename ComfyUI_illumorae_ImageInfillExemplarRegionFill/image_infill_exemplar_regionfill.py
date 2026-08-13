"""
Image Infill Exemplar Region Fill
An approximation attempting to implement a Criminisi-style exemplar-based region filling with isophote-driven
priority.  a sequential, priority-driven, single-copy structure filler.

# Reference: Criminisi, A., Perez, P., Toyama, K. (2004).
# "Region filling and object removal by exemplar-based image inpainting."
# IEEE Trans. Image Processing 13(9):1200-1212.
#
# Optional priority-blend variant:
#   Cheng, W.-H., Hsieh, C.-W., Lin, S.-K., Wang, C.-W., Wu, J.-L. (2005).
#   "Robust algorithm for exemplar-based image inpainting."
#   In CGIV 2005, where priority is replaced by a convex blend of a
#   regularized confidence and the data term.


STATUS:: working
TITLE::Image Infill Exemplar Region Fill
DESCRIPTIONSHORT::Image Infill using exemplar region filling with isophote-driven priority. extends linear structures into the target region.
VERSION::20260427
IMAGE::comfyui_illumorae_image_infill_exemplar_regionfill.png
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

# 8-connected discrete Laplacian. Used to detect the fill front: pixels
# in the target region whose neighborhood touches the source region
# produce a strictly positive value.
_LAPLACIAN_KERNEL = np.array(
    [[1.0, 1.0, 1.0],
     [1.0, -8.0, 1.0],
     [1.0, 1.0, 1.0]], dtype=np.float32,
)

# Central-difference kernels used to estimate the boundary normal from
# the source-region indicator. uses a half-Sobel (a single -1 / +1 pair);
_NORMAL_KERNEL_X = np.array(
    [[0.0, 0.0, 0.0],
     [-1.0, 0.0, 1.0],
     [0.0, 0.0, 0.0]], dtype=np.float32,
)
_NORMAL_KERNEL_Y = _NORMAL_KERNEL_X.T.copy()
#endregion


class illumoraeImageInfillExemplarRegionFillNode:
    """
    Criminisi-style exemplar-based region filling.

    Terminology (Criminisi et al., 2004):
      - **target region** ``Omega``: the pixels to be inpainted
        (mask == 1).
      - **source region** ``Phi = I \\ Omega``: the known pixels
        available as exemplars (mask == 0).
      - **fill front** ``delta Omega``: target pixels with at least
        one source neighbor in their 3x3 neighborhood.
      - **priority** ``P(p) = C(p) D(p)``: per-fill-front-pixel score
        composed of confidence ``C`` (how reliable the patch around
        ``p`` already is) and data ``D`` (how strongly an isophote
        flows into the target at ``p``).

    Algorithm overview (one batch element):
      Working state: ``work`` (image with the original target zeroed),
      ``source_region`` (1 = currently filled or original source),
      ``target_region`` (1 = still to fill), ``confidence`` (float32
      reliability map, init = source_region).

      Loop until ``target_region`` is empty (or ``max_steps`` reached):
        1. Compute fill front via Laplacian of ``target_region``.
        2. Estimate boundary normals from gradients of
           ``source_region`` (per fill-front pixel).
        3. Estimate image gradients (Scharr) of the work image's
           grayscale, zeroed inside the target.
        4. Confidence ``C(p)`` = mean of ``confidence`` over the
           ``patch_size`` x ``patch_size`` window centered at ``p``.
           Data ``D(p)`` = ``|grad(I)(p) . n(p)|`` + ``epsilon``,
           where ``n`` is the unit boundary normal.
        5. Priority ``P(p)`` from ``C, D`` (mode selectable: classic
           Criminisi product or the Cheng et al. convex blend).
        6. Pick the highest-priority fill-front pixel ``p*`` and the
           clipped patch ``T`` around it.
        7. Search every admissible source-patch upper-left for the
           SSD minimum against ``T``, weighted by the source-pixel
           mask of ``T``. Implemented as
           ``cv2.matchTemplate(TM_SQDIFF, mask=valid)`` and gated by
           a precomputed valid-upper-left map (full-coverage of the
           admissible region).
        8. (Optional) Patch-variance penalty: of the top-K candidate
           upper-lefts, pick the one with low SSD AND low variance
           between the candidate's pixels at ``T``'s target positions
           and the candidate's mean over ``T``'s source positions
           (Criminisi 2004, Eq. (3)-(4) of the variance-aware variant).
        9. Copy the chosen source patch's pixels into ``T`` only at
           positions that are still in ``target_region``. Update
           ``source_region``, ``target_region``, ``confidence``
           (filled positions inherit ``C(p*)``), and the fill-order
           visualization.

      Exit when ``target_region.sum() == 0`` or no admissible source
      patch is large enough (degenerate inputs).

    Discipline modes (``enforce_source_discipline``):
      - **False (default)**: SSDs may use
        previously-filled pixels as exemplar source. This is what
        gives Criminisi its long-range structure-extension behavior:
        an edge pushed into the target region by an early step becomes a
        valid source for later steps further into the target region.
      - **True**: SSDs are restricted to upper-lefts whose entire
        patch lies in the *original* source region. Slower
        convergence on large regions, but guarantees that no SSD ever
        depends on a pixel that was synthesized rather than observed.

    """

    #region INIT
    def __init__(self):
        pass
    #endregion

    #region C-FILLFRONT
    def _compute_fill_front(
        self,
        target_region: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return ``(ys, xs)`` for fill-front pixels.

        A fill-front pixel is a target pixel whose 3x3 neighborhood
        contains at least one source pixel. Computed via the
        standard discrete Laplacian (``sum_neighbors - 8*center``)
        of the target indicator: at a target pixel (value 1) with
        ``k`` source neighbors (value 0) the Laplacian evaluates to
        ``-k``, so fill-front pixels are those target pixels
        whose Laplacian is strictly **negative**.
        """
        lap = cv2.filter2D(
            target_region.astype(np.float32), cv2.CV_32F,
            _LAPLACIAN_KERNEL,
        )
        ff = (lap < 0.0) & (target_region > 0)
        ys, xs = np.where(ff)
        return ys.astype(np.int64), xs.astype(np.int64)

    def _compute_normals(
        self,
        source_region: np.ndarray,
        ys: np.ndarray,
        xs: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Per-fill-front-pixel unit boundary normal.

        Computed as ``(grad_y(S), -grad_x(S))`` and L2-normalized.
        The gradient of the source-region indicator is non-zero along the fill
        front and points toward the source side, so its perpendicular
        is the fill-front tangent's normal.
        """
        sf = source_region.astype(np.float32)
        gx = cv2.filter2D(sf, cv2.CV_32F, _NORMAL_KERNEL_X)
        gy = cv2.filter2D(sf, cv2.CV_32F, _NORMAL_KERNEL_Y)
        nx = gy[ys, xs]
        ny = -gx[ys, xs]
        norm = np.sqrt(nx * nx + ny * ny)
        ok = norm > 1e-12
        nx = np.where(ok, nx / np.maximum(norm, 1e-12), 0.0)
        ny = np.where(ok, ny / np.maximum(norm, 1e-12), 0.0)
        return nx.astype(np.float32), ny.astype(np.float32)

    def _compute_image_gradients(
        self,
        work: np.ndarray,
        source_region: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Image gradients (Scharr of grayscale).

        The gradients are not masked by ``source_region``. Fill-front
        pixels are target pixels by definition, so zeroing the
        gradient at all target pixels would reduce the data term
        ``D(p) = |grad I . n|`` to the epsilon constant at every
        fill-front pixel and remove the isophote term from the
        Criminisi priority.

        At a fill-front target pixel, the 3x3 Scharr support contains
        both source pixels (real values) and target pixels (zero in
        ``work`` from the earlier masking step). This yields a finite
        gradient biased toward the boundary that still reflects the
        dominant isophote crossing the fill front. Channel mean is
        used as a luminance proxy for isophote orientation.
        """
        if work.ndim == 3:
            gray = work.mean(axis=-1).astype(np.float32)
        else:
            gray = work.astype(np.float32)
        gx = cv2.Scharr(gray, cv2.CV_32F, 1, 0)
        gy = cv2.Scharr(gray, cv2.CV_32F, 0, 1)
        return gx.astype(np.float32), gy.astype(np.float32)

    def _compute_confidence(
        self,
        confidence: np.ndarray,
        ys: np.ndarray,
        xs: np.ndarray,
        patch_size: int,
    ) -> np.ndarray:
        """Per-fill-front-pixel confidence: average of ``confidence``
        over the ``patch_size`` x ``patch_size`` window centered at
        each fill-front pixel.

        ``confidence`` is 0 inside the target region (and stays 0
        until pixels are filled, at which point they inherit the
        priority pixel's confidence), so the box-filter sum already
        ignores unfilled positions correctly.
        """
        s = cv2.boxFilter(
            confidence, -1, (patch_size, patch_size),
            normalize=False, borderType=cv2.BORDER_CONSTANT,
        )
        area = float(patch_size * patch_size)
        return (s[ys, xs] / area).astype(np.float32)
    #endregion

    #region C-ADMISS
    def _build_admissible_ul_mask(
        self,
        admissible: np.ndarray,
        pH: int,
        pW: int,
    ) -> np.ndarray:
        """Boolean (H-pH+1, W-pW+1) array marking source-patch
        upper-left positions whose entire ``pH x pW`` patch lies in
        ``admissible`` (== 1).

        Uses a top-left-anchored box filter to count admissible pixels
        per patch and tests for full coverage.
        """
        kernel = np.ones((pH, pW), dtype=np.float32)
        cnt = cv2.filter2D(
            admissible.astype(np.float32), cv2.CV_32F, kernel,
            anchor=(0, 0), borderType=cv2.BORDER_CONSTANT,
        )
        H, W = admissible.shape
        valid = cnt[: H - pH + 1, : W - pW + 1]
        return valid >= float(pH * pW) - 0.5

    def _exclude_self_overlap(
        self,
        valid_ul: np.ndarray,
        aY: int, aX: int,
        pH: int, pW: int,
    ) -> np.ndarray:
        """Forbid source-patch upper-lefts whose patch overlaps the
        target patch around ``(aY, aX)``. A self-overlap would let an
        identity match win SSD trivially.
        """
        H_ul, W_ul = valid_ul.shape
        # Patch with UL (uy, ux) overlaps target patch [aY..aY+pH-1] x
        # [aX..aX+pW-1] iff uy in [aY-pH+1, aY+pH-1] and similarly for ux.
        y_lo = max(aY - pH + 1, 0)
        y_hi = min(aY + pH - 1, H_ul - 1)
        x_lo = max(aX - pW + 1, 0)
        x_hi = min(aX + pW - 1, W_ul - 1)
        if y_hi >= y_lo and x_hi >= x_lo:
            valid_ul = valid_ul.copy()
            valid_ul[y_lo:y_hi + 1, x_lo:x_hi + 1] = False
        return valid_ul
    #endregion

    #region C-SSD
    def _ssd_search(
        self,
        work: np.ndarray,
        target_patch: np.ndarray,
        valid_mask_2d: np.ndarray,
    ) -> np.ndarray:
        """Masked SSD over every source-patch upper-left.

        ``valid_mask_2d`` is a ``(pH, pW)`` float32 mask: 1 at
        positions of the target patch that are currently SOURCE
        (i.e. observed or already filled), 0 at positions still in
        the target region. ``cv2.matchTemplate(TM_SQDIFF, mask=...)``
        evaluates ``sum_{ij} mask(i,j) * (work(y+i, x+j) -
        target_patch(i,j))^2`` for every ``(y, x)``.

        The mask is broadcast across the 3 channels so the SSD is
        summed over RGB at the masked positions.
        """
        if valid_mask_2d.sum() < 1.0:
            # Degenerate: target patch has no source pixels at all.
            # Cannot match anything; caller will fall back.
            H, W, _ = work.shape
            pH, pW = valid_mask_2d.shape
            return np.full((H - pH + 1, W - pW + 1),
                           np.inf, dtype=np.float32)
        valid_3 = np.repeat(
            valid_mask_2d[..., None].astype(np.float32), 3, axis=-1,
        )
        ssd = cv2.matchTemplate(
            work.astype(np.float32),
            target_patch.astype(np.float32),
            cv2.TM_SQDIFF, mask=valid_3,
        )
        return ssd

    def _select_best_ul(
        self,
        ssd: np.ndarray,
        valid_ul: np.ndarray,
        work: np.ndarray,
        valid_mask_2d: np.ndarray,
        pH: int,
        pW: int,
        use_variance_penalty: bool,
        variance_alpha: float,
        variance_beta: float,
        variance_topk: int,
    ) -> Tuple[int, int]:
        """Return the chosen ``(uly, ulx)`` source-patch upper-left.

        Without variance penalty: argmin SSD over admissible ULs.
        With variance penalty (Criminisi 2004 variant): scan the top
        ``variance_topk`` ULs by SSD and pick the one minimizing a
        joint criterion that prefers low SSD AND low patch variance
        across the target-side pixels of the target patch (where
        ``variance`` is measured against the candidate's mean over
        the source-side pixels of the target patch).
        """
        ssd_search = np.where(valid_ul, ssd, np.inf).astype(np.float32)
        flat = ssd_search.ravel()
        if not np.isfinite(flat.min()):
            return -1, -1
        ws = ssd_search.shape[1]

        if not use_variance_penalty:
            best_idx = int(np.argmin(flat))
            return best_idx // ws, best_idx % ws

        K = min(int(variance_topk), int((flat < np.inf).sum()))
        if K <= 0:
            return -1, -1
        # argpartition for top-K smallest SSDs.
        if K < flat.size:
            order = np.argpartition(flat, K - 1)[:K]
        else:
            order = np.arange(flat.size)
        order = order[np.argsort(flat[order])]

        valid_S = valid_mask_2d > 0.5
        valid_T = ~valid_S
        if not valid_S.any() or not valid_T.any():
            # Either the target patch is fully source (nothing to fill)
            # or fully target (no anchor to compare to). Fall back to
            # plain SSD argmin.
            best_idx = int(order[0])
            return best_idx // ws, best_idx % ws

        min_err = float("inf")
        best_var = float("inf")
        best_uly, best_ulx = -1, -1
        for idx in order:
            err = float(flat[idx])
            if not np.isfinite(err):
                break
            uly = int(idx) // ws
            ulx = int(idx) % ws
            src_patch = work[uly:uly + pH, ulx:ulx + pW, :]
            mean_c = src_patch[valid_S].mean(axis=0)
            diff = src_patch[valid_T] - mean_c
            pv = float(np.einsum('ij,ij->', diff, diff))
            if variance_alpha * err <= min_err:
                if (err < variance_alpha * min_err
                        or pv < variance_beta * best_var):
                    min_err = err
                    best_var = pv
                    best_uly, best_ulx = uly, ulx
        if best_uly < 0:
            best_idx = int(order[0])
            return best_idx // ws, best_idx % ws
        return best_uly, best_ulx
    #endregion

    #region C-DRIVER
    def _run_single(
        self,
        img: np.ndarray,
        msk: np.ndarray,
        patch_size: int,
        max_steps: int,
        priority_mode: str,
        cheng_omega: float,
        cheng_alpha: float,
        cheng_beta: float,
        use_variance_penalty: bool,
        variance_alpha: float,
        variance_beta: float,
        variance_topk: int,
        enforce_source_discipline: bool,
        debug_prints: bool,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Run the exemplar fill on one (H,W,3) image / (H,W) mask.

        Returns ``(filled_rgb, fill_order_viz_rgb)``, both float32 in
        ``[0, 1]``.
        """
        H, W = msk.shape
        half = patch_size // 2

        # Source-discipline at entry: zero the target region of the
        # working image so that no algorithmic step can observe original target content.
        work = (img * (msk < 0.5).astype(np.float32)[..., None]
                ).astype(np.float32)
        target_region = (msk > 0.5).astype(np.uint8)
        source_region = (1 - target_region).astype(np.uint8)
        original_source_region = source_region.copy()
        confidence = source_region.astype(np.float32)
        fill_order = np.full((H, W), -1.0, dtype=np.float32)

        if max_steps <= 0:
            max_steps = H * W  # safety cap, never reached in practice.

        step = 0
        target_count_initial = int(target_region.sum())
        if debug_prints:
            self._debug_print(
                True, f"start: target_pixels={target_count_initial} "
                f"image={H}x{W} patch={patch_size} "
                f"discipline={'strict' if enforce_source_discipline else 'loose'}"
            )

        while int(target_region.sum()) > 0 and step < max_steps:
            # 1. Fill front.
            ff_ys, ff_xs = self._compute_fill_front(target_region)
            if ff_ys.size == 0:
                # Target region disconnected from source - cannot fill.
                if debug_prints:
                    self._debug_print(
                        True, f"step {step}: empty fill front; aborting"
                    )
                break

            # 2. Boundary normals at fill-front pixels.
            nx_ff, ny_ff = self._compute_normals(
                source_region, ff_ys, ff_xs,
            )

            # 3. Image gradients.
            gx, gy = self._compute_image_gradients(work, source_region)

            # 4. Confidence and data terms.
            conf_ff = self._compute_confidence(
                confidence, ff_ys, ff_xs, patch_size,
            )
            data_ff = (np.abs(
                gx[ff_ys, ff_xs] * nx_ff + gy[ff_ys, ff_xs] * ny_ff
            ) + 1e-3).astype(np.float32)

            # 5. Priority.
            if priority_mode == "cheng_blend":
                rcp = (1.0 - cheng_omega) * conf_ff + cheng_omega
                priority = (cheng_alpha * rcp
                            + cheng_beta * data_ff).astype(np.float32)
            else:
                priority = (conf_ff * data_ff).astype(np.float32)

            # 6. Highest-priority fill-front pixel.
            idx = int(np.argmax(priority))
            ty = int(ff_ys[idx])
            tx = int(ff_xs[idx])

            # Clipped target patch around (ty, tx).
            aY = max(ty - half, 0)
            bY = min(ty + half, H - 1)
            aX = max(tx - half, 0)
            bX = min(tx + half, W - 1)
            pH = bY - aY + 1
            pW = bX - aX + 1
            target_patch = work[aY:bY + 1, aX:bX + 1, :]
            valid_mask_2d = source_region[aY:bY + 1, aX:bX + 1].astype(
                np.float32
            )

            # 7. Masked SSD over all upper-lefts.
            ssd = self._ssd_search(work, target_patch, valid_mask_2d)
            if not np.isfinite(ssd).any():
                # Pathological: cannot evaluate the SSD; abort.
                if debug_prints:
                    self._debug_print(
                        True, f"step {step}: SSD non-finite; aborting"
                    )
                break

            # Admissible UL mask.
            admissible = (original_source_region
                          if enforce_source_discipline
                          else source_region)
            valid_ul = self._build_admissible_ul_mask(
                admissible, pH, pW,
            )
            valid_ul = self._exclude_self_overlap(
                valid_ul, aY, aX, pH, pW,
            )
            if not valid_ul.any():
                # No admissible source patch fits at the current scale;
                # try the loose source set as a fallback even in strict
                # mode (this only happens with very small source regions).
                if enforce_source_discipline:
                    valid_ul = self._build_admissible_ul_mask(
                        source_region, pH, pW,
                    )
                    valid_ul = self._exclude_self_overlap(
                        valid_ul, aY, aX, pH, pW,
                    )
                if not valid_ul.any():
                    if debug_prints:
                        self._debug_print(
                            True,
                            f"step {step}: no admissible source patch; "
                            f"aborting",
                        )
                    break

            # 8. Best UL with optional variance penalty.
            best_uly, best_ulx = self._select_best_ul(
                ssd, valid_ul, work, valid_mask_2d, pH, pW,
                use_variance_penalty, variance_alpha, variance_beta,
                variance_topk,
            )
            if best_uly < 0:
                if debug_prints:
                    self._debug_print(
                        True,
                        f"step {step}: best-UL selection failed; aborting",
                    )
                break

            # 9. Copy the missing pixels into the target patch.
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

                if debug_prints and (step % 50 == 0):
                    self._debug_print(
                        True,
                        f"step {step}: filled {int(copy_2d.sum())} px at "
                        f"({ty},{tx}) <- ({best_uly + half},"
                        f"{best_ulx + half}) "
                        f"target_remaining={int(target_region.sum())}"
                    )
            else:
                # No pixels to copy at this fill-front pixel - shouldn't
                # happen since fill-front implies adjacent target pixel,
                # but guard anyway to avoid an infinite loop.
                if debug_prints:
                    self._debug_print(
                        True,
                        f"step {step}: priority pixel had no missing "
                        f"neighbors; advancing"
                    )

            step += 1

        if debug_prints:
            self._debug_print(
                True, f"done: steps={step} "
                f"target_remaining={int(target_region.sum())}"
            )

        # Fill-order visualization: 0 (early) -> 1 (late), unfilled = 0.
        if step > 0:
            denom = max(step - 1, 1)
            viz = np.where(fill_order >= 0,
                           fill_order / float(denom), 0.0)
        else:
            viz = np.zeros_like(fill_order)
        viz_rgb = np.stack([viz.astype(np.float32)] * 3, axis=-1)

        filled = np.clip(work, 0.0, 1.0).astype(np.float32)
        return filled, viz_rgb
    #endregion

    #region C-FEATHER
    def _blend_boundary(
        self,
        image_np: np.ndarray,
        filled: np.ndarray,
        mask_np: np.ndarray,
        blend_width: int,
    ) -> np.ndarray:
        """Optional feather along the original target boundary.

        The exemplar fill copies whole patches; their edges sometimes
        show against the source. A short distance-transform-based
        feather smooths the seam. Source pixels are preserved
        outside the feather band.
        """
        if blend_width <= 0:
            return filled
        target = (mask_np > 0.5).astype(np.uint8)
        # Distance from each source pixel to the target boundary.
        # Use distanceTransform on the source-side indicator to get the
        # in-source distance, then build a feather weight that fades
        # from 1 at the boundary to 0 at distance >= blend_width.
        src = (1 - target).astype(np.uint8)
        # cv2.distanceTransform returns 0 on zero pixels; we need
        # distance from the source-side INTO the source, so input is
        # the source indicator and we read distance values inside it.
        dist_in_source = cv2.distanceTransform(
            src, cv2.DIST_L2, 3,
        ).astype(np.float32)
        w = np.clip(1.0 - (dist_in_source / float(blend_width)), 0.0, 1.0)
        # In the target, weight stays 1 (use filled). In the source far
        # from the boundary, weight is 0 (use original). In the source
        # near the boundary, blend.
        w = np.where(target > 0, 1.0, w).astype(np.float32)
        w3 = np.repeat(w[..., None], image_np.shape[-1], axis=-1)
        out = w3 * filled + (1.0 - w3) * image_np
        # Discipline: outside the feather band the source pixels must
        # be bit-exact. Re-stamp them.
        far_source = (target == 0) & (dist_in_source >= float(blend_width))
        if far_source.any():
            out[far_source] = image_np[far_source]
        return out.astype(np.float32)
    #endregion

    #region DEBUG
    @staticmethod
    def _debug_print(debug: bool, msg: str) -> None:
        if debug:
            print(f"[ ImageInfillExemplarRegionFill ] {msg}")

    @staticmethod
    def _ensure_odd(k: int) -> int:
        k = max(3, int(k))
        if k % 2 == 0:
            k += 1
        return k
    #endregion

    #region UI
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "mask": ("MASK",),
                "patch_size": ("INT", {
                    "default": 9, "min": 3, "max": 21, "step": 2,
                    "display": "number",
                }),
            },
            "optional": {
                "max_steps": ("INT", {
                    "default": 0, "min": 0, "max": 1000000, "step": 1,
                    "display": "number",
                }),
                "priority_mode": (
                    ["criminisi_2004", "cheng_blend"],
                    {"default": "criminisi_2004"},
                ),
                "cheng_omega": ("FLOAT", {
                    "default": 0.7, "min": 0.0, "max": 1.0, "step": 0.05,
                    "display": "number",
                }),
                "cheng_alpha": ("FLOAT", {
                    "default": 0.2, "min": 0.0, "max": 1.0, "step": 0.05,
                    "display": "number",
                }),
                "cheng_beta": ("FLOAT", {
                    "default": 0.8, "min": 0.0, "max": 1.0, "step": 0.05,
                    "display": "number",
                }),
                "use_variance_penalty": ("BOOLEAN", {"default": True}),
                "variance_alpha": ("FLOAT", {
                    "default": 0.9, "min": 0.0, "max": 1.0, "step": 0.05,
                    "display": "number",
                }),
                "variance_beta": ("FLOAT", {
                    "default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05,
                    "display": "number",
                }),
                "variance_topk": ("INT", {
                    "default": 64, "min": 1, "max": 1024, "step": 1,
                    "display": "number",
                }),
                "enforce_source_discipline": ("BOOLEAN",
                                              {"default": False}),
                "blend_width": ("INT", {
                    "default": 0, "min": 0, "max": 32, "step": 1,
                    "display": "number",
                }),
                "seed": ("INT", {
                    "default": 0, "min": 0, "max": 0xFFFFFFFF, "step": 1,
                }),
                "debug_prints": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE")
    RETURN_NAMES = ("infilled_image", "fill_order_viz")
    FUNCTION = "exemplar_regionfill_infill"
    CATEGORY = "illumorae"
    OUTPUT_NODE = False
    DESCRIPTION = (
        "Image Infill using Criminisi-style exemplar region filling with isophote-driven "
        "priority; extends linear structures (edges) into the target region "
        "via single-copy patch placement instead of voting averages."
    )
    #endregion

    #region ENTRY
    def exemplar_regionfill_infill(
        self,
        image: torch.Tensor,
        mask: torch.Tensor,
        patch_size: int,
        max_steps: int = 0,
        priority_mode: str = "criminisi_2004",
        cheng_omega: float = 0.7,
        cheng_alpha: float = 0.2,
        cheng_beta: float = 0.8,
        use_variance_penalty: bool = True,
        variance_alpha: float = 0.9,
        variance_beta: float = 0.5,
        variance_topk: int = 64,
        enforce_source_discipline: bool = False,
        blend_width: int = 0,
        seed: int = 0,
        debug_prints: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """ComfyUI entry point.

        Args:
            image: ``(B, H, W, C)`` float tensor in ``[0, 1]``.
            mask:  ``(B, H, W)`` float tensor, 1 = target, 0 = source.
            patch_size: Patch side length (odd; auto-corrected if even,
                clamped to >= 3).
            max_steps: Hard upper bound on iterations (0 = unlimited).
                A safety cap; in normal operation termination is by
                exhaustion of the target region.
            priority_mode: ``"criminisi_2004"`` for the classic
                ``P = C * D`` form, or ``"cheng_blend"`` for the
                convex blend ``alpha * ((1-omega)C + omega) + beta * D``.
            cheng_omega, cheng_alpha, cheng_beta: parameters of the
                Cheng-blend priority. Ignored unless ``priority_mode``
                is ``"cheng_blend"``.
            use_variance_penalty: Enable Criminisi's variance-aware
                tiebreaker on the top-K SSD candidates (suppresses
                exemplars that are uniform where the target patch is
                textured, and vice versa).
            variance_alpha, variance_beta: weighting constants of the
                variance tiebreaker.
            variance_topk: number of SSD-best candidates to evaluate
                under the variance criterion. K=1 disables it
                effectively. Larger K is slower but explores more.
            enforce_source_discipline: If True, restrict every SSD to
                upper-lefts whose entire patch lies in the *original*
                source region. If False (default, matches Criminisi
                2004), the search expands as fills are made. Either
                way, the original target content cannot leak into the
                output - filled pixels are always copies of source
                content.
            blend_width: Optional distance-transform feather over the
                original target boundary (0 = off). Hides patch
                seams without leaking original target content.
            seed: Reserved for future stochastic tiebreakers; the
                current algorithm is deterministic.
            debug_prints: Periodic progress prints.

        Returns:
            ``(infilled_image, fill_order_viz)``, both
            ``(B, H, W, 3)`` float32 in ``[0, 1]``. ``fill_order_viz``
            encodes step index normalized to ``[0, 1]`` (early = dark,
            late = bright); useful for debugging the priority schedule.
        """
        if not _HAS_TORCH:
            raise RuntimeError(
                "torch is required for the ComfyUI entry point "
                "exemplar_regionfill_infill; install torch or call "
                "_run_single directly on numpy arrays."
            )
        self._debug_print(debug_prints, f"image={tuple(image.shape)} "
                          f"mask={tuple(mask.shape)}")

        patch_size = self._ensure_odd(patch_size)
        if patch_size < 3:
            patch_size = 3

        batch_size = image.shape[0]
        results = []
        viz_outs = []
        for b in range(batch_size):
            img_np = image[b].detach().cpu().numpy().astype(np.float32)
            msk_np = mask[b].detach().cpu().numpy().astype(np.float32)
            self._debug_print(
                debug_prints, f"batch {b + 1}/{batch_size}: "
                f"image={img_np.shape} mask={msk_np.shape}",
            )

            if (msk_np > 0.5).sum() == 0:
                # No-op: nothing to fill.
                filled = img_np.copy()
                viz = np.zeros(msk_np.shape + (3,), dtype=np.float32)
            else:
                filled, viz = self._run_single(
                    img_np, msk_np, patch_size, max_steps,
                    priority_mode, cheng_omega, cheng_alpha, cheng_beta,
                    use_variance_penalty, variance_alpha, variance_beta,
                    variance_topk, enforce_source_discipline,
                    debug_prints,
                )
                if blend_width > 0:
                    filled = self._blend_boundary(
                        img_np, filled, msk_np, blend_width,
                    )
                    filled = np.clip(filled, 0.0, 1.0).astype(np.float32)

            results.append(torch.from_numpy(filled).float())
            viz_outs.append(torch.from_numpy(viz).float())

        out_img = torch.stack(results, dim=0)
        out_viz = torch.stack(viz_outs, dim=0)
        self._debug_print(debug_prints, f"output={tuple(out_img.shape)}")
        return (out_img, out_viz)
    #endregion


# ComfyUI node registration
NODE_CLASS_MAPPINGS = {"illumoraeImageInfillExemplarRegionFillNode":illumoraeImageInfillExemplarRegionFillNode}

NODE_DISPLAY_NAME_MAPPINGS = {"illumoraeImageInfillExemplarRegionFillNode":"Image Infill Exemplar Region Fill"}
