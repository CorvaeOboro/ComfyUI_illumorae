"""
Image Infill PatchMatch
a rough approximation of the PatchMatch algorithm
finding nearest neighbor patches to fill a masked area ,
usage example = for the removal of objects from image then using the infilled image as guidance for inpainting

STATUS:: working
TITLE::Image Infill PatchMatch
DESCRIPTIONSHORT::Approximate PatchMatch-style content fill to infill masked regions; useful as inpainting guidance.
VERSION::20260425
IMAGE::comfyui_illumorae_image_infill_patchmatch.png
GROUP::Image
"""
#region IMPORTS
from __future__ import annotations

import torch
import numpy as np
import cv2
from typing import Tuple
#endregion

class illumoraeImageInfillPatchMatchNode:
    """
    PatchMatch-style content-aware fill.
    Given an input image and a binary mask (1 = target, 0 = source),
    this node reconstructs the target region using approximate
    nearest-neighbor patches drawn from the source region.

    NOTES = 
    Terminology (Criminisi et al., 2004):
      - **target region**  : the pixels to be inpainted (mask == 1).
        Often denoted ``Omega`` in the inpainting literature.
      - **source region**  : the known pixels available as exemplars
        (mask == 0). Denoted ``Phi = I \ Omega``.
      - **mask**           : the user-supplied indicator array that
        defines which pixels are target vs source. For clear wording, "mask"
        in this file always refers to that input array (or to a
        local boolean derived from it), never to the target region
        itself.

    Source-discipline invariant (strict):
      NO original pixel content from inside the target region is ever
      read by the algorithm. Every stage - patch distance, pyramid
      downsample, voting reconstruction, boundary feather - consults
      only source pixels. Concretely:
        - The working image has its target region zeroed at entry.
        - Patch SSD is computed only on positions that are in the source
          region in BOTH patches.
        - Pyramid downsampling uses a weighted Gaussian that normalizes
          by source-region support, so coarse "source" pixels contain no
          leaked target content.
        - Voting reconstruction accumulates only from source pixels.
        - Boundary feather blends against the nearest source pixel.
      Source pixels pass through to the output unchanged.

    Pipeline per image:
      1. Optional coarse-to-fine Gaussian pyramid (multiscale_levels >= 2).
         Offsets are solved at the coarsest level, upsampled, and refined
         at each finer level. This captures large-scale structure before
         detail.
      2. Random-offset initialization with rejection sampling, so every
         initial offset lands in the source region rather than inside the
         target region.
      3. Iterations of:
           a. Forward propagation  - checks left and top neighbors.
           b. Backward propagation - checks right and bottom neighbors.
           c. Random search with exponentially decreasing radius.
      4. Patch-voting reconstruction: each target pixel is the average
         of all overlapping patch contributions from source pixels,
         producing smoother results than single-source copying.
      5. Optional feather blending against the nearest source pixel to
         hide the target boundary; width controlled by blend_width
         (0 = no blending).

    Toggleable patch-selection priors (added to the SSD distance):
      These are addressed by independently visible ComfyUI inputs and
      are recomputed at each pyramid level so they steer matching at
      every scale.

      - boundary-color prior (``use_boundary_color_prior``):
        For every query target pixel ``(y1, x1)`` the expected color
        is sampled from the nearest source pixel. ``_patch_distance``
        adds ``w_color * ||candidate_center_color - expected||^2`` to
        the SSD, biasing matches toward patches whose color blends
        with the immediately-adjacent source. This addresses cases
        where pure SSD picks a same-shape but wrong-tone patch and
        leaves a visible color seam at the target boundary.

      - local-contrast prior (``use_local_contrast_prior``):
        A per-pixel local-contrast map is computed as the windowed
        standard deviation of luminance over a square neighborhood,
        evaluated over source pixels only via a source-weighted box
        filter. This statistic is the unnormalized RMS contrast of
        the local window (cf. Peli, "Contrast in complex images",
        JOSA A, 1990) and is a standard low-level descriptor of
        local textural energy / detail density. The expected local
        contrast at a query target pixel is sampled at its
        nearest-source neighbor (so the target side reads from the
        boundary-side source content, not the zero-padded interior).
        ``_patch_distance`` adds
        ``w_contrast * (contrast(candidate) - expected_contrast(query))^2``
        to the SSD. The intent is to suppress mismatches in textural
        regime - e.g. a high-contrast detail patch dropped into a
        flat region (which produces the salt-and-pepper artifact),
        or conversely a uniform patch chosen for a visibly textured
        region.

      Both priors read ONLY source pixel content (the local-contrast
      map is source-weighted; the expected-color map indexes through
      ``_nearest_source_lookup`` which by construction lands on
      source pixels), so the source-discipline invariant is preserved
      whether the priors are enabled or not.
    """

    #region INIT
    def __init__(self):
        pass
    #endregion

    #region C-DIST
    def _patch_distance(
        self,
        image_np: np.ndarray,
        mask_np: np.ndarray,
        y1: int, x1: int,
        y2: int, x2: int,
        half: int,
        priors: dict | None = None,
    ) -> float:
        """Vectorized SSD between two patches, plus optional priors.

        Source-discipline invariant: the distance is computed using
        **only** pixel positions that are in the source region in BOTH
        patches. This is critical for correctness when the query patch
        (centered at a target pixel) overlaps the target region -
        without the query-side mask, original target content would leak
        into the similarity score and bias offset selection, effectively
        using the ground-truth content we are trying to reconstruct.

        Returns +inf if no co-source position exists (e.g. a query patch
        entirely inside the target region), in which case propagation
        / random search will simply retain the current best offset.

        Priors (optional, opt-in via :meth:`_build_priors`):
          ``priors`` is either ``None`` (= base SSD only) or a dict
          with keys ``expected_color`` (H,W,C),
          ``contrast_source`` (H,W), ``contrast_target`` (H,W),
          ``w_color`` (float), ``w_contrast`` (float). Both penalty
          terms read only source-derived quantities
          (``image_np[y2, x2]`` is a source pixel because callers
          gate candidate offsets on ``mask_np < 0.5``;
          ``expected_color`` and ``contrast_target`` are sampled
          through the nearest-source lookup), so enabling priors
          does not violate source-discipline.
        """
        h, w, _ = image_np.shape
        dy_lo = -min(half, y1, y2)
        dy_hi = min(half, h - 1 - y1, h - 1 - y2)
        dx_lo = -min(half, x1, x2)
        dx_hi = min(half, w - 1 - x1, w - 1 - x2)
        if dy_hi < dy_lo or dx_hi < dx_lo:
            return float("inf")
        a = image_np[y1 + dy_lo:y1 + dy_hi + 1,
                     x1 + dx_lo:x1 + dx_hi + 1]
        b = image_np[y2 + dy_lo:y2 + dy_hi + 1,
                     x2 + dx_lo:x2 + dx_hi + 1]
        ma = mask_np[y1 + dy_lo:y1 + dy_hi + 1,
                     x1 + dx_lo:x1 + dx_hi + 1]
        mb = mask_np[y2 + dy_lo:y2 + dy_hi + 1,
                     x2 + dx_lo:x2 + dx_hi + 1]
        valid = (ma < 0.5) & (mb < 0.5)
        if not valid.any():
            return float("inf")
        diff = ((a - b) ** 2).sum(axis=-1)
        base = float((diff * valid).sum() / (valid.sum() + 1e-8))
        if priors is None:
            return base
        extra = 0.0
        wc = priors.get("w_color", 0.0)
        if wc > 0.0:
            ec = priors["expected_color"][y1, x1]
            sc = image_np[y2, x2]
            extra += wc * float(((ec - sc) ** 2).sum())
        wlc = priors.get("w_contrast", 0.0)
        if wlc > 0.0:
            lc_query = float(priors["contrast_target"][y1, x1])
            lc_cand = float(priors["contrast_source"][y2, x2])
            d = lc_query - lc_cand
            extra += wlc * d * d
        return base + extra
    #endregion

    #region C-PYRAMID
    def _source_aware_pyrdown(
        self,
        img: np.ndarray,
        mask: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Source-aware pyramid downsample.

        A standard cv2.pyrDown applies a 5x5 Gaussian before subsampling,
        which blends target pixels into nearby coarse pixels - polluting
        the source region at every coarser level. To prevent this we
        compute a weighted Gaussian average that uses **only** source
        pixels:

            small_img[p] = pyrDown(img * w_src)[p] / pyrDown(w_src)[p]

        where ``w_src`` = 1 in the source region, 0 in the target region.
        The coarse mask marks any coarse pixel whose source support is
        below 50% as target, so boundary-contaminated pixels are
        conservatively excluded from the source region at coarser levels.
        """
        src_weight = (mask < 0.5).astype(np.float32)
        img_w = img * src_weight[..., None]
        img_blur = cv2.pyrDown(img_w)
        weight_blur = cv2.pyrDown(src_weight)
        denom = np.maximum(weight_blur, 1e-6)[..., None]
        small_img = (img_blur / denom).astype(np.float32)
        small_mask = (weight_blur < 0.5).astype(np.float32)
        return small_img, small_mask
    #endregion

    #region C-SOLVE
    def _initialize_offsets(
        self,
        mask_np: np.ndarray,
        search_radius: int,
        rng: np.random.Generator,
        max_tries: int = 16,
    ) -> np.ndarray:
        """Random offset map with rejection sampling: each initial offset
        is guaranteed (when feasible) to point into the source region."""
        h, w = mask_np.shape
        offset_map = np.zeros((h, w, 2), dtype=np.int32)
        is_target = mask_np > 0.5
        # Fallback: nearest source pixel (used if rejection sampling fails).
        is_source = (~is_target).astype(np.uint8)
        if is_source.any():
            _, labels = cv2.distanceTransformWithLabels(
                is_source, cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_PIXEL,
            )
            src_ys, src_xs = np.where(is_source == 1)
            lut = np.zeros((int(labels.max()) + 1, 2), dtype=np.int32)
            lut[labels[src_ys, src_xs], 0] = src_ys
            lut[labels[src_ys, src_xs], 1] = src_xs
        else:
            labels = None
            lut = None

        ys, xs = np.where(is_target)
        for y, x in zip(ys, xs):
            sy = sx = -1
            for _ in range(max_tries):
                dy = int(rng.integers(-search_radius, search_radius + 1))
                dx = int(rng.integers(-search_radius, search_radius + 1))
                sy = int(np.clip(y + dy, 0, h - 1))
                sx = int(np.clip(x + dx, 0, w - 1))
                if mask_np[sy, sx] < 0.5:
                    break
            else:
                if lut is not None:
                    ly, lx = lut[int(labels[y, x])]
                    sy, sx = int(ly), int(lx)
            offset_map[y, x] = (sy - y, sx - x)
        return offset_map

    def _propagate(
        self,
        image_np: np.ndarray,
        mask_np: np.ndarray,
        offset_map: np.ndarray,
        half: int,
        reverse: bool,
        priors: dict | None = None,
    ) -> np.ndarray:
        """Propagation pass over the target region. Forward scan checks
        left/top neighbors; backward scan correctly checks right/bottom
        neighbors. ``priors`` is forwarded to :meth:`_patch_distance`."""
        h, w = mask_np.shape
        is_target = mask_np > 0.5
        out = offset_map.copy()
        if reverse:
            y_iter = range(h - 1, -1, -1)
            x_iter = range(w - 1, -1, -1)
            neigh = ((0, 1), (1, 0))   # right, bottom
        else:
            y_iter = range(h)
            x_iter = range(w)
            neigh = ((0, -1), (-1, 0))  # left, top

        for y in y_iter:
            for x in x_iter:
                if not is_target[y, x]:
                    continue
                cur = out[y, x]
                sy, sx = y + int(cur[0]), x + int(cur[1])
                best_d = self._patch_distance(
                    image_np, mask_np, y, x, sy, sx, half, priors=priors,
                )
                best = cur
                for dy, dx in neigh:
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < h and 0 <= nx < w and is_target[ny, nx]:
                        no = out[ny, nx]
                        sy2, sx2 = y + int(no[0]), x + int(no[1])
                        if (0 <= sy2 < h and 0 <= sx2 < w
                                and mask_np[sy2, sx2] < 0.5):
                            d = self._patch_distance(
                                image_np, mask_np, y, x, sy2, sx2, half,
                                priors=priors,
                            )
                            if d < best_d:
                                best_d = d
                                best = no
                out[y, x] = best
        return out

    def _random_search(
        self,
        image_np: np.ndarray,
        mask_np: np.ndarray,
        offset_map: np.ndarray,
        half: int,
        search_radius: int,
        rng: np.random.Generator,
        priors: dict | None = None,
    ) -> np.ndarray:
        """Random search with exponentially decreasing radius around the
        current best offset. Only accepts candidates that point into the
        source region. ``priors`` is forwarded to
        :meth:`_patch_distance`."""
        h, w = mask_np.shape
        is_target = mask_np > 0.5
        out = offset_map.copy()
        ys, xs = np.where(is_target)
        for y, x in zip(ys, xs):
            cur = out[y, x]
            sy, sx = y + int(cur[0]), x + int(cur[1])
            best_d = self._patch_distance(
                image_np, mask_np, y, x, sy, sx, half, priors=priors,
            )
            best = cur
            radius = search_radius
            while radius >= 1:
                dy = int(rng.integers(-radius, radius + 1))
                dx = int(rng.integers(-radius, radius + 1))
                sy2 = int(np.clip(y + int(cur[0]) + dy, 0, h - 1))
                sx2 = int(np.clip(x + int(cur[1]) + dx, 0, w - 1))
                if mask_np[sy2, sx2] < 0.5:
                    d = self._patch_distance(
                        image_np, mask_np, y, x, sy2, sx2, half,
                        priors=priors,
                    )
                    if d < best_d:
                        best_d = d
                        best = np.array([sy2 - y, sx2 - x], dtype=np.int32)
                radius //= 2
            out[y, x] = best
        return out
    #endregion

    #region C-PRIORS
    def _nearest_source_lookup(
        self,
        mask_np: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Per-pixel nearest-source-pixel lookup tables.

        Returns ``(sy_map, sx_map)`` of shape ``(H, W)`` where for any
        pixel ``(y, x)`` the value ``(sy_map[y, x], sx_map[y, x])`` is
        the coordinate of the nearest pixel in the SOURCE region
        (mask < 0.5).

        Implementation note: the OpenCV API ``distanceTransformWithLabels``
        assigns a unique label to each ZERO pixel of its input and, at
        every other pixel, returns the label of the nearest such zero.
        We therefore feed it ``mask_u8`` directly (0 at source, 255 in
        target region) so each *source* pixel gets a unique label and
        every pixel in the image (target or source) reports the label of
        its nearest source pixel. A small label->coord LUT then converts
        that label into the actual ``(sy, sx)`` coordinate.

        An earlier version of this lookup fed the inverted mask, which
        instead labels *target* pixels - making the resulting LUT
        unreliable (boundary target pixels whose label was not claimed
        by any source pixel during the iteration silently fell back to
        the default coordinate ``(0, 0)``, i.e. the image's top-left
        corner, which is often dark / brownish in real photos and showed
        up as a dark band along the target boundary in the blended
        output).
        """
        h, w = mask_np.shape
        mask_u8 = (mask_np > 0.5).astype(np.uint8) * 255
        src_y, src_x = np.where(mask_u8 == 0)
        if src_y.size == 0:
            # No source at all - degenerate: every pixel maps to itself.
            yy = np.broadcast_to(np.arange(h)[:, None], (h, w)).astype(np.int32)
            xx = np.broadcast_to(np.arange(w)[None, :], (h, w)).astype(np.int32)
            return yy.copy(), xx.copy()
        _, labels = cv2.distanceTransformWithLabels(
            mask_u8, cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_PIXEL,
        )
        # Label IDs start at 1; size lut to cover labels.max().
        lut = np.zeros((int(labels.max()) + 1, 2), dtype=np.int32)
        lut[labels[src_y, src_x], 0] = src_y
        lut[labels[src_y, src_x], 1] = src_x
        sy_map = lut[labels, 0]
        sx_map = lut[labels, 1]
        return sy_map, sx_map

    def _compute_expected_color_map(
        self,
        image_np: np.ndarray,
        mask_np: np.ndarray,
    ) -> np.ndarray:
        """Per-pixel expected-color map for the boundary-color prior.

        For every pixel ``(y, x)`` (target or source) the value is the
        color of the NEAREST source pixel. At source positions this is
        the pixel's own color; at target positions this is the color
        of the closest source pixel, i.e. a piecewise-constant
        extrapolation of the source content into the target region.

        Source discipline: the lookup goes through
        :meth:`_nearest_source_lookup` which by construction lands on
        source pixels, and ``image_np`` is expected to have its target
        region zeroed by the caller, so even an erroneous lookup
        cannot return original target content.
        """
        sy_map, sx_map = self._nearest_source_lookup(mask_np)
        return image_np[sy_map, sx_map].astype(np.float32)

    def _compute_local_contrast_map(
        self,
        image_np: np.ndarray,
        mask_np: np.ndarray,
        window: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Per-pixel local contrast (windowed luminance std-dev)
        computed from SOURCE pixels only, plus the corresponding
        target-side expected-local-contrast map.

        The local-contrast statistic is the windowed standard
        deviation of per-pixel luminance - the unnormalized RMS
        contrast of the local window (Peli, "Contrast in complex
        images", JOSA A, 1990). Each window is normalized by its
        source-pixel support so that target pixels inside the
        window do not bias the estimate. Concretely::

            w_box(p)  = sum_{q in W(p)} src_w(q)
            mu(p)     = sum_{q in W(p)} L(q) src_w(q) / w_box(p)
            mu2(p)    = sum_{q in W(p)} L(q)^2 src_w(q) / w_box(p)
            sigma(p)  = sqrt(max(mu2(p) - mu(p)^2, 0))

        where ``L`` is per-pixel luminance, ``src_w`` is 1 in the
        source region / 0 in the target region, and ``W(p)`` is the
        ``window`` x ``window`` square neighborhood centered at
        ``p``. ``contrast_source`` is well-defined wherever there is
        at least one source pixel in the window.

        ``contrast_target`` is the source-side local contrast
        sampled at each pixel's nearest-source neighbor (so target
        pixels see the boundary-side local contrast, not the
        zero-padded interior).

        Returns ``(contrast_source, contrast_target)`` both shape
        ``(H, W)``, float32.
        """
        k = max(3, int(window))
        if k % 2 == 0:
            k += 1
        src_w = (mask_np < 0.5).astype(np.float32)
        # Cheap luminance: channel mean. Sufficient for the variance
        # ordering used by the prior; perceptual luminance weights
        # would only rescale the statistic and are not needed here.
        if image_np.ndim == 3:
            lum = image_np.mean(axis=-1).astype(np.float32)
        else:
            lum = image_np.astype(np.float32)
        ksize = (k, k)
        w_box = cv2.boxFilter(src_w, -1, ksize, normalize=False)
        denom = np.maximum(w_box, 1e-6)
        mu = cv2.boxFilter(lum * src_w, -1, ksize, normalize=False) / denom
        mu2 = cv2.boxFilter(lum * lum * src_w, -1, ksize, normalize=False) / denom
        var = np.maximum(mu2 - mu * mu, 0.0)
        contrast_source = np.sqrt(var).astype(np.float32)
        # Target-side expected local contrast: nearest-source lookup
        # so that target-interior pixels (where the windowed source
        # support may be small or zero) see the boundary-side local
        # contrast instead.
        sy_map, sx_map = self._nearest_source_lookup(mask_np)
        contrast_target = contrast_source[sy_map, sx_map].astype(np.float32)
        return contrast_source, contrast_target

    def _build_priors(
        self,
        image_np: np.ndarray,
        mask_np: np.ndarray,
        use_color: bool,
        w_color: float,
        use_contrast: bool,
        w_contrast: float,
        contrast_window: int,
    ) -> dict | None:
        """Build the priors dict consumed by :meth:`_patch_distance`.

        Returns ``None`` if both priors are disabled (or have zero
        weight), so the matching path is bit-exactly the legacy SSD.
        Otherwise returns a dict with all maps and weights pre-computed
        for the given level.
        """
        wc = float(w_color) if use_color else 0.0
        wlc = float(w_contrast) if use_contrast else 0.0
        if wc <= 0.0 and wlc <= 0.0:
            return None
        priors: dict = {"w_color": wc, "w_contrast": wlc}
        if wc > 0.0:
            priors["expected_color"] = self._compute_expected_color_map(
                image_np, mask_np
            )
        else:
            # Sentinel: zero-weight branch never reads it, but keep the
            # key present so consumers can index unconditionally.
            priors["expected_color"] = np.zeros_like(image_np, dtype=np.float32)
        if wlc > 0.0:
            cs, ct = self._compute_local_contrast_map(
                image_np, mask_np, contrast_window
            )
            priors["contrast_source"] = cs
            priors["contrast_target"] = ct
        else:
            zero = np.zeros(mask_np.shape, dtype=np.float32)
            priors["contrast_source"] = zero
            priors["contrast_target"] = zero
        return priors
    #endregion

    #region C-RECON
    def _reconstruct_voting(
        self,
        image_np: np.ndarray,
        mask_np: np.ndarray,
        offset_map: np.ndarray,
        half: int,
    ) -> np.ndarray:
        """Gaussian-weighted patch voting reconstruction.

        Each target pixel votes its matched source patch onto its
        ``patch_size x patch_size`` neighborhood. Each contribution is
        weighted by a 2D Gaussian centered at the voter so that, at an
        accumulator pixel ``P``, the per-voter weight falls off with the
        offset ``P - V``: voters near ``P`` dominate, voters far away
        contribute very little.

        This is a noticeable quality improvement over pure unweighted
        averaging at the **boundary** of the target region. Boundary
        target pixels receive votes from many nearby target pixels whose
        offsets often point to *different* source patches; averaging
        those equally produces a muddy / desaturated (often brownish)
        color. With Gaussian weighting the boundary pixel is dominated
        by the source pixel at its own matched offset (V = P case has
        full weight), giving a clean color that closely tracks the
        immediately-adjacent source region.

        Target pixels that received no source contribution at all fall
        back to the nearest source pixel via
        :meth:`_nearest_source_lookup`.

        Source-discipline: ``image_np`` is expected to have its target
        region zeroed by the caller, and ``in_source`` is computed from
        ``mask_np`` so that target-side positions of any candidate patch
        are excluded from the accumulation.
        """
        h, w, c = image_np.shape
        is_target = mask_np > 0.5
        accum = np.zeros((h, w, c), dtype=np.float32)
        wsum = np.zeros((h, w), dtype=np.float32)

        # Pre-compute a 2D Gaussian over the full (2*half+1, 2*half+1)
        # patch domain. Sigma = half/2 keeps weights non-trivial across
        # the whole patch while still strongly biasing toward the center.
        coords = np.arange(-half, half + 1, dtype=np.float32)
        yy, xx = np.meshgrid(coords, coords, indexing="ij")
        sigma = max(float(half) / 2.0, 0.5)
        g_full = np.exp(-(yy * yy + xx * xx) /
                        (2.0 * sigma * sigma)).astype(np.float32)

        ys, xs = np.where(is_target)
        for y, x in zip(ys, xs):
            o = offset_map[y, x]
            sy, sx = y + int(o[0]), x + int(o[1])
            dy_lo = -min(half, y, sy)
            dy_hi = min(half, h - 1 - y, h - 1 - sy)
            dx_lo = -min(half, x, sx)
            dx_hi = min(half, w - 1 - x, w - 1 - sx)
            if dy_hi < dy_lo or dx_hi < dx_lo:
                continue
            patch = image_np[sy + dy_lo:sy + dy_hi + 1,
                             sx + dx_lo:sx + dx_hi + 1]
            msrc = mask_np[sy + dy_lo:sy + dy_hi + 1,
                           sx + dx_lo:sx + dx_hi + 1]
            in_source = (msrc < 0.5).astype(np.float32)
            gk = g_full[half + dy_lo:half + dy_hi + 1,
                        half + dx_lo:half + dx_hi + 1]
            w_arr = gk * in_source                     # (ph, pw)
            accum[y + dy_lo:y + dy_hi + 1,
                  x + dx_lo:x + dx_hi + 1] += patch * w_arr[..., None]
            wsum[y + dy_lo:y + dy_hi + 1,
                 x + dx_lo:x + dx_hi + 1] += w_arr

        out = image_np.copy()
        voted = (wsum > 1e-6) & is_target
        if voted.any():
            out[voted] = accum[voted] / wsum[voted][..., None]

        # Fall back to the nearest source pixel for any target pixel
        # that received no source contribution at all (rare; can happen
        # for very small isolated target components).
        remaining = is_target & (wsum <= 1e-6)
        if remaining.any():
            sy_map, sx_map = self._nearest_source_lookup(mask_np)
            ry, rx = np.where(remaining)
            out[ry, rx] = image_np[sy_map[ry, rx], sx_map[ry, rx]]
        return out

    def _blend_boundary(
        self,
        original: np.ndarray,
        filled: np.ndarray,
        mask_np: np.ndarray,
        blend_width: int,
    ) -> np.ndarray:
        """Smoothstep feather toward the nearest source pixel.

        Within a band of width ``blend_width`` measured from the target
        boundary inward, the reconstruction is blended toward the
        nearest source-pixel color so that the seam between source and
        infill is hidden. Pixels deeper than ``blend_width`` inside the
        target region are unaffected (alpha = 1, output = ``filled``).

        The transition curve is a cubic smoothstep
        ``alpha = t^2 * (3 - 2 t)``, which is C^1-continuous at both
        endpoints and avoids the visible seams that a linear ramp leaves
        at the inner edge of the band.

        The "nearest source pixel" map is built via
        :meth:`_nearest_source_lookup`, which is the correct OpenCV-API
        construction (labels assigned to source pixels) and avoids the
        dark-corner fallback artifact described in that helper's
        docstring.
        """
        if blend_width <= 0:
            return filled
        mask_u8 = (mask_np > 0.5).astype(np.uint8) * 255
        if mask_u8.sum() == 0 or (mask_u8 == 0).sum() == 0:
            return filled

        dist_inside = cv2.distanceTransform(mask_u8, cv2.DIST_L2, 5)
        sy_map, sx_map = self._nearest_source_lookup(mask_np)
        base = original[sy_map, sx_map]

        # Cubic smoothstep falloff: t=0 at the target boundary -> alpha=0
        # (pure base / nearest-source); t=1 at depth blend_width -> alpha=1
        # (pure filled). C^1 at both endpoints.
        t = np.clip(dist_inside / float(blend_width), 0.0, 1.0)
        alpha = (t * t * (3.0 - 2.0 * t))[..., None].astype(np.float32)

        out = filled.copy()
        is_target = mask_np > 0.5
        out[is_target] = (alpha[is_target] * filled[is_target]
                          + (1.0 - alpha[is_target]) * base[is_target])
        return out
    #endregion

    #region C-DRIVER
    def _run_single_level(
        self,
        image_np: np.ndarray,
        mask_np: np.ndarray,
        patch_size: int,
        iterations: int,
        search_radius: int,
        init_offsets: np.ndarray,
        rng: np.random.Generator,
        debug_prints: bool,
        priors: dict | None = None,
    ) -> np.ndarray:
        """Run PatchMatch (init + iterations of prop+prop+search) at a
        single pyramid level. ``init_offsets`` may be None to start
        from random. ``priors``, when provided, is forwarded to all
        distance evaluations within this level."""
        half = patch_size // 2
        offsets = init_offsets
        if offsets is None:
            offsets = self._initialize_offsets(mask_np, search_radius, rng)
        for it in range(iterations):
            offsets = self._propagate(
                image_np, mask_np, offsets, half, reverse=False,
                priors=priors,
            )
            offsets = self._propagate(
                image_np, mask_np, offsets, half, reverse=True,
                priors=priors,
            )
            offsets = self._random_search(
                image_np, mask_np, offsets, half, search_radius, rng,
                priors=priors,
            )
            self._debug_print(
                debug_prints,
                f"    iter {it + 1}/{iterations} "
                f"(r={search_radius}, size={mask_np.shape[1]}x{mask_np.shape[0]})"
            )
        return offsets
    #endregion

    #region DEBUG
    def _debug_print(self, debug_prints, *args, **kwargs):
        if debug_prints:
            print("[ ImageInfillPatchMatch ]", *args, **kwargs)
    #endregion

    #region UI
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),  # (B, H, W, C)
                "mask": ("MASK",),    # (B, H, W), 1 = target, 0 = source
                "patch_size": ("INT", {
                    "default": 7, "min": 3, "max": 31, "step": 2,
                    "display": "number",
                }),
                "iterations": ("INT", {
                    "default": 5, "min": 1, "max": 20, "step": 1,
                    "display": "number",
                }),
                "search_radius": ("INT", {
                    "default": 50, "min": 10, "max": 500, "step": 10,
                    "display": "number",
                }),
                "blend_width": ("INT", {
                    "default": 12, "min": 0, "max": 50, "step": 1,
                    "display": "number",
                }),
            },
            "optional": {
                "multiscale_levels": ("INT", {
                    "default": 2, "min": 1, "max": 5, "step": 1,
                    "display": "number",
                }),
                "use_boundary_color_prior": ("BOOLEAN", {"default": True}),
                "boundary_color_weight": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 10.0, "step": 0.1,
                    "display": "number",
                }),
                "use_local_contrast_prior": ("BOOLEAN", {"default": True}),
                "local_contrast_weight": ("FLOAT", {
                    "default": 2.0, "min": 0.0, "max": 50.0, "step": 0.1,
                    "display": "number",
                }),
                "local_contrast_window": ("INT", {
                    "default": 5, "min": 3, "max": 21, "step": 2,
                    "display": "number",
                }),
                "seed": ("INT", {
                    "default": 0, "min": 0, "max": 0xFFFFFFFF, "step": 1,
                }),
                "debug_prints": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE")
    RETURN_NAMES = ("infilled_image", "offset_map_viz")
    FUNCTION = "patchmatch_infill"
    CATEGORY = "illumorae"
    OUTPUT_NODE = False
    DESCRIPTION = "Approximate PatchMatch-style content fill to infill masked regions; useful as inpainting guidance."
    #endregion

    #region ENTRY
    def patchmatch_infill(
        self,
        image: torch.Tensor,
        mask: torch.Tensor,
        patch_size: int,
        iterations: int,
        search_radius: int,
        blend_width: int,
        multiscale_levels: int = 2,
        use_boundary_color_prior: bool = True,
        boundary_color_weight: float = 1.0,
        use_local_contrast_prior: bool = True,
        local_contrast_weight: float = 2.0,
        local_contrast_window: int = 5,
        seed: int = 0,
        debug_prints: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            image: (B, H, W, C) float tensor in [0, 1].
            mask:  (B, H, W) float tensor, 1 = target, 0 = source.
            patch_size: Patch side length (odd; auto-corrected if even).
            iterations: PatchMatch iterations per pyramid level.
            search_radius: Random-search radius at the finest level.
            blend_width: Target-boundary feather width. 0 disables
                feathering.
            multiscale_levels: Coarse-to-fine pyramid levels (>=1).
            use_boundary_color_prior: Bias matches toward patches whose
                center color matches the nearest-source-pixel color of
                the query target pixel. See class docstring for the
                full description.
            boundary_color_weight: Weight on the boundary-color penalty
                term added to the SSD distance. Same scale as the SSD
                (sum-of-squared-RGB-difference); 1.0 makes the two
                terms roughly comparable in magnitude.
            use_local_contrast_prior: Bias matches toward patches
                whose local contrast (windowed luminance std-dev,
                i.e. the unnormalized RMS contrast of the window;
                Peli 1990) matches the local contrast expected at
                the query target pixel (sampled from its
                nearest-source neighbor). Suppresses salt-and-pepper
                artifacts in flat regions and over-smoothing in
                textured regions by enforcing textural-regime
                coherence between query and candidate.
            local_contrast_weight: Weight on the local-contrast
                penalty term. The std-dev of luminance is in
                ``[0, ~0.5]`` for natural images in [0, 1], so the
                squared difference is small and the natural weight
                is larger than the color one (default 2.0).
            local_contrast_window: Side length of the box window
                used for the local std-dev estimate; auto-bumped
                to odd >= 3.
            seed: Base RNG seed; each batch item uses seed + b.
            debug_prints: Print progress.

        Returns:
            (infilled_image, offset_map_viz) both (B, H, W, C) float32.
        """
        self._debug_print(debug_prints, f"Input image shape: {image.shape}")
        self._debug_print(debug_prints, f"Input mask shape: {mask.shape}")

        if patch_size % 2 == 0:
            patch_size += 1
            self._debug_print(
                debug_prints, f"Adjusted patch_size to odd: {patch_size}"
            )
        half = patch_size // 2
        levels = max(1, int(multiscale_levels))

        batch_size = image.shape[0]
        results = []
        offset_vizs = []

        for b in range(batch_size):
            img = image[b].detach().cpu().numpy().astype(np.float32)
            msk = mask[b].detach().cpu().numpy().astype(np.float32)

            # Source-discipline enforcement: zero out the target region of
            # the working image so that NO downstream step - even by
            # accident - can read original pixel content from inside the
            # target region. Every algorithmic step (patch distance,
            # pyramid downsample, voting reconstruction, boundary blend)
            # independently respects the mask; this extra zeroing is a
            # belt-and-suspenders guarantee. Source pixels are preserved
            # bit-exact.
            img = img * (msk < 0.5).astype(np.float32)[..., None]

            self._debug_print(
                debug_prints, f"Batch {b + 1}/{batch_size}: "
                f"image={img.shape} mask={msk.shape}"
            )

            # No-op if the mask has no target region.
            if (msk > 0.5).sum() == 0:
                filled = img.copy()
                offs = np.zeros(msk.shape + (2,), dtype=np.int32)
            else:
                # Build pyramid using the source-aware downsampler so that
                # target content never contaminates coarse source pixels.
                pyr_imgs = [img]
                pyr_msks = [msk]
                for _ in range(levels - 1):
                    small_img, small_msk = self._source_aware_pyrdown(
                        pyr_imgs[-1], pyr_msks[-1]
                    )
                    # Stop growing the pyramid if it becomes degenerate.
                    if (min(small_img.shape[:2]) < patch_size * 2
                            or small_msk.sum() == 0
                            or small_msk.sum() == small_msk.size):
                        break
                    pyr_imgs.append(small_img)
                    pyr_msks.append(small_msk)

                rng = np.random.default_rng(int(seed) + b)
                offs = None
                # Traverse coarse -> fine.
                for lvl in range(len(pyr_imgs) - 1, -1, -1):
                    lvl_img = pyr_imgs[lvl]
                    lvl_msk = pyr_msks[lvl]
                    lvl_h, lvl_w = lvl_msk.shape
                    # Scale the random-search radius with the level.
                    lvl_radius = max(4, search_radius // (2 ** lvl))
                    if offs is not None:
                        # Upsample offsets from the coarser level.
                        offs_f = cv2.resize(
                            offs.astype(np.float32), (lvl_w, lvl_h),
                            interpolation=cv2.INTER_NEAREST,
                        )
                        offs_up = (offs_f * 2.0).astype(np.int32)
                        yy = np.arange(lvl_h)[:, None]
                        xx = np.arange(lvl_w)[None, :]
                        ty = np.clip(yy + offs_up[..., 0], 0, lvl_h - 1)
                        tx = np.clip(xx + offs_up[..., 1], 0, lvl_w - 1)
                        offs_up[..., 0] = ty - yy
                        offs_up[..., 1] = tx - xx
                        offs = offs_up
                        self._debug_print(
                            debug_prints,
                            f"  level {lvl}: upsampled to {lvl_w}x{lvl_h}"
                        )
                    else:
                        self._debug_print(
                            debug_prints,
                            f"  level {lvl}: init at {lvl_w}x{lvl_h}"
                        )
                    # Build per-level priors (or None if both disabled).
                    # Recomputed at each level so that boundary-color
                    # extrapolation and local-contrast statistics are
                    # taken from the source content at the appropriate
                    # scale.
                    lvl_priors = self._build_priors(
                        lvl_img, lvl_msk,
                        use_boundary_color_prior, boundary_color_weight,
                        use_local_contrast_prior, local_contrast_weight,
                        local_contrast_window,
                    )
                    if debug_prints and lvl_priors is not None:
                        self._debug_print(
                            debug_prints,
                            f"    priors: w_color={lvl_priors['w_color']:.2f} "
                            f"w_contrast={lvl_priors['w_contrast']:.2f}"
                        )
                    offs = self._run_single_level(
                        lvl_img, lvl_msk, patch_size, iterations,
                        lvl_radius, offs, rng, debug_prints,
                        priors=lvl_priors,
                    )

                # Reconstruct at the finest (original) level.
                filled = self._reconstruct_voting(img, msk, offs, half)
                filled = self._blend_boundary(img, filled, msk, blend_width)

            filled = np.clip(filled, 0.0, 1.0).astype(np.float32)

            # Offset-magnitude visualization.
            mag = np.sqrt(
                offs[..., 0].astype(np.float32) ** 2
                + offs[..., 1].astype(np.float32) ** 2
            )
            if mag.max() > 0:
                mag = mag / mag.max()
            offset_viz_rgb = np.stack([mag] * 3, axis=-1).astype(np.float32)

            results.append(torch.from_numpy(filled).float())
            offset_vizs.append(torch.from_numpy(offset_viz_rgb).float())

        output = torch.stack(results, dim=0)
        offset_output = torch.stack(offset_vizs, dim=0)

        self._debug_print(debug_prints, f"Output shape: {output.shape}")
        return (output, offset_output)
    #endregion


# ComfyUI node registration
NODE_CLASS_MAPPINGS = {
    "illumoraeImageInfillPatchMatchNode": illumoraeImageInfillPatchMatchNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "illumoraeImageInfillPatchMatchNode": "Image Infill PatchMatch",
}
