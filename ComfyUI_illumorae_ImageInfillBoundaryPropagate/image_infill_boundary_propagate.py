"""
Image Infill Boundary Propagate
A ComfyUI node that fills the target region by propagating colors inward
from the target boundary, one frontier ring at a time.

Takes a mask (white=target region, black=source region) and fills the
target region by repeatedly extending the source area inward by a
single 8-connected ring. Each frontier pixel is assigned a *random
weighted average* of its already-known 8-neighbors (a stochastic
onion-peel rule), which avoids the directional streak / ridge artifacts
of deterministic max-filter dilation. Useful for preparing images for
inpainting or background extension.

Terminology (Criminisi et al., 2004):
  - target region (Omega): pixels to be inpainted (mask == 1)
  - source region (Phi):   known pixels (mask == 0)
  - target boundary:       the interface between the two
  - mask:                  the user-supplied indicator array

TITLE::Image Infill Boundary Propagate
DESCRIPTIONSHORT::Fills the target region by propagating colors inward from the target boundary using a random-weighted onion-peel rule (simple content fill for inpainting prep).
VERSION::20260427
IMAGE::comfyui_illumorae_image_infill_boundary_propagate.png
GROUP::Image
"""
#region IMPORTS
from __future__ import annotations

import numpy as np
import cv2
from typing import Tuple

# torch is required at runtime inside ComfyUI but optional for unit tests
try:
    import torch
    _HAS_TORCH = True
except ImportError:  # pragma: no cover
    torch = None  # type: ignore
    _HAS_TORCH = False
#endregion


class illumoraeImageInfillBoundaryPropagateNode:
    """
    A ComfyUI node that fills the target region by propagating colors
    inward from the target boundary.

    Takes a mask (white=target region, black=source region) and fills the
    target region by iteratively extending the source area one
    8-connected frontier ring at a time. At each iteration, every
    frontier pixel (an unknown pixel with at least one already-known
    neighbor) is assigned a *random weighted average* of its 8 known
    neighbors. The randomness breaks the directional streak / ridge
    artifacts produced by deterministic max-filter dilation and gives a
    softer fill that tracks the local color statistics of the source
    region. Useful for preparing images for inpainting or background
    extension.

    Note on naming: ``cv2.dilate`` is used internally only as a way to
    detect each new frontier ring; the fill rule itself is the
    random-weighted neighbor average above. The technique is not PDE
    diffusion (no Laplacian, no isotropic smoothing) and is not pure
    morphological dilation (which would be a max filter).

    Fill behavior:
      - The fill is *unbounded* / auto-extending: the loop runs until
        the target region is completely filled. The
        ``propagate_iterations`` parameter is treated as a *minimum*
        ring budget - the algorithm extends past it whenever the target
        region is deeper than that, using the L2 distance transform of
        the target as an upper bound on the required ring count.
      - Random weighting is deterministic (seed = 0 by default), so
        identical inputs always produce identical outputs and the
        source-discipline regression test can compare bit-exact.

    Source-discipline invariant (strict):
      NO original pixel content from inside the target region is ever
      read by the algorithm. Concretely:
        - The working image has its target pixels zeroed at entry, so
          any code path that reads target positions sees 0, not the
          original content.
        - The propagation loop only ever reads from already-known
          pixels (source pixels, or target pixels that have already
          been filled by an earlier ring); source pixels are never
          modified.
        - The final composite is implicit (the algorithm leaves source
          pixels bit-exact); no soft-mask blend with the original image
          can smuggle target content in.
        - The center-blur post-pass operates on the already-clean
          propagated image, so it cannot introduce a leak.
      Source pixels pass through to the output bit-exact.

    Toggleable propagation-quality features:
      Both features below are opt-in via independently visible ComfyUI
      inputs, preserve source-discipline (they read only source
      pixels or source-derived propagated pixels), and are addressed
      at visual artifacts of the base algorithm.

      - angular drift (``use_angular_drift``):
        The fixed 8-connected neighborhood has a strong bias toward
        diagonal streaks in the fill - once a direction is
        established at the boundary, the same 8 offsets get chosen
        over and over and the propagation front advances along a
        preferred diagonal. Angular drift replaces the fixed
        8-neighbor ring with a set of random offsets drawn fresh
        each iteration from a disk of radius ``drift_radius``. The
        sample set changes every ring, so no single direction can
        dominate the propagation for long. The frontier detector
        still uses the 3x3 structuring element, so the algorithm
        still grows exactly one ring per iteration; only the colors
        used for the random-weighted average are pulled from a
        wider, direction-shuffled neighborhood.

      - center blend (``use_center_blend``):
        When opposing propagation fronts meet deep inside the target
        region the random-weighted onion-peel rule has no way to
        reconcile them cleanly and you get a harsh color seam where
        they collide. Center blend precomputes the mean color of
        the source pixels that lie immediately adjacent to the
        target boundary (the "starting boundary" that the
        propagation is extending inward), and at each frontier
        pixel mixes the local random-weighted average toward this
        boundary-mean color with a weight that grows with the
        pixel's depth inside the target region. Pixels on the
        boundary itself (depth 0) are unaffected; pixels at the
        deepest point of the target region are pulled toward the
        boundary-mean by the full ``center_blend_strength``. The
        net effect is that interior fronts converge toward a
        shared neutral tone rather than colliding, while the
        boundary transition still tracks the local source content.
    """

    #region INIT
    def __init__(self):
        pass
    #endregion

    #region C-PROPGATE
    def _sample_disk_offsets(
        self,
        rng: np.random.Generator,
        radius: int,
        num_desired: int = 12,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Random ``(dy, dx)`` offsets drawn uniformly from a disk of
        the given radius, excluding the origin.

        Uses vectorized rejection sampling: oversample a bounding
        square, keep samples inside the disk and different from
        ``(0, 0)``. Returns at most ``num_desired`` offsets (can be
        fewer if the radius is very small; the caller always gets at
        least one usable offset, with a hard fallback to a 4-connected
        ring if the disk is degenerate).
        """
        radius = max(1, int(radius))
        batch = max(int(num_desired) * 4, 48)
        dy = rng.integers(-radius, radius + 1, size=batch)
        dx = rng.integers(-radius, radius + 1, size=batch)
        ok = ((dy * dy + dx * dx) <= radius * radius) & ((dy != 0) | (dx != 0))
        dy = dy[ok][:num_desired]
        dx = dx[ok][:num_desired]
        if dy.size == 0:
            # Degenerate fallback (radius effectively 0 after rejection).
            dy = np.array([0,  0, 1, -1], dtype=np.int64)
            dx = np.array([1, -1, 0,  0], dtype=np.int64)
        return dy.astype(np.int64), dx.astype(np.int64)

    def propagate_boundary_rgb(
        self,
        image_np: np.ndarray,
        mask_np: np.ndarray,
        iterations: int,
        seed: int = 0,
        use_angular_drift: bool = True,
        drift_radius: int = 2,
        use_center_blend: bool = True,
        center_blend_strength: float = 0.14,
    ) -> np.ndarray:
        """Random-weighted onion-peel infill.

        Fills the target region by iteratively extending the known/source
        area inward, one 8-connected frontier ring per iteration. At each
        iteration, every frontier pixel (an unknown pixel with at least
        one already-known neighbor) is set to a *random weighted average*
        of its known neighbors. The random weighting breaks the
        directional streak / ridge artifacts produced by deterministic
        max-filter dilation (e.g. ``cv2.dilate``) and gives a softer fill
        that follows the local color statistics of the source region.

        The loop runs until the target region is fully filled (no
        remaining frontier). The user-supplied ``iterations`` is treated
        as a *minimum* ring budget; the algorithm auto-extends past it
        whenever the target region is deeper than that, using the L2
        distance transform of the target as an upper bound on the
        required ring count. ``iterations`` is therefore effectively a
        no-op cap and any value >= 1 produces a fully filled output.

        Source discipline: the input image's target pixels are expected
        to have been zeroed by ``process_single_numpy`` before this
        routine is called. As a defensive measure we re-zero them here
        as well. Source pixels are never modified, so the returned
        image is the final composite directly (target = propagated
        content, source = bit-exact original).

        Args:
            image_np: Image as numpy array (H, W, C) in range [0, 1].
            mask_np:  Mask as numpy array (H, W) where 1 = target
                region, 0 = source region.
            iterations: Minimum frontier-ring iterations. The algorithm
                extends past this automatically until the target region
                is filled.
            seed: RNG seed for the random weighting. Fixed by default
                so the algorithm is deterministic (and unit-testable).
            use_angular_drift: When True, the per-iteration color
                neighborhood is a random disk of radius
                ``drift_radius`` (resampled each ring) instead of
                the fixed 8-connected ring. Breaks diagonal-streak
                artifacts.
            drift_radius: Radius of the disk used for angular drift.
            use_center_blend: When True, frontier colors are mixed
                toward the mean color of the source pixels
                immediately adjacent to the target boundary, with a
                weight that grows linearly with the pixel's
                normalized L2 depth inside the target region.
            center_blend_strength: Maximum mix weight toward the
                boundary-mean color, reached at the deepest point
                of the target region. 0 = off, 1 = pure boundary
                mean at the deepest point.

        Returns:
            Infilled image as numpy array (H, W, C) in float32 [0, 1].
        """
        image_f = np.asarray(image_np, dtype=np.float32)
        squeeze_chan = (image_f.ndim == 2)
        if squeeze_chan:
            image_f = image_f[..., None]
        H, W, C = image_f.shape

        is_target = (mask_np > 0.5)
        # Nothing to fill = return original unchanged.
        if not is_target.any():
            return image_f[..., 0] if squeeze_chan else image_f.copy()

        # Defensive re-zero of the target region (process_single_numpy
        # already did this; we repeat it here so the routine is also
        # safe to call standalone).
        result = image_f.copy()
        result[is_target] = 0.0

        # `is_known[y, x]` is 1.0 if (y, x) currently holds a real
        # value (source pixel or already-propagated target pixel),
        # else 0.0.
        is_known = (~is_target).astype(np.float32)

        # Auto-extend iteration budget. The target region's max L2
        # depth is an upper bound on the number of 8-connected frontier
        # rings required to fill it.
        target_uint8 = is_target.astype(np.uint8) * 255
        target_dist = cv2.distanceTransform(target_uint8, cv2.DIST_L2, 5)
        needed = int(np.ceil(float(target_dist.max()))) + 2
        total_iters = max(int(iterations), needed)

        # Fixed fallback 8-connected offsets (used when angular drift is disabled).
        FIXED_OFFS_Y = np.array(
            [-1, -1, -1,  0, 0,  1, 1, 1], dtype=np.int64
        )
        FIXED_OFFS_X = np.array(
            [-1,  0,  1, -1, 1, -1, 0, 1], dtype=np.int64
        )

        rng = np.random.default_rng(int(seed))
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

        # Center-blend precomputation: the mean color over source
        # pixels that lie immediately adjacent to the target boundary
        # (a 1-pixel ring of source touching target). This is the
        # "starting boundary" whose average the interior will converge
        # toward. Source-discipline safe: reads only source pixels.
        use_center = bool(use_center_blend) and center_blend_strength > 0.0
        depth_norm = None
        boundary_mean = None
        if use_center:
            dilated_target = cv2.dilate(
                target_uint8, kernel
            ) > 0            # source pixels touching target
            boundary_src = dilated_target & (~is_target)
            if boundary_src.any():
                boundary_mean = (
                    image_f[boundary_src].mean(axis=0).astype(np.float32)
                )
            elif (~is_target).any():
                # Fallback: mean over all source pixels.
                boundary_mean = (
                    image_f[~is_target].mean(axis=0).astype(np.float32)
                )
            else:
                # No source at all: disable the blend.
                use_center = False
            max_depth = float(target_dist.max())
            if max_depth > 0.0:
                depth_norm = (target_dist / max_depth).astype(np.float32)
            else:
                use_center = False

        for _ in range(total_iters):
            # Frontier = unknown pixels with at least one known neighbor.
            # cv2.dilate is used only as a fast 8-connected ring detector
            # here; the fill rule below is random-weighted averaging,
            # not max-filter dilation.
            dilated_known = cv2.dilate(is_known, kernel)
            frontier_mask = (dilated_known > 0.5) & (is_known < 0.5)
            if not frontier_mask.any():
                break

            ys, xs = np.where(frontier_mask)
            N = ys.size

            # Per-iteration color neighborhood: either a fresh random
            # disk (angular drift) or the fixed 8-connected ring.
            if use_angular_drift and drift_radius >= 1:
                OFFS_Y, OFFS_X = self._sample_disk_offsets(
                    rng, drift_radius, num_desired=12,
                )
            else:
                OFFS_Y = FIXED_OFFS_Y
                OFFS_X = FIXED_OFFS_X
            NUM_OFFS = OFFS_Y.size

            # Vectorized neighbor lookup, clipped to image bounds.
            ny = ys[:, None] + OFFS_Y[None, :]      # (N, K)
            nx = xs[:, None] + OFFS_X[None, :]
            in_bounds = (ny >= 0) & (ny < H) & (nx >= 0) & (nx < W)
            ny_c = np.clip(ny, 0, H - 1)
            nx_c = np.clip(nx, 0, W - 1)

            # `n_known` is 1 where the neighbor is in-bounds AND already
            # known; 0 elsewhere. Out-of-bounds neighbors are treated as
            # unknown so they never contribute (no edge wrap-around).
            # This same mask also excludes disk samples that happen to
            # land on still-unknown target pixels, which is essential
            # when ``drift_radius >= 2``.
            n_known = (is_known[ny_c, nx_c] *
                       in_bounds.astype(np.float32))      # (N, K)
            n_values = result[ny_c, nx_c]                 # (N, K, C)

            # Random weights, zeroed for unknown / out-of-bounds neighbors.
            rw = rng.random((N, NUM_OFFS), dtype=np.float32) * n_known
            wsum = rw.sum(axis=1)
            safe = np.maximum(wsum, 1e-6)
            avg = (n_values * rw[..., None]).sum(axis=1) / safe[..., None]

            # Center blend: pull the random-weighted average toward the
            # boundary-mean color by a depth-weighted amount, so
            # opposing fronts converge to a shared neutral tone in the
            # interior instead of colliding harshly.
            if use_center:
                d = depth_norm[ys, xs]                    # (N,)
                blend_w = (center_blend_strength * d).astype(np.float32)
                bw = blend_w[:, None]                     # (N, 1)
                avg = (1.0 - bw) * avg + bw * boundary_mean[None, :]

            # Only update frontier pixels with at least one valid neighbor.
            valid = wsum > 1e-6
            if not valid.any():
                # No progress this iteration; break to avoid spinning.
                break
            sel_y = ys[valid]
            sel_x = xs[valid]
            result[sel_y, sel_x] = avg[valid]
            is_known[sel_y, sel_x] = 1.0

        if squeeze_chan:
            result = result[..., 0]
        # The output is a hard-mask composite
        return result.astype(np.float32)
    #endregion

    #region C-BLUR
    def create_distance_map(self, mask_np: np.ndarray) -> np.ndarray:
        """
        Create a distance map from the target boundary into the target
        region.

        Args:
            mask_np: Mask as numpy array (H, W) where 1 = target region
                (white), 0 = source region (black).

        Returns:
            Distance map as numpy array (H, W) with distances from the
            target boundary into the target region.
        """
        mask_uint8 = (mask_np * 255).astype(np.uint8)

        # Distance transform into the target region (white area).
        dist_transform = cv2.distanceTransform(mask_uint8, cv2.DIST_L2, 5)

        return dist_transform

    def apply_center_blur(
        self,
        image_np: np.ndarray,
        mask_np: np.ndarray,
        blur_strength: float,
        falloff_distance: int
    ) -> np.ndarray:
        """
        Apply blur near the center of the target region where opposing
        propagation fronts meet.

        The blur is strongest at the target's interior (where fronts
        collide) and fades toward the target boundary based on distance
        from the boundary.

        Args:
            image_np: Image as numpy array (H, W, C) in range [0, 1].
            mask_np: Mask as numpy array (H, W) where 1 = target region,
                0 = source region.
            blur_strength: Blur kernel size (will be converted to odd
                integer).
            falloff_distance: Distance from the target boundary at
                which blur reaches full strength.

        Returns:
            Blurred image as numpy array (H, W, C).
        """
        # Get number of channels.
        num_channels = image_np.shape[2] if len(image_np.shape) == 3 else 1

        # Convert blur strength to odd kernel size.
        kernel_size = int(blur_strength)
        if kernel_size % 2 == 0:
            kernel_size += 1
        kernel_size = max(3, kernel_size)

        # Convert to uint8 for OpenCV.
        image_uint8 = (image_np * 255).astype(np.uint8)
        mask_uint8 = (mask_np * 255).astype(np.uint8)

        # Distance map from the target boundary into the target region.
        dist_map = cv2.distanceTransform(mask_uint8, cv2.DIST_L2, 5)

        # Normalize distance to [0, 1] based on falloff distance.
        # Pixels at the target boundary  -> 0 (no blur);
        # pixels deep inside the target -> 1 (full blur).
        blur_weight = np.clip(dist_map / falloff_distance, 0, 1)

        # Apply Gaussian blur to the propagated image.
        blurred = cv2.GaussianBlur(image_uint8, (kernel_size, kernel_size), 0)

        # Stack blur weight to match channels.
        blur_weight_3ch = np.stack([blur_weight] * num_channels, axis=-1)

        # Blend sharp (propagated) with blurred based on distance from
        # the target boundary. Near the boundary (blur_weight=0): keep
        # the sharp propagated result. Deep inside the target
        # (blur_weight=1): use the blurred result. This blending
        # happens everywhere; the source region is unaffected because
        # the propagation step did not modify source pixels.
        result = (image_uint8.astype(np.float32) * (1 - blur_weight_3ch) +
                  blurred.astype(np.float32) * blur_weight_3ch)

        # Convert back to float [0, 1].
        result_float = result.astype(np.float32) / 255.0

        return result_float
    #endregion

    #region C-FEATHER
    def feather_mask(self, mask_np: np.ndarray, feather_amount: int) -> np.ndarray:
        """
        Feather the target boundary for smooth transitions.

        Args:
            mask_np: Mask as numpy array (H, W) where 1 = target region, 0 = source region.
            feather_amount: Number of pixels to feather.

        Returns:
            Feathered mask as numpy array (H, W).
        """
        if feather_amount == 0:
            return mask_np

        # Binarize mask for distance transform.
        mask_binary = (mask_np > 0.5).astype(np.uint8) * 255

        # Distance from the target boundary into the target region
        # (positive inside the target).
        dist_inside = cv2.distanceTransform(mask_binary, cv2.DIST_L2, 5)
        # Distance from the target boundary into the source region
        # (positive inside the source).
        dist_outside = cv2.distanceTransform(255 - mask_binary, cv2.DIST_L2, 5)

        # Signed distance field (positive inside the target, negative
        # in the source), normalized by feather_amount to create a
        # gradient at the boundary:
        #   inside  the target: ramps from 0.5 at boundary to 1.0 at depth
        #   inside the source:  ramps from 0.5 at boundary to 0.0 at depth
        feathered = np.clip(
            0.5 + (dist_inside - dist_outside) / (2.0 * feather_amount),
            0, 1,
        )

        return feathered.astype(np.float32)

    def _resolve_mask(
        self,
        msk: np.ndarray,
        mask_mode: str,
        debug_prints: bool = False,
    ) -> np.ndarray:
        """Apply the mask_mode policy and return a [0, 1] mask whose
        convention is: 1 = target region, 0 = source region."""
        msk = np.clip(msk, 0.0, 1.0)
        if mask_mode == "white=fill":
            self._debug_print(debug_prints, "Mask mode: white=fill")
        elif mask_mode == "black=fill":
            self._debug_print(debug_prints,
                              "Mask mode: black=fill (inverting)")
            msk = 1.0 - msk
        else:
            mask_mean = float(np.mean(msk))
            self._debug_print(debug_prints,
                              f"Mask mode: auto (mean={mask_mean:.4f})")
            if mask_mean > 0.5:
                self._debug_print(debug_prints, "Auto mask invert triggered")
                msk = 1.0 - msk
        return msk
    #endregion

    #region C-DRIVER
    def process_single_numpy(
        self,
        img: np.ndarray,
        msk: np.ndarray,
        mask_mode: str,
        propagate_iterations: int,
        blur_center: bool,
        blur_strength: float,
        blur_falloff_distance: int,
        feather_amount: int,
        use_angular_drift: bool = True,
        drift_radius: int = 2,
        use_center_blend: bool = True,
        center_blend_strength: float = 0.4,
        debug_prints: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Pure-numpy per-image core.

        Returns (filled_image_HWC, dist_map_viz_HWC, blur_weight_viz_HWC),
        each in float32 [0, 1] range.

        Source discipline: the input ``img`` has its target pixels zeroed
        before any further processing, so no original target content can
        be read by any downstream step. See class docstring for the full
        invariant.
        """
        msk = self._resolve_mask(msk, mask_mode, debug_prints=debug_prints)

        # SOURCE-DISCIPLINE ENFORCEMENT
        # Zero the target pixels of the working image immediately. Source
        # pixels are preserved bit-exact. From this point on, no
        # algorithm step can possibly observe the original target
        # content - even if a future code path reads target positions of
        # ``img``, it will see 0 rather than ground-truth.
        is_target = (msk > 0.5).astype(np.float32)
        img = img.astype(np.float32) * (1.0 - is_target)[..., None]

        self._debug_print(debug_prints,
                          f"Image range: [{img.min():.3f}, {img.max():.3f}]")
        self._debug_print(debug_prints,
                          f"Mask range:  [{msk.min():.3f}, {msk.max():.3f}]")

        # Feathered mask is used only for distance / blur-weight
        # computation; the final boundary composite uses the binarized
        # mask (see propagate_boundary_rgb) to keep the source-discipline
        # invariant.
        if feather_amount > 0:
            msk_feathered = self.feather_mask(msk, feather_amount)
            self._debug_print(debug_prints,
                              f"Applied feathering: {feather_amount} pixels")
        else:
            msk_feathered = msk

        # Distance map / blur weight visualizations.
        mask_binary_viz = ((msk_feathered > 0.5).astype(np.uint8)) * 255
        dist_map = cv2.distanceTransform(mask_binary_viz, cv2.DIST_L2, 5)
        dist_map_norm = (dist_map / dist_map.max()
                         if dist_map.max() > 0 else dist_map)
        blur_weight_map = np.clip(dist_map / max(blur_falloff_distance, 1),
                                  0, 1)
        dist_map_rgb = np.stack([dist_map_norm] * 3, axis=-1).astype(np.float32)
        blur_weight_rgb = np.stack([blur_weight_map] * 3,
                                   axis=-1).astype(np.float32)

        # Boundary-propagate to fill the target region.
        self._debug_print(
            debug_prints,
            f"Propagating boundary: {propagate_iterations} iterations"
            f" (angular_drift={use_angular_drift} r={drift_radius},"
            f" center_blend={use_center_blend}"
            f" strength={center_blend_strength})",
        )
        infilled = self.propagate_boundary_rgb(
            img, msk_feathered, propagate_iterations,
            use_angular_drift=use_angular_drift,
            drift_radius=drift_radius,
            use_center_blend=use_center_blend,
            center_blend_strength=center_blend_strength,
        )

        # Optional center blur (operates on the already-clean propagated image).
        if blur_center and blur_strength > 0:
            self._debug_print(
                debug_prints,
                f"Applying center blur: strength={blur_strength}, "
                f"falloff={blur_falloff_distance}"
            )
            infilled = self.apply_center_blur(
                infilled, msk_feathered, blur_strength, blur_falloff_distance
            )

        return (infilled.astype(np.float32),
                dist_map_rgb,
                blur_weight_rgb)
    #endregion

    #region DEBUG
    # All debug output flows through this single helper so every line
    # carries the consistent ``[ ImageInfillBoundaryPropagate ]`` prefix
    # and is easy to filter from a noisy ComfyUI console.
    def _debug_print(self, debug_prints, *args, **kwargs):
        if debug_prints:
            print("[ ImageInfillBoundaryPropagate ]", *args, **kwargs)
    #endregion

    #region UI
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),  # Input image (B, H, W, C) in ComfyUI format
                "mask": ("MASK",),    # Mask (B, H, W) where 1=target region (white), 0=source region (black)
                "mask_mode": (["auto", "white=fill", "black=fill"],),
                "propagate_iterations": ("INT", {
                    "default": 50,
                    "min": 1,
                    "max": 5000,
                    "step": 1,
                    "display": "number"
                }),
                "use_angular_drift": ("BOOLEAN", {"default": True}),
                "drift_radius": ("INT", {
                    "default": 3,
                    "min": 1,
                    "max": 8,
                    "step": 1,
                    "display": "number"
                }),
                "use_center_blend": ("BOOLEAN", {"default": True}),
                "center_blend_strength": ("FLOAT", {
                    "default": 0.1,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05,
                    "display": "number"
                }),
                "blur_center": ("BOOLEAN", {"default": True}),
                "blur_strength": ("FLOAT", {
                    "default": 10.0,
                    "min": 0.0,
                    "max": 10000.0,
                    "step": 0.1,
                    "display": "number"
                }),
                "blur_falloff_distance": ("INT", {
                    "default": 50,
                    "min": 1,
                    "max": 5000,
                    "step": 1,
                    "display": "number"
                }),
                "feather_amount": ("INT", {
                    "default": 10,
                    "min": 0,
                    "max": 1000,
                    "step": 1,
                    "display": "number"
                }),
            },
            "optional": {
                "debug_prints": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE")
    RETURN_NAMES = ("infilled_image", "distance_map_viz", "blur_weight_viz")
    FUNCTION = "infill_target_region"
    CATEGORY = "illumorae"
    OUTPUT_NODE = False
    DESCRIPTION = (
        "Fills the target region by propagating colors inward from the "
        "target boundary using a random-weighted onion-peel rule "
        "(simple content fill for inpainting prep)."
    )
    #endregion

    #region ENTRY
    def infill_target_region(
        self,
        image: "torch.Tensor",
        mask: "torch.Tensor",
        mask_mode: str,
        propagate_iterations: int,
        blur_center: bool,
        blur_strength: float,
        blur_falloff_distance: int,
        feather_amount: int,
        use_angular_drift: bool = True,
        drift_radius: int = 2,
        use_center_blend: bool = True,
        center_blend_strength: float = 0.4,
        debug_prints: bool = False,
    ) -> "Tuple[torch.Tensor, torch.Tensor, torch.Tensor]":
        """Tensor-wrapping batch loop. Per-image work is delegated to
        ``process_single_numpy`` so it can be unit-tested without torch."""
        self._debug_print(debug_prints, f"Input image shape: {image.shape}")
        self._debug_print(debug_prints, f"Input mask shape: {mask.shape}")

        batch_size = image.shape[0]
        results, distance_maps, blur_weights = [], [], []

        for b in range(batch_size):
            self._debug_print(debug_prints,
                              f"Processing batch {b + 1}/{batch_size}")
            img = image[b].cpu().numpy()
            msk = mask[b].cpu().numpy()
            filled, dist_rgb, blur_rgb = self.process_single_numpy(
                img, msk, mask_mode,
                propagate_iterations, blur_center, blur_strength,
                blur_falloff_distance, feather_amount,
                use_angular_drift=use_angular_drift,
                drift_radius=drift_radius,
                use_center_blend=use_center_blend,
                center_blend_strength=center_blend_strength,
                debug_prints=debug_prints,
            )
            results.append(torch.from_numpy(filled).float())
            distance_maps.append(torch.from_numpy(dist_rgb).float())
            blur_weights.append(torch.from_numpy(blur_rgb).float())

        output = torch.stack(results, dim=0)
        distance_output = torch.stack(distance_maps, dim=0)
        blur_weight_output = torch.stack(blur_weights, dim=0)

        self._debug_print(debug_prints, f"Output shape: {output.shape}")
        self._debug_print(debug_prints,
                          f"Distance map output shape: {distance_output.shape}")
        self._debug_print(debug_prints,
                          f"Blur weight output shape: {blur_weight_output.shape}")

        return (output, distance_output, blur_weight_output)
    #endregion


# ComfyUI node registration
NODE_CLASS_MAPPINGS = {
    "illumoraeImageInfillBoundaryPropagateNode": illumoraeImageInfillBoundaryPropagateNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "illumoraeImageInfillBoundaryPropagateNode": "Image Infill Boundary Propagate",
}
