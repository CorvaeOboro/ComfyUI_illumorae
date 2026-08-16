"""
Procedural Patterns Generator
A ComfyUI node that generates procedural symmetrical patterns for visual influence.

Generates patterns including:
- Concentric torus rings with repeating scale
- Radial beams shooting from center point
- Spiral rings combining radial and angular phase
- Combined patterns for complex visual direction

Useful for influencing image generation toward visually interesting results
with directed principles of visual communication.

TITLE::Procedural Patterns
DESCRIPTIONSHORT::Generates procedural symmetrical patterns (torus rings, radial beams, spiral rings, combined) for visual influence in image generation. Seed only affects output when pattern_type='random' or chaos_mode=True.
VERSION::20260814
IMAGE::comfyui_illumorae_procedural_patterns.png
GROUP::Image
GROUPORDER::1
LISTORDER::100
STATUS::working
"""
#region IMPORTS
import torch
import numpy as np
import math
import random
import warnings
from typing import Tuple
#endregion


#region CONST
# Single source of truth for pattern names. Add a new concrete pattern here
# and to the dispatch chain in generate_pattern(); the UI enum below derives
# from this so the two lists cannot drift.
PATTERN_TYPES_CONCRETE = ("torus_rings", "radial_beams", "combined", "spiral_rings")
# UI enum = "random" (seeded family selection) + the concrete families.
PATTERN_TYPES_UI = ("random",) + PATTERN_TYPES_CONCRETE
#endregion


#region FALLOFF
def _band_falloff(phase: np.ndarray, thickness: float, falloff: float) -> np.ndarray:
    """Build a repeating band mask from a phase field in ``[0, 1]``.

    The band is centered at phase 0.5 with the given ``thickness`` (full
    width in phase units). ``falloff`` in ``[0, 1]`` controls the soft
    edge as a fraction of the half-thickness: ``0`` gives a hard edge,
    ``1`` turns the whole band into a linear ramp from 1 at its center
    to 0 at its edge (the hard core shrinks to a measure-zero line).

    The phase wrap-around at 0/1 is handled so a band straddling the
    seam stays continuous. The division in the soft-edge branch is
    evaluated only inside the falloff zone (via ``np.divide(where=...)``)
    so tiny ``falloff_zone`` values cannot produce transient ``inf`` /
    ``nan`` arrays or ``RuntimeWarning``s; out-of-zone pixels are set
    directly via ``out``.
    """
    half_thickness = thickness / 2.0
    # Distance from band center in phase space, with wrap-around.
    phase_dist = np.abs(phase - 0.5)
    phase_dist = np.minimum(phase_dist, 1.0 - phase_dist)

    if falloff > 0:
        falloff_zone = half_thickness * falloff
        inner_edge = half_thickness - falloff_zone
        # Falloff mask: pixels strictly inside the band but outside the
        # hard core. Using inner_edge <= phase_dist <= half_thickness.
        soft_mask = (phase_dist > inner_edge) & (phase_dist <= half_thickness)
        # Linear ramp from 1 at inner_edge to 0 at half_thickness.
        # np.divide with where=/out= avoids dividing outside soft_mask,
        # so falloff_zone underflow cannot produce inf/nan temporaries.
        ramp = np.divide(
            phase_dist - inner_edge,
            falloff_zone,
            out=np.zeros_like(phase_dist),
            where=soft_mask,
        )
        pattern = np.where(
            phase_dist <= inner_edge,
            1.0,
            np.where(soft_mask, 1.0 - ramp, 0.0),
        )
    else:
        pattern = np.where(phase_dist <= half_thickness, 1.0, 0.0)

    return pattern.astype(np.float32)
#endregion


class illumoraeProceduralPatternsNode:
    """
    A ComfyUI node that generates procedural symmetrical patterns.

    Creates visually interesting patterns that can be used to influence
    image generation with directed visual principles:
    - Concentric torus rings create depth and focus
    - Radial beams create energy and direction
    """

    #region CORE
    # Main entry point: seeds RNG, resolves chaos/random mode, dispatches to
    # the pattern builders, applies invert + range clamp, and emits the
    # ComfyUI IMAGE + MASK tensors.
    def generate_pattern(
        self,
        width: int,
        height: int,
        pattern_type: str,
        ring_count: int,
        ring_thickness: float,
        ring_falloff: float,
        beam_count: int,
        beam_width: float,
        beam_falloff: float,
        center_x: float,
        center_y: float,
        rotation: float,
        scale: float,
        invert: bool,
        colorize: bool,
        chaos_mode: bool = False,
        seed: int = 0,
        color_a: str = "#000000",
        color_b: str = "#FFFFFF",
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Main processing function for pattern generation.
        """
        rng = random.Random(int(seed))

        # Chaos mode: randomize every value and preset based on seed.
        # Width/height are preserved (output resolution is a user intent).
        if chaos_mode:
            pattern_type = rng.choice(PATTERN_TYPES_CONCRETE)
            ring_count = rng.randint(2, 40)
            ring_thickness = rng.uniform(0.05, 0.95)
            ring_falloff = rng.uniform(0.0, 1.0)
            beam_count = rng.randint(2, 64)
            beam_width = rng.uniform(0.05, 0.95)
            beam_falloff = rng.uniform(0.0, 1.0)
            center_x = rng.uniform(0.2, 0.8)
            center_y = rng.uniform(0.2, 0.8)
            rotation = rng.uniform(0.0, 360.0)
            scale = rng.uniform(0.3, 4.0)
            invert = rng.random() < 0.5
            colorize = rng.random() < 0.5
            color_a = "#{:06X}".format(rng.randint(0, 0xFFFFFF))
            color_b = "#{:06X}".format(rng.randint(0, 0xFFFFFF))
        elif pattern_type == "random":
            # Only randomize the pattern type selection (not other params),
            # seeded so output is deterministic given the seed.
            pattern_type = rng.choice(PATTERN_TYPES_CONCRETE)

        # Generate base pattern based on type
        if pattern_type == "torus_rings":
            pattern = self.generate_torus_rings(
                width, height, ring_count, ring_thickness, ring_falloff,
                center_x, center_y, scale
            )
        elif pattern_type == "radial_beams":
            pattern = self.generate_radial_beams(
                width, height, beam_count, beam_width, beam_falloff,
                center_x, center_y, rotation
            )
        elif pattern_type == "spiral_rings":
            pattern = self.generate_spiral_rings(
                width, height, ring_count, ring_thickness, ring_falloff,
                center_x, center_y, scale, rotation
            )
        elif pattern_type == "combined":
            # Combine rings and beams
            rings = self.generate_torus_rings(
                width, height, ring_count, ring_thickness, ring_falloff,
                center_x, center_y, scale
            )
            beams = self.generate_radial_beams(
                width, height, beam_count, beam_width, beam_falloff,
                center_x, center_y, rotation
            )
            # Multiply for intersection effect
            pattern = rings * beams
        else:
            # Default to rings
            pattern = self.generate_torus_rings(
                width, height, ring_count, ring_thickness, ring_falloff,
                center_x, center_y, scale
            )

        # Invert if requested
        if invert:
            pattern = 1.0 - pattern

        # Intended range guidance: the builders produce values in [0, 1] by
        # construction, but clamp here so the range is robust to future
        # edits and so no nan/inf can reach downstream nodes.
        pattern = np.clip(pattern, 0.0, 1.0)

        # Create mask output (single channel). Copy after the clip so mask
        # and image share the same guaranteed range.
        mask = pattern.copy()

        # Create image output (RGB)
        if colorize:
            # Parse colors
            rgb_a = self.hex_to_rgb(color_a)
            rgb_b = self.hex_to_rgb(color_b)

            # Interpolate between colors based on pattern. Vectorised broadcast
            # over channels: rgb_a/rgb_b as (3,) arrays, pattern as (H, W, 1).
            rgb_a_arr = np.array(rgb_a, dtype=np.float32).reshape(3)
            rgb_b_arr = np.array(rgb_b, dtype=np.float32).reshape(3)
            image = (
                rgb_a_arr * (1.0 - pattern)[..., None]
                + rgb_b_arr * pattern[..., None]
            )
        else:
            # Grayscale
            image = np.stack([pattern, pattern, pattern], axis=-1)

        # Convert to tensors (add batch dimension). image/mask are already
        # float32, so .float() is a no-op; kept for ComfyUI convention.
        image_tensor = torch.from_numpy(image).unsqueeze(0).float()
        mask_tensor = torch.from_numpy(mask).unsqueeze(0).float()

        return (image_tensor, mask_tensor)
    #endregion

    #region PATTERNS
    # Pure pattern builders, each returning an (H, W) float32 [0, 1] array.
    # All are @staticmethod so they can be unit-tested without an instance.

    @staticmethod
    def generate_torus_rings(
        width: int,
        height: int,
        ring_count: int,
        ring_thickness: float,
        ring_falloff: float,
        center_x: float,
        center_y: float,
        scale: float
    ) -> np.ndarray:
        """
        Generate concentric torus-like rings pattern.

        Creates thick rings that repeat at regular intervals from center,
        with smooth falloff at edges for a torus-like appearance.
        """
        # Create coordinate grids
        y_coords = np.linspace(0, 1, height)
        x_coords = np.linspace(0, 1, width)
        xx, yy = np.meshgrid(x_coords, y_coords)

        # Calculate distance from center (normalized)
        dx = (xx - center_x) * (width / min(width, height))
        dy = (yy - center_y) * (height / min(width, height))
        distance = np.sqrt(dx**2 + dy**2) * scale

        # Create repeating ring pattern
        # Map distance to ring phase (0 to 1 repeating)
        ring_phase = (distance * ring_count) % 1.0

        # Build the band mask (centered at phase 0.5 with wrap-around and
        # safe soft-edge division).
        pattern = _band_falloff(ring_phase, ring_thickness, ring_falloff)

        return pattern

    @staticmethod
    def generate_radial_beams(
        width: int,
        height: int,
        beam_count: int,
        beam_width: float,
        beam_falloff: float,
        center_x: float,
        center_y: float,
        rotation: float
    ) -> np.ndarray:
        """
        Generate radial beams shooting from center point.

        Creates evenly spaced beams radiating outward from the center,
        with configurable width and edge falloff.
        """
        # Create coordinate grids
        y_coords = np.linspace(0, 1, height)
        x_coords = np.linspace(0, 1, width)
        xx, yy = np.meshgrid(x_coords, y_coords)

        # Calculate angle from center. Scale by the aspect ratio so beams are
        # isotropic in pixel space on non-square canvases (matches the
        # torus_rings / spiral_rings builders).
        dx = (xx - center_x) * (width / min(width, height))
        dy = (yy - center_y) * (height / min(width, height))
        angle = np.arctan2(dy, dx)

        # Apply rotation (convert degrees to radians)
        rotation_rad = rotation * math.pi / 180.0
        angle = angle - rotation_rad

        # Normalize angle to 0-1 range. After rotation `angle` may leave
        # [-pi, pi]; the later modulo on beam_phase absorbs the wrap, so the
        # output is correct even though this value is not strictly in [0, 1].
        angle_norm = (angle + math.pi) / (2 * math.pi)

        # Create repeating beam pattern
        beam_phase = (angle_norm * beam_count) % 1.0

        # Build the band mask (centered at phase 0.5 with wrap-around and
        # safe soft-edge division).
        pattern = _band_falloff(beam_phase, beam_width, beam_falloff)

        return pattern

    @staticmethod
    def generate_spiral_rings(
        width: int,
        height: int,
        ring_count: int,
        ring_thickness: float,
        ring_falloff: float,
        center_x: float,
        center_y: float,
        scale: float,
        rotation: float
    ) -> np.ndarray:
        """
        Generate spiral ring pattern (rings that twist with angle).

        Creates a hypnotic spiral effect by combining radial distance
        with angular position.
        """
        # Create coordinate grids
        y_coords = np.linspace(0, 1, height)
        x_coords = np.linspace(0, 1, width)
        xx, yy = np.meshgrid(x_coords, y_coords)

        # Calculate distance and angle from center
        dx = (xx - center_x) * (width / min(width, height))
        dy = (yy - center_y) * (height / min(width, height))
        distance = np.sqrt(dx**2 + dy**2) * scale
        angle = np.arctan2(dy, dx)

        # Apply rotation
        rotation_rad = rotation * math.pi / 180.0
        angle = angle - rotation_rad

        # Normalize angle to 0-1 range
        angle_norm = (angle + math.pi) / (2 * math.pi)

        # Combine distance and angle for spiral effect
        spiral_phase = (distance * ring_count + angle_norm) % 1.0

        # Build the band mask (centered at phase 0.5 with wrap-around and
        # safe soft-edge division).
        pattern = _band_falloff(spiral_phase, ring_thickness, ring_falloff)

        return pattern
    #endregion

    #region COLOR
    # Hex color parsing helper used by generate_pattern's colorize path.
    @staticmethod
    def hex_to_rgb(hex_color: str) -> Tuple[float, float, float]:
        """Convert a hex color string to an RGB tuple in the 0-1 range.

        Accepts ``#RRGGBB`` and the 3-digit shorthand ``#RGB`` (expanded to
        ``#RRGGBB``). Malformed input emits a ``UserWarning`` and falls back
        to black so the node keeps running instead of crashing.
        """
        s = hex_color.strip().lstrip('#')
        if len(s) == 3:
            s = ''.join(ch * 2 for ch in s)
        if len(s) != 6:
            warnings.warn(
                f"hex_to_rgb: malformed color {hex_color!r}; expected "
                f"'#RRGGBB' or '#RGB'. Falling back to black.",
                UserWarning,
                stacklevel=2,
            )
            return (0.0, 0.0, 0.0)
        try:
            r = int(s[0:2], 16) / 255.0
            g = int(s[2:4], 16) / 255.0
            b = int(s[4:6], 16) / 255.0
            return (r, g, b)
        except ValueError:
            warnings.warn(
                f"hex_to_rgb: non-hex characters in color {hex_color!r}. "
                f"Falling back to black.",
                UserWarning,
                stacklevel=2,
            )
            return (0.0, 0.0, 0.0)
    #endregion

    #region UI
    # ComfyUI interface: input schema, return types, function binding, and
    # node metadata used by the docs generator and the ComfyUI registry.
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "width": ("INT", {
                    "default": 512,
                    "min": 64,
                    "max": 8192,
                    "step": 8,
                    "display": "number"
                }),
                "height": ("INT", {
                    "default": 512,
                    "min": 64,
                    "max": 8192,
                    "step": 8,
                    "display": "number"
                }),
                "pattern_type": (list(PATTERN_TYPES_UI),),
                "ring_count": ("INT", {
                    "default": 8,
                    "min": 1,
                    "max": 100,
                    "step": 1,
                    "display": "number"
                }),
                "ring_thickness": ("FLOAT", {
                    "default": 0.3,
                    "min": 0.01,
                    "max": 1.0,
                    "step": 0.01,
                    "display": "number"
                }),
                "ring_falloff": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    # max < 1.0 keeps a non-degenerate hard core; at 1.0 the
                    # core shrinks to a measure-zero line and the whole ring
                    # becomes a linear ramp.
                    "max": 0.99,
                    "step": 0.01,
                    "display": "number"
                }),
                "beam_count": ("INT", {
                    "default": 12,
                    "min": 1,
                    "max": 360,
                    "step": 1,
                    "display": "number"
                }),
                "beam_width": ("FLOAT", {
                    "default": 0.3,
                    "min": 0.01,
                    "max": 1.0,
                    "step": 0.01,
                    "display": "number"
                }),
                "beam_falloff": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    # max < 1.0 keeps a non-degenerate hard core; at 1.0 the
                    # core shrinks to a measure-zero line and the whole beam
                    # becomes a linear ramp.
                    "max": 0.99,
                    "step": 0.01,
                    "display": "number"
                }),
                "center_x": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "display": "number"
                }),
                "center_y": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "display": "number"
                }),
                "rotation": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 360.0,
                    "step": 1.0,
                    "display": "number"
                }),
                "scale": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.1,
                    "max": 10.0,
                    "step": 0.1,
                    "display": "number"
                }),
                "invert": ("BOOLEAN", {"default": False}),
                "colorize": ("BOOLEAN", {"default": False}),
                "chaos_mode": ("BOOLEAN", {"default": False}),
                "seed": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 0xffffffffffffffff,
                    "control_after_generate": True,
                }),
            },
            "optional": {
                "color_a": ("STRING", {"default": "#000000"}),
                "color_b": ("STRING", {"default": "#FFFFFF"}),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("pattern_image", "pattern_mask")
    FUNCTION = "generate_pattern"
    CATEGORY = "illumorae"
    OUTPUT_NODE = False
    DESCRIPTION = ("Generates procedural symmetrical patterns (torus rings, "
                   "radial beams, spiral rings, combined) for visual influence "
                   "in image generation. The seed only takes effect when "
                   "pattern_type='random' (selects the family) or "
                   "chaos_mode=True (randomizes all parameters); otherwise the "
                   "output is fully deterministic and seed-independent.")
    #endregion


#region MAPPING
# Module-level ComfyUI registry mappings consumed by __init__.py.
NODE_CLASS_MAPPINGS = {
    "illumoraeProceduralPatternsNode": illumoraeProceduralPatternsNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "illumoraeProceduralPatternsNode": "Procedural Patterns",
}
#endregion
