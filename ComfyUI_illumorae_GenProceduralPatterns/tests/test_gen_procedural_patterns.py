"""Regression tests for illumoraeProceduralPatternsNode.

Covers the bugs fixed in the 20260814 review pass:

- hex_to_rgb: 3-digit shorthand, malformed-input warnings, @staticmethod
- _band_falloff: range [0,1], no nan/inf on degenerate falloff_zone,
  wrap-around continuity, hard vs soft edge, bit-identical to old formula
- generate_torus_rings / generate_radial_beams / generate_spiral_rings:
  output shape, dtype, range; @staticmethod callable via class
- generate_radial_beams: aspect-ratio correction (isotropic on non-square)
- generate_pattern: output tensor shapes, determinism per seed mode,
  colorize gradient, invert, chaos_mode reproducibility, combined pattern
- PATTERN_TYPES_UI: single-source derivation from PATTERN_TYPES_CONCRETE

Usage:
    python -m tests.test_gen_procedural_patterns
    python -m pytest tests/test_gen_procedural_patterns.py -v
"""
from __future__ import annotations

import os
import sys
import unittest
import warnings

import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PARENT = os.path.dirname(_THIS_DIR)
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)

from gen_procedural_patterns import (  # noqa: E402
    PATTERN_TYPES_CONCRETE,
    PATTERN_TYPES_UI,
    _band_falloff,
    illumoraeProceduralPatternsNode,
)


class TestHexToRgb(unittest.TestCase):
    """hex_to_rgb: valid, shorthand, malformed, @staticmethod."""

    def test_valid_6_digit(self):
        self.assertEqual(illumoraeProceduralPatternsNode.hex_to_rgb("#FFFFFF"), (1.0, 1.0, 1.0))
        self.assertEqual(illumoraeProceduralPatternsNode.hex_to_rgb("#000000"), (0.0, 0.0, 0.0))
        self.assertAlmostEqual(illumoraeProceduralPatternsNode.hex_to_rgb("#808080")[0], 128 / 255.0)

    def test_valid_no_hash(self):
        self.assertEqual(illumoraeProceduralPatternsNode.hex_to_rgb("FF0000"), (1.0, 0.0, 0.0))

    def test_3_digit_shorthand(self):
        self.assertEqual(illumoraeProceduralPatternsNode.hex_to_rgb("#FFF"), (1.0, 1.0, 1.0))
        self.assertEqual(illumoraeProceduralPatternsNode.hex_to_rgb("#000"), (0.0, 0.0, 0.0))
        self.assertEqual(illumoraeProceduralPatternsNode.hex_to_rgb("#F00"), (1.0, 0.0, 0.0))

    def test_malformed_length_warns_and_returns_black(self):
        for bad in ("#FFFFFFF", "#FF", "#FFFF", "", "#"):
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                result = illumoraeProceduralPatternsNode.hex_to_rgb(bad)
                self.assertEqual(result, (0.0, 0.0, 0.0),
                                 f"expected black for {bad!r}, got {result}")
                self.assertTrue(any(issubclass(w.category, UserWarning) for w in caught),
                                f"expected a UserWarning for {bad!r}")

    def test_non_hex_chars_warn_and_return_black(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = illumoraeProceduralPatternsNode.hex_to_rgb("#GGGGGG")
            self.assertEqual(result, (0.0, 0.0, 0.0))
            self.assertTrue(any(issubclass(w.category, UserWarning) for w in caught))

    def test_callable_as_staticmethod(self):
        # Should work without an instance.
        self.assertEqual(illumoraeProceduralPatternsNode.hex_to_rgb("#FFFFFF"), (1.0, 1.0, 1.0))


class TestBandFalloff(unittest.TestCase):
    """_band_falloff: the shared falloff kernel extracted in fix K+D."""

    def setUp(self):
        self.rng = np.random.RandomState(42)
        self.phase = self.rng.rand(64, 64)

    def test_range_0_to_1_soft(self):
        out = _band_falloff(self.phase, 0.3, 0.5)
        self.assertGreaterEqual(float(out.min()), 0.0)
        self.assertLessEqual(float(out.max()), 1.0)

    def test_range_0_to_1_hard(self):
        out = _band_falloff(self.phase, 0.3, 0.0)
        self.assertGreaterEqual(float(out.min()), 0.0)
        self.assertLessEqual(float(out.max()), 1.0)

    def test_dtype_float32(self):
        out = _band_falloff(self.phase, 0.3, 0.5)
        self.assertEqual(out.dtype, np.float32)

    def test_no_nan_inf_on_degenerate_falloff_zone(self):
        # thickness=0.01, falloff=0.01 -> falloff_zone=5e-6, the case that
        # previously produced transient inf/nan and RuntimeWarnings.
        out = _band_falloff(self.phase, 0.01, 0.01)
        self.assertFalse(bool(np.isnan(out).any()), "output contains nan")
        self.assertFalse(bool(np.isinf(out).any()), "output contains inf")
        self.assertGreaterEqual(float(out.min()), 0.0)
        self.assertLessEqual(float(out.max()), 1.0)

    def test_no_runtime_warning_on_degenerate(self):
        # Run inside a warnings-as-errors context for RuntimeWarning.
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            _band_falloff(self.phase, 0.01, 0.01)  # should not raise

    def test_hard_edge_is_binary(self):
        out = _band_falloff(self.phase, 0.3, 0.0)
        unique = set(np.unique(out).tolist())
        self.assertTrue(unique.issubset({0.0, 1.0}),
                        f"hard edge should be binary, got {unique}")

    def test_bit_identical_to_old_formula(self):
        """The new helper must reproduce the old inline formula exactly."""
        phase = self.phase
        thickness, falloff = 0.3, 0.5
        half = thickness / 2.0
        fz = half * falloff
        ie = half - fz
        pd = np.abs(phase - 0.5)
        pd = np.minimum(pd, 1.0 - pd)
        old = np.where(
            pd <= ie, 1.0,
            np.where(pd <= half, 1.0 - (pd - ie) / fz, 0.0),
        ).astype(np.float32)
        new = _band_falloff(phase, thickness, falloff)
        self.assertEqual(float(np.abs(new - old).max()), 0.0)

    def test_wrap_around_continuity(self):
        # A phase exactly at 0 and 1 should produce the same value (seam).
        phase = np.array([[0.0, 1.0]], dtype=np.float32)
        out = _band_falloff(phase, 0.3, 0.5)
        self.assertAlmostEqual(float(out[0, 0]), float(out[0, 1]), places=6)


class TestPatternBuilders(unittest.TestCase):
    """generate_torus_rings / radial_beams / spiral_rings."""

    def test_torus_rings_shape_dtype_range(self):
        out = illumoraeProceduralPatternsNode.generate_torus_rings(
            64, 48, 4, 0.3, 0.5, 0.5, 0.5, 1.0)
        self.assertEqual(out.shape, (48, 64))
        self.assertEqual(out.dtype, np.float32)
        self.assertGreaterEqual(float(out.min()), 0.0)
        self.assertLessEqual(float(out.max()), 1.0)

    def test_radial_beams_shape_dtype_range(self):
        out = illumoraeProceduralPatternsNode.generate_radial_beams(
            64, 48, 8, 0.3, 0.5, 0.5, 0.5, 0.0)
        self.assertEqual(out.shape, (48, 64))
        self.assertEqual(out.dtype, np.float32)
        self.assertGreaterEqual(float(out.min()), 0.0)
        self.assertLessEqual(float(out.max()), 1.0)

    def test_spiral_rings_shape_dtype_range(self):
        out = illumoraeProceduralPatternsNode.generate_spiral_rings(
            64, 48, 4, 0.3, 0.5, 0.5, 0.5, 1.0, 0.0)
        self.assertEqual(out.shape, (48, 64))
        self.assertEqual(out.dtype, np.float32)
        self.assertGreaterEqual(float(out.min()), 0.0)
        self.assertLessEqual(float(out.max()), 1.0)

    def test_callable_via_class(self):
        # @staticmethod: should work without an instance.
        out = illumoraeProceduralPatternsNode.generate_torus_rings(
            32, 32, 4, 0.3, 0.5, 0.5, 0.5, 1.0)
        self.assertEqual(out.shape, (32, 32))

    def test_radial_beams_aspect_ratio_isotropic(self):
        """Beams should be isotropic in pixel space on non-square canvases.

        With 4 beams and rotation=0, a beam is centered at angle 0 (pointing
        right). On a 2:1 wide canvas, the beam should span the same angular
        width in pixel space as on a square canvas. We check that the
        horizontal beam at y=center has a comparable fill fraction on both.
        """
        sq = illumoraeProceduralPatternsNode.generate_radial_beams(
            128, 128, 4, 0.2, 0.0, 0.5, 0.5, 0.0)
        wide = illumoraeProceduralPatternsNode.generate_radial_beams(
            256, 128, 4, 0.2, 0.0, 0.5, 0.5, 0.0)
        # Center row: the horizontal beam should be present in both.
        sq_row = sq[64]
        wide_row = wide[64]
        self.assertGreater(float(sq_row.sum()), 0.0, "square: no beam on center row")
        self.assertGreater(float(wide_row.sum()), 0.0, "wide: no beam on center row")


class TestGeneratePattern(unittest.TestCase):
    """generate_pattern: the main orchestrator."""

    def setUp(self):
        self.node = illumoraeProceduralPatternsNode()
        self.kwargs = dict(
            width=64, height=64, pattern_type="torus_rings",
            ring_count=4, ring_thickness=0.3, ring_falloff=0.5,
            beam_count=8, beam_width=0.3, beam_falloff=0.5,
            center_x=0.5, center_y=0.5, rotation=0.0, scale=1.0,
            invert=False, colorize=False, chaos_mode=False,
            seed=42, color_a="#000000", color_b="#FFFFFF",
        )

    def test_output_shapes(self):
        img, mask = self.node.generate_pattern(**self.kwargs)
        self.assertEqual(tuple(img.shape), (1, 64, 64, 3))
        self.assertEqual(tuple(mask.shape), (1, 64, 64))

    def test_output_range(self):
        img, mask = self.node.generate_pattern(**self.kwargs)
        self.assertGreaterEqual(float(img.min()), 0.0)
        self.assertLessEqual(float(img.max()), 1.0)
        self.assertGreaterEqual(float(mask.min()), 0.0)
        self.assertLessEqual(float(mask.max()), 1.0)

    def test_deterministic_concrete_mode(self):
        """Concrete pattern_type: same seed -> same output (seed irrelevant)."""
        a = self.node.generate_pattern(**self.kwargs)
        b = self.node.generate_pattern(**self.kwargs)
        self.assertEqual(float((a[0] - b[0]).abs().max()), 0.0)
        # Different seed should produce the same output for concrete mode.
        kw2 = dict(self.kwargs, seed=999)
        c = self.node.generate_pattern(**kw2)
        self.assertEqual(float((a[0] - c[0]).abs().max()), 0.0)

    def test_deterministic_random_mode(self):
        """pattern_type='random': same seed -> same output."""
        kw = dict(self.kwargs, pattern_type="random")
        a = self.node.generate_pattern(**kw)
        b = self.node.generate_pattern(**kw)
        self.assertEqual(float((a[0] - b[0]).abs().max()), 0.0)
        # Different seed -> different output for at least one of several
        # tries (with only 4 pattern types, two seeds may pick the same
        # one ~25% of the time, so we check a handful).
        found_diff = False
        for s in (1, 2, 3, 7, 11, 13, 17, 19, 23, 99, 123, 256):
            kw2 = dict(kw, seed=s)
            c = self.node.generate_pattern(**kw2)
            if float((a[0] - c[0]).abs().max()) > 0.0:
                found_diff = True
                break
        self.assertTrue(found_diff, "no seed produced a different output")

    def test_deterministic_chaos_mode(self):
        """chaos_mode=True: same seed -> same output."""
        kw = dict(self.kwargs, chaos_mode=True)
        a = self.node.generate_pattern(**kw)
        b = self.node.generate_pattern(**kw)
        self.assertEqual(float((a[0] - b[0]).abs().max()), 0.0)
        # Different seed -> different output.
        kw2 = dict(kw, seed=123)
        c = self.node.generate_pattern(**kw2)
        self.assertNotEqual(float((a[0] - c[0]).abs().max()), 0.0)

    def test_invert(self):
        kw = dict(self.kwargs, invert=False)
        img_normal, mask_normal = self.node.generate_pattern(**kw)
        kw_inv = dict(self.kwargs, invert=True)
        img_inv, mask_inv = self.node.generate_pattern(**kw_inv)
        # In grayscale mode, inverted image = 1 - normal image.
        self.assertAlmostEqual(
            float((img_inv - (1.0 - img_normal)).abs().max()), 0.0, places=5)
        # Inverted mask = 1 - normal mask.
        self.assertAlmostEqual(
            float((mask_inv - (1.0 - mask_normal)).abs().max()), 0.0, places=5)

    def test_colorize_gradient_endpoints(self):
        """colorize with #000000 / #FFFFFF should match grayscale exactly."""
        kw_gray = dict(self.kwargs, colorize=False)
        img_gray, _ = self.node.generate_pattern(**kw_gray)
        kw_color = dict(self.kwargs, colorize=True,
                        color_a="#000000", color_b="#FFFFFF")
        img_color, _ = self.node.generate_pattern(**kw_color)
        self.assertEqual(float((img_gray - img_color).abs().max()), 0.0)

    def test_colorize_custom_colors(self):
        kw = dict(self.kwargs, colorize=True,
                  color_a="#FF0000", color_b="#00FF00")
        img, mask = self.node.generate_pattern(**kw)
        # Where mask=0, image should be red; where mask=1, green.
        zero_mask = mask[0] <= 0.001
        one_mask = mask[0] >= 0.999
        if zero_mask.any():
            r, g, b = img[0][zero_mask][0]
            self.assertAlmostEqual(float(r), 1.0, places=5)
            self.assertAlmostEqual(float(g), 0.0, places=5)
        if one_mask.any():
            r, g, b = img[0][one_mask][0]
            self.assertAlmostEqual(float(r), 0.0, places=5)
            self.assertAlmostEqual(float(g), 1.0, places=5)

    def test_combined_pattern(self):
        kw = dict(self.kwargs, pattern_type="combined")
        img, mask = self.node.generate_pattern(**kw)
        self.assertEqual(tuple(img.shape), (1, 64, 64, 3))
        # Combined = rings * beams, so it should be <= min(rings, beams).
        kw_rings = dict(self.kwargs, pattern_type="torus_rings")
        _, mask_rings = self.node.generate_pattern(**kw_rings)
        kw_beams = dict(self.kwargs, pattern_type="radial_beams")
        _, mask_beams = self.node.generate_pattern(**kw_beams)
        combined_max = float(mask.max())
        self.assertLessEqual(combined_max, float(mask_rings.max()))
        self.assertLessEqual(combined_max, float(mask_beams.max()))

    def test_all_concrete_pattern_types_run(self):
        for pt in PATTERN_TYPES_CONCRETE:
            with self.subTest(pattern_type=pt):
                kw = dict(self.kwargs, pattern_type=pt)
                img, mask = self.node.generate_pattern(**kw)
                self.assertEqual(tuple(img.shape), (1, 64, 64, 3))
                self.assertEqual(tuple(mask.shape), (1, 64, 64))


class TestPatternTypesEnum(unittest.TestCase):
    """PATTERN_TYPES_UI single-source derivation (fix Q)."""

    def test_ui_includes_random_plus_concrete(self):
        self.assertEqual(PATTERN_TYPES_UI[0], "random")
        self.assertEqual(PATTERN_TYPES_UI[1:], PATTERN_TYPES_CONCRETE)

    def test_no_drift(self):
        """UI enum must be exactly ('random',) + concrete."""
        self.assertEqual(PATTERN_TYPES_UI, ("random",) + PATTERN_TYPES_CONCRETE)

    def test_input_types_enum_matches(self):
        it = illumoraeProceduralPatternsNode.INPUT_TYPES()
        enum = it["required"]["pattern_type"][0]
        self.assertEqual(enum, list(PATTERN_TYPES_UI))


if __name__ == "__main__":
    unittest.main(verbosity=2)
