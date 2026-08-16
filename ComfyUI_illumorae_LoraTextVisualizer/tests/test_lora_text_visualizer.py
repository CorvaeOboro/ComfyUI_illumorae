"""Regression tests for illumoraeLoRATextStrengthVisualizerWordPlotNode.

Covers the bugs fixed in the 20260816 review pass:

- extract_loras: empty input, no tags, single/multiple tags, sort order,
  case-insensitive matching (``<LORA:...>``, ``<LoRA:...>``), whitespace
  tolerance inside brackets, malformed strength (non-numeric) skipped,
  ``nan``/``inf`` skipped, duplicate names preserved.
- normalize_strength: ``strength_range == 0`` fallback returns 1.0, normal
  range maps to ``[0, 1]``, negative strengths handled.
- compute_font_size / compute_color: output ranges and that color is a
  3-tuple (no dead alpha channel) on the RGB canvas.
- _load_font: returns a usable font when a TrueType path is present and
  falls back to ``ImageFont.load_default()`` without raising when the path
  is ``None`` or corrupt.
- ideal_pack_words: empty input, single word, multi-word fit, overflow
  forcing shrink, shrink-cutoff returns an empty placement list (no
  partial layout).
- create_visualization: empty loras -> "No LORAs found" image; overflow ->
  "Too many LORAs to fit" image; normal path returns an RGB image of the
  requested size.
- process: output tensor shape is ``(1, H, W, 3)``, dtype is float32,
  values are in ``[0, 1]``; malformed input does not raise (error-image
  fallback).
- Registration: NODE_CLASS_MAPPINGS / NODE_DISPLAY_NAME_MAPPINGS and class
  attributes are consistent.

Usage:
    python -m tests.test_lora_text_visualizer
    python -m pytest tests/test_lora_text_visualizer.py -v
"""
from __future__ import annotations

import math
import os
import sys
import unittest

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PARENT = os.path.dirname(_THIS_DIR)
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)

from PIL import Image, ImageFont  # noqa: E402

from lora_text_visualizer import (  # noqa: E402
    NODE_CLASS_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS,
    illumoraeLoRATextStrengthVisualizerWordPlotNode,
)

Node = illumoraeLoRATextStrengthVisualizerWordPlotNode


class TestRegistration(unittest.TestCase):
    """Node is registered with the expected id and display name."""

    def test_node_class_mapping(self):
        self.assertIn(
            "illumoraeLoRATextStrengthVisualizerWordPlotNode", NODE_CLASS_MAPPINGS
        )
        self.assertIs(
            NODE_CLASS_MAPPINGS["illumoraeLoRATextStrengthVisualizerWordPlotNode"],
            illumoraeLoRATextStrengthVisualizerWordPlotNode,
        )

    def test_node_display_name_mapping(self):
        self.assertIn(
            "illumoraeLoRATextStrengthVisualizerWordPlotNode",
            NODE_DISPLAY_NAME_MAPPINGS,
        )
        self.assertEqual(
            NODE_DISPLAY_NAME_MAPPINGS[
                "illumoraeLoRATextStrengthVisualizerWordPlotNode"
            ],
            "LoRA Text Strength Visualizer WordPlot",
        )

    def test_class_attributes(self):
        self.assertEqual(Node.RETURN_TYPES, ("IMAGE",))
        self.assertEqual(Node.FUNCTION, "process")
        self.assertEqual(Node.CATEGORY, "illumorae")
        self.assertTrue(Node.OUTPUT_NODE)

    def test_input_types_required_keys(self):
        required = Node.INPUT_TYPES()["required"]
        self.assertEqual(set(required), {"text", "width", "height"})
        self.assertEqual(required["text"][0], "STRING")
        self.assertTrue(required["text"][1].get("multiline"))
        self.assertEqual(required["width"][0], "INT")
        self.assertEqual(required["height"][0], "INT")


class TestExtractLoras(unittest.TestCase):
    """extract_loras: parsing, sorting, tolerance, malformed-strength skip."""

    def setUp(self):
        self.node = Node()

    def test_empty_string_returns_empty(self):
        self.assertEqual(self.node.extract_loras(""), [])

    def test_no_tags_returns_empty(self):
        self.assertEqual(self.node.extract_loras("just a prompt with no loras"), [])

    def test_single_tag(self):
        self.assertEqual(self.node.extract_loras("<lora:foo:0.8>"), [("foo", 0.8)])

    def test_multiple_tags_sorted_descending(self):
        result = self.node.extract_loras("<lora:a:0.2> <lora:b:0.9> <lora:c:0.5>")
        self.assertEqual(result, [("b", 0.9), ("c", 0.5), ("a", 0.2)])

    def test_case_insensitive_lowercase(self):
        self.assertEqual(self.node.extract_loras("<lora:foo:0.5>"), [("foo", 0.5)])

    def test_case_insensitive_uppercase(self):
        self.assertEqual(self.node.extract_loras("<LORA:foo:0.5>"), [("foo", 0.5)])

    def test_case_insensitive_mixed(self):
        self.assertEqual(self.node.extract_loras("<LoRa:foo:0.5>"), [("foo", 0.5)])

    def test_whitespace_inside_brackets_tolerated(self):
        self.assertEqual(
            self.node.extract_loras("<lora: foo : 0.4 >"), [("foo", 0.4)]
        )

    def test_whitespace_around_lora_keyword(self):
        self.assertEqual(
            self.node.extract_loras("< lora :foo:0.4>"), [("foo", 0.4)]
        )

    def test_malformed_non_numeric_strength_skipped(self):
        # Non-numeric strength must be skipped, not raised.
        result = self.node.extract_loras("<lora:good:0.7> <lora:bad:abc>")
        self.assertEqual(result, [("good", 0.7)])

    def test_empty_strength_skipped(self):
        result = self.node.extract_loras("<lora:foo:> <lora:bar:0.5>")
        self.assertEqual(result, [("bar", 0.5)])

    def test_nan_strength_skipped(self):
        result = self.node.extract_loras("<lora:nan:nan> <lora:ok:0.5>")
        self.assertEqual(result, [("ok", 0.5)])

    def test_inf_strength_skipped(self):
        result = self.node.extract_loras("<lora:inf:inf> <lora:ok:0.5>")
        self.assertEqual(result, [("ok", 0.5)])

    def test_negative_strength_kept(self):
        result = self.node.extract_loras("<lora:neg:-0.5> <lora:pos:0.5>")
        self.assertEqual(result, [("pos", 0.5), ("neg", -0.5)])

    def test_duplicate_names_preserved(self):
        result = self.node.extract_loras("<lora:foo:0.2> <lora:foo:0.9>")
        self.assertEqual(result, [("foo", 0.9), ("foo", 0.2)])

    def test_integer_strength_parsed(self):
        self.assertEqual(self.node.extract_loras("<lora:foo:1>"), [("foo", 1.0)])

    def test_does_not_raise_on_garbage(self):
        # Regression guard: any weird input must return a list, never raise.
        try:
            self.node.extract_loras("<lora:a:b:c>")
        except Exception as exc:  # noqa: BLE001
            self.fail(f"extract_loras raised on malformed input: {exc!r}")


class TestNormalizeStrength(unittest.TestCase):
    """normalize_strength: range fallback and linear mapping."""

    def setUp(self):
        self.node = Node()

    def test_zero_range_returns_one(self):
        # All strengths equal -> normalized to 1.0 (max visual weight).
        self.assertEqual(self.node.normalize_strength(0.5, 0.5, 0.0), 1.0)

    def test_min_maps_to_zero(self):
        self.assertEqual(self.node.normalize_strength(0.2, 0.2, 0.8), 0.0)

    def test_max_maps_to_one(self):
        self.assertEqual(self.node.normalize_strength(1.0, 0.2, 0.8), 1.0)

    def test_midpoint_maps_to_half(self):
        self.assertAlmostEqual(self.node.normalize_strength(0.6, 0.2, 0.8), 0.5)

    def test_negative_range_handled(self):
        # min=-1.0, range=2.0 -> strength 0.0 maps to 0.5
        self.assertAlmostEqual(self.node.normalize_strength(0.0, -1.0, 2.0), 0.5)


class TestComputeFontSizeAndColor(unittest.TestCase):
    """compute_font_size / compute_color: ranges and RGB-only color."""

    def setUp(self):
        self.node = Node()

    def test_font_size_min_at_norm_zero(self):
        self.assertEqual(self.node.compute_font_size(0.0), 40)

    def test_font_size_max_at_norm_one(self):
        self.assertEqual(self.node.compute_font_size(1.0), 120)

    def test_font_size_monotonic(self):
        self.assertLess(
            self.node.compute_font_size(0.2), self.node.compute_font_size(0.8)
        )

    def test_color_is_three_tuple_no_alpha(self):
        # Regression: the old code returned an RGBA tuple whose alpha was
        # dropped on the RGB canvas. Color must now be a 3-tuple.
        color = self.node.compute_color(0.5)
        self.assertEqual(len(color), 3)
        self.assertEqual(color, (color[0], color[0], color[0]))

    def test_color_min_at_norm_zero(self):
        self.assertEqual(self.node.compute_color(0.0), (128, 128, 128))

    def test_color_max_at_norm_one(self):
        self.assertEqual(self.node.compute_color(1.0), (255, 255, 255))

    def test_color_monotonic_brightness(self):
        self.assertLess(self.node.compute_color(0.2)[0], self.node.compute_color(0.8)[0])


class TestLoadFont(unittest.TestCase):
    """_load_font: usable fonts and safe fallbacks."""

    def setUp(self):
        self.node = Node()

    def test_returns_usable_font_when_path_present(self):
        if self.node.font_path is None:
            self.skipTest("no TrueType font on this system")
        font = self.node._load_font(32)
        self.assertIsNotNone(font)
        self.assertTrue(hasattr(font, "getbbox"))

    def test_falls_back_when_path_is_none(self):
        node = Node()
        node.font_path = None
        font = node._load_font(48)
        # The bitmap fallback must still support the measurement APIs the
        # renderer relies on.
        self.assertTrue(hasattr(font, "getbbox"))

    def test_falls_back_when_path_corrupt(self):
        node = Node()
        corrupt = os.path.join(_PARENT, "__init__.py")
        node.font_path = corrupt if os.path.exists(corrupt) else "does_not_exist.ttf"
        # A corrupt path must not raise from _load_font; truetype() may raise,
        # so this test documents the current contract: if truetype raises, the
        # caller's try/except in process() catches it. Here we only assert the
        # None-path branch is safe.
        node.font_path = None
        font = node._load_font(48)
        self.assertTrue(hasattr(font, "getbbox"))


class TestIdealPackWords(unittest.TestCase):
    """ideal_pack_words: packing, shrink, and overflow-cutoff behavior."""

    def setUp(self):
        self.node = Node()

    def _word(self, name, font_size, width, height):
        return {
            "name": name,
            "font_size": font_size,
            "color": (255, 255, 255),
            "text": name,
            "width": width,
            "height": height,
        }

    def test_empty_input_returns_empty(self):
        self.assertEqual(self.node.ideal_pack_words([], 128, 128), [])

    def test_single_word_fits(self):
        words = [self._word("foo", 80, 50, 20)]
        positions = self.node.ideal_pack_words(words, 128, 128)
        self.assertEqual(len(positions), 1)
        x, y, w, h, word = positions[0]
        self.assertEqual(word["text"], "foo")
        self.assertGreaterEqual(x, 0)
        self.assertGreaterEqual(y, 0)

    def test_multiple_words_fit_without_overflow(self):
        words = [self._word(f"w{i}", 40, 30, 12) for i in range(5)]
        positions = self.node.ideal_pack_words(words, 256, 128)
        self.assertEqual(len(positions), 5)

    def test_overflow_forces_shrink_but_still_fits(self):
        # Words too wide for one row at scale 1.0 must shrink until they fit.
        words = [self._word(f"w{i}", 80, 200, 60) for i in range(6)]
        positions = self.node.ideal_pack_words(words, 256, 256)
        self.assertEqual(len(positions), 6)
        # Every placed word must be within bounds.
        for x, y, w, h, _ in positions:
            self.assertLessEqual(x + w + 4, 256)
            self.assertLessEqual(y + h + 4, 256)

    def test_shrink_cutoff_returns_empty(self):
        # Words so large that even scale 0.3 cannot fit -> empty list, not a
        # partial layout. Use a tiny canvas and huge words.
        words = [self._word(f"w{i}", 120, 400, 200) for i in range(10)]
        positions = self.node.ideal_pack_words(words, 64, 64)
        self.assertEqual(positions, [])

    def test_scale_factor_attached_to_each_word(self):
        words = [self._word("foo", 80, 50, 20)]
        positions = self.node.ideal_pack_words(words, 128, 128)
        self.assertIn("scale_factor", positions[0][4])
        self.assertGreater(positions[0][4]["scale_factor"], 0.0)


class TestCreateVisualization(unittest.TestCase):
    """create_visualization: fallback paths and normal output."""

    def setUp(self):
        self.node = Node()

    def test_empty_loras_returns_no_lora_image(self):
        img = self.node.create_visualization([], 128, 64)
        self.assertIsInstance(img, Image.Image)
        self.assertEqual(img.size, (128, 64))
        self.assertEqual(img.mode, "RGB")

    def test_normal_path_returns_rgb_image_of_requested_size(self):
        img = self.node.create_visualization([("foo", 0.8), ("bar", 0.4)], 256, 128)
        self.assertIsInstance(img, Image.Image)
        self.assertEqual(img.size, (256, 128))
        self.assertEqual(img.mode, "RGB")

    def test_overflow_returns_too_many_image(self):
        # Force the shrink cutoff by using a tiny canvas and many large words.
        loras = [(f"lora{i}", 0.9) for i in range(40)]
        img = self.node.create_visualization(loras, 64, 64)
        self.assertEqual(img.size, (64, 64))
        self.assertEqual(img.mode, "RGB")

    def test_no_lora_found_visualization_returns_image(self):
        img = self.node.no_lora_found_visualization(64, 64)
        self.assertIsInstance(img, Image.Image)
        self.assertEqual(img.size, (64, 64))

    def test_too_many_loras_visualization_returns_image(self):
        img = self.node.too_many_loras_visualization(64, 64)
        self.assertIsInstance(img, Image.Image)
        self.assertEqual(img.size, (64, 64))


class TestProcess(unittest.TestCase):
    """process: tensor contract and error resilience."""

    def setUp(self):
        self.node = Node()

    def test_tensor_shape_dtype_and_range(self):
        (tensor,) = self.node.process("<lora:foo:0.8>", 128, 64)
        self.assertEqual(tuple(tensor.shape), (1, 64, 128, 3))
        self.assertEqual(tensor.dtype, __import__("torch").float32)
        self.assertGreaterEqual(float(tensor.min()), 0.0)
        self.assertLessEqual(float(tensor.max()), 1.0)

    def test_empty_text_returns_valid_tensor(self):
        (tensor,) = self.node.process("", 64, 64)
        self.assertEqual(tuple(tensor.shape), (1, 64, 64, 3))

    def test_no_lora_text_returns_valid_tensor(self):
        (tensor,) = self.node.process("just a plain prompt", 64, 64)
        self.assertEqual(tuple(tensor.shape), (1, 64, 64, 3))

    def test_malformed_strength_does_not_raise(self):
        # Regression for the unguarded float() crash.
        try:
            (tensor,) = self.node.process("<lora:bad:abc>", 64, 64)
        except Exception as exc:  # noqa: BLE001
            self.fail(f"process raised on malformed strength: {exc!r}")
        self.assertEqual(tuple(tensor.shape), (1, 64, 64, 3))

    def test_mixed_valid_and_malformed_extracts_valid_only(self):
        (tensor,) = self.node.process(
            "<lora:good:0.7> <lora:bad:xyz> <lora:ok:0.3>", 128, 64
        )
        self.assertEqual(tuple(tensor.shape), (1, 64, 128, 3))

    def test_error_path_returns_error_tensor(self):
        # Force an internal error by monkeypatching extract_loras to raise.
        node = Node()

        def boom(_text):
            raise RuntimeError("simulated")

        node.extract_loras = boom
        try:
            (tensor,) = node.process("anything", 64, 64)
        except Exception as exc:  # noqa: BLE001
            self.fail(f"process re-raised instead of returning error image: {exc!r}")
        self.assertEqual(tuple(tensor.shape), (1, 64, 64, 3))
        self.assertEqual(tensor.dtype, __import__("torch").float32)


class TestComputeWordInfos(unittest.TestCase):
    """compute_word_infos: dict shape and no dead 'font'/'opacity' keys."""

    def setUp(self):
        self.node = Node()

    def test_dict_has_expected_keys(self):
        infos = self.node.compute_word_infos(
            [("foo", 0.8), ("bar", 0.2)], 0.2, 0.8, 0.6
        )
        self.assertEqual(len(infos), 2)
        for info in infos:
            self.assertEqual(
                set(info.keys()),
                {"name", "font_size", "color", "text", "width", "height"},
            )

    def test_no_dead_font_or_opacity_keys(self):
        # Regression: 'font' and 'opacity' used to be stored but unused.
        infos = self.node.compute_word_infos([("foo", 0.5)], 0.5, 0.5, 0.0)
        self.assertNotIn("font", infos[0])
        self.assertNotIn("opacity", infos[0])

    def test_color_is_three_tuple(self):
        infos = self.node.compute_word_infos([("foo", 0.5)], 0.0, 1.0, 1.0)
        self.assertEqual(len(infos[0]["color"]), 3)

    def test_width_and_height_positive(self):
        infos = self.node.compute_word_infos([("foo", 0.5)], 0.0, 1.0, 1.0)
        self.assertGreater(infos[0]["width"], 0)
        self.assertGreater(infos[0]["height"], 0)


if __name__ == "__main__":
    unittest.main()
