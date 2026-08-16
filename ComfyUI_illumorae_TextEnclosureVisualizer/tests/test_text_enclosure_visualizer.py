"""
Tests for illumoraeEnclosureVisualizerNode.

Covers the bug fixes from the 2026-08-16 code review:

- parse_enclosures: depth tracking, structured warnings, max_depth return,
  newline skipping, deep-nesting-once, closing-paren warning inheritance,
  O(1) unclosed-paren marking
- get_color_for_depth: error/warning/depth color selection
- _load_font / _measure_char_width: cross-platform font fallback, width > 0
- create_visualization_image: image width accounts for warnings, no clipping
- visualize: tensor shape [1,H,W,C], report content, issues_text categorization,
  OUTPUT_NODE = True, no *args/**kwargs

Usage:
    cd ComfyUI_illumorae_TextEnclosureVisualizer
    python -m unittest tests.test_text_enclosure_visualizer -v
"""
import os
import sys
import unittest

# Add parent directory to path so we can import the node module
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from text_enclosure_visualizer import (
    illumoraeEnclosureVisualizerNode,
    NODE_CLASS_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS,
)


class TestParseEnclosures(unittest.TestCase):
    """Tests for parse_enclosures — the core scanner."""

    def setUp(self):
        self.node = illumoraeEnclosureVisualizerNode()

    def test_returns_three_values(self):
        """parse_enclosures returns (char_info, warnings, max_depth) — 3 values."""
        char_info, warnings, max_depth = self.node.parse_enclosures("hello")
        self.assertIsInstance(char_info, list)
        self.assertIsInstance(warnings, list)
        self.assertIsInstance(max_depth, int)

    def test_no_parenthesis(self):
        """Plain text: all chars at depth 0, no warnings, max_depth 0."""
        char_info, warnings, max_depth = self.node.parse_enclosures("hello world")
        self.assertEqual(max_depth, 0)
        self.assertEqual(warnings, [])
        for info in char_info:
            self.assertEqual(info['depth'], 0)
            self.assertFalse(info['is_error'])
            self.assertFalse(info['is_warning'])

    def test_single_level_nesting(self):
        """(abc) → depth 1 for parens, depth 0 outside, max_depth 1."""
        char_info, warnings, max_depth = self.node.parse_enclosures("a(bc)d")
        self.assertEqual(max_depth, 1)
        self.assertEqual(warnings, [])

        # a: depth 0, (: depth 1, b: depth 1, c: depth 1, ): depth 1, d: depth 0
        depths = [info['depth'] for info in char_info]
        self.assertEqual(depths, [0, 1, 1, 1, 1, 0])

        # Closing paren should be at depth 1 (the level it closes)
        paren_close = char_info[4]
        self.assertEqual(paren_close['char'], ')')
        self.assertEqual(paren_close['depth'], 1)

    def test_multi_level_nesting(self):
        """(a(b)c) → max_depth 2, correct depths."""
        char_info, warnings, max_depth = self.node.parse_enclosures("(a(b)c)")
        self.assertEqual(max_depth, 2)
        self.assertEqual(warnings, [])
        depths = [info['depth'] for info in char_info]
        # ( : 1, a : 1, ( : 2, b : 2, ) : 2, c : 1, ) : 1
        self.assertEqual(depths, [1, 1, 2, 2, 2, 1, 1])

    def test_hanging_closing_paren(self):
        """Extra ) with empty stack → is_error True, hanging warning."""
        char_info, warnings, max_depth = self.node.parse_enclosures("abc)")
        self.assertEqual(max_depth, 0)
        self.assertEqual(len(warnings), 1)
        self.assertEqual(warnings[0][0], "hanging")
        self.assertIn("Hanging closing", warnings[0][1])

        close_info = char_info[3]
        self.assertEqual(close_info['char'], ')')
        self.assertTrue(close_info['is_error'])

    def test_unclosed_opening_paren(self):
        """Unclosed ( → is_error True on the opener, hanging warning."""
        char_info, warnings, max_depth = self.node.parse_enclosures("a(bc")
        self.assertEqual(max_depth, 1)
        self.assertEqual(len(warnings), 1)
        self.assertEqual(warnings[0][0], "hanging")
        self.assertIn("Unclosed opening", warnings[0][1])

        # The opener at index 1 should be marked as error
        opener = char_info[1]
        self.assertEqual(opener['char'], '(')
        self.assertTrue(opener['is_error'])

    def test_multiple_unclosed_opening(self):
        """Multiple unclosed ( → all marked as error, all get warnings."""
        char_info, warnings, max_depth = self.node.parse_enclosures("(((")
        self.assertEqual(max_depth, 3)
        self.assertEqual(len(warnings), 3)
        for info in char_info:
            self.assertTrue(info['is_error'])

    def test_deep_nesting_warns_once(self):
        """Depth > 3 produces exactly one nesting warning per threshold crossing."""
        # ((((a)))) → depth reaches 4, one warning
        char_info, warnings, max_depth = self.node.parse_enclosures("((((a))))")
        self.assertEqual(max_depth, 4)
        nesting_warnings = [w for cat, w in warnings if cat == "nesting"]
        self.assertEqual(len(nesting_warnings), 1)

    def test_deep_nesting_five_levels_still_one_warning(self):
        """(((((a))))) → depth 5, still only one nesting warning (at threshold)."""
        char_info, warnings, max_depth = self.node.parse_enclosures("(((((a)))))")
        self.assertEqual(max_depth, 5)
        nesting_warnings = [w for cat, w in warnings if cat == "nesting"]
        self.assertEqual(len(nesting_warnings), 1)

    def test_closing_paren_inherits_deep_warning(self):
        """Closing paren at deep level should have is_warning=True."""
        char_info, warnings, max_depth = self.node.parse_enclosures("((((a))))")
        # Find the closing parens (last 4 chars)
        closers = [info for info in char_info if info['char'] == ')']
        self.assertEqual(len(closers), 4)
        # The first two closers close depth 4 and 3 → depth 4 closer should be warning
        # Opener at depth 4 was deep, so its matching closer inherits is_warning
        deep_closer = closers[0]
        self.assertTrue(deep_closer['is_warning'])

    def test_closing_paren_at_normal_depth_no_warning(self):
        """Closing paren at depth <= 3 should not be a warning."""
        char_info, warnings, max_depth = self.node.parse_enclosures("(a(b(c)d)e)")
        self.assertEqual(max_depth, 3)
        for info in char_info:
            self.assertFalse(info['is_warning'])

    def test_newlines_skipped_in_char_info(self):
        """Newline characters should not appear in char_info."""
        char_info, warnings, max_depth = self.node.parse_enclosures("a\nb")
        chars = [info['char'] for info in char_info]
        self.assertNotIn('\n', chars)
        self.assertEqual(chars, ['a', 'b'])

    def test_newline_with_parens(self):
        """Parenthesis tracking works across newlines; positions reflect original text."""
        char_info, warnings, max_depth = self.node.parse_enclosures("(a\nb)")
        self.assertEqual(max_depth, 1)
        self.assertEqual(warnings, [])
        # char_info should be: ( a b )  (newline skipped)
        chars = [info['char'] for info in char_info]
        self.assertEqual(chars, ['(', 'a', 'b', ')'])

    def test_empty_string(self):
        """Empty text → empty char_info, no warnings, max_depth 0."""
        char_info, warnings, max_depth = self.node.parse_enclosures("")
        self.assertEqual(char_info, [])
        self.assertEqual(warnings, [])
        self.assertEqual(max_depth, 0)

    def test_warning_categories_are_valid(self):
        """All warning categories must be 'hanging' or 'nesting'."""
        char_info, warnings, max_depth = self.node.parse_enclosures("((((")
        for cat, msg in warnings:
            self.assertIn(cat, ("hanging", "nesting"))

    def test_char_info_pos_reflects_original_index(self):
        """pos field stores the original index in the input text (including newlines)."""
        char_info, warnings, max_depth = self.node.parse_enclosures("a\n(b")
        # ( is at original index 2
        paren_info = char_info[1]
        self.assertEqual(paren_info['char'], '(')
        self.assertEqual(paren_info['pos'], 2)


class TestGetColorForDepth(unittest.TestCase):
    """Tests for get_color_for_depth."""

    def setUp(self):
        self.node = illumoraeEnclosureVisualizerNode()

    def test_error_takes_priority(self):
        """Error color is returned even if is_warning is also True."""
        color = self.node.get_color_for_depth(5, is_error=True, is_warning=True)
        self.assertEqual(color, self.node.error_color)

    def test_warning_takes_priority_over_depth(self):
        """Warning color is returned when is_warning=True and not error."""
        color = self.node.get_color_for_depth(5, is_error=False, is_warning=True)
        self.assertEqual(color, self.node.warning_color)

    def test_depth_zero_returns_base(self):
        color = self.node.get_color_for_depth(0, is_error=False, is_warning=False)
        self.assertEqual(color, self.node.base_color)

    def test_depth_cycles_through_muted_colors(self):
        """Depth 1→colors[0], depth 2→colors[1], ..., depth 6→colors[0] (cycle)."""
        for depth in range(1, 10):
            color = self.node.get_color_for_depth(depth, is_error=False, is_warning=False)
            expected = self.node.muted_colors[(depth - 1) % len(self.node.muted_colors)]
            self.assertEqual(color, expected)


class TestFontLoading(unittest.TestCase):
    """Tests for _load_font and _measure_char_width (cross-platform fixes)."""

    def setUp(self):
        self.node = illumoraeEnclosureVisualizerNode()

    def test_load_font_returns_a_font(self):
        """_load_font always returns some font object, never raises."""
        font = self.node._load_font(24)
        self.assertIsNotNone(font)

    def test_load_font_different_sizes(self):
        """Loading at different sizes should not crash."""
        for size in [12, 24, 48, 72]:
            font = self.node._load_font(size)
            self.assertIsNotNone(font)

    def test_measure_char_width_positive(self):
        """_measure_char_width must return a positive value for any font."""
        font = self.node._load_font(24)
        width = self.node._measure_char_width(font, 24)
        self.assertGreater(width, 0)

    def test_measure_char_width_fallback(self):
        """Fallback estimate (font_size * 0.6) is used when getlength fails."""
        class FakeFont:
            def getlength(self, s):
                raise AttributeError("no getlength")
        width = self.node._measure_char_width(FakeFont(), 24)
        self.assertAlmostEqual(width, 24 * 0.6)


class TestCreateVisualizationImage(unittest.TestCase):
    """Tests for create_visualization_image — rendering and dimensions."""

    def setUp(self):
        self.node = illumoraeEnclosureVisualizerNode()

    def test_returns_pil_image(self):
        from PIL import Image
        char_info, warnings, _ = self.node.parse_enclosures("hello")
        img = self.node.create_visualization_image(
            "hello", char_info, warnings, font_size=24, line_height=1.5,
            padding=20, background_color="dark"
        )
        self.assertIsInstance(img, Image.Image)

    def test_image_size_positive(self):
        char_info, warnings, _ = self.node.parse_enclosures("hello")
        img = self.node.create_visualization_image(
            "hello", char_info, warnings, font_size=24, line_height=1.5,
            padding=20, background_color="dark"
        )
        self.assertGreater(img.width, 0)
        self.assertGreater(img.height, 0)

    def test_dark_background_pixel(self):
        """Top-left pixel should be the dark bg color."""
        char_info, warnings, _ = self.node.parse_enclosures("x")
        img = self.node.create_visualization_image(
            "x", char_info, warnings, font_size=24, line_height=1.5,
            padding=20, background_color="dark"
        )
        self.assertEqual(img.getpixel((0, 0)), (30, 30, 35))

    def test_light_background_pixel(self):
        char_info, warnings, _ = self.node.parse_enclosures("x")
        img = self.node.create_visualization_image(
            "x", char_info, warnings, font_size=24, line_height=1.5,
            padding=20, background_color="light"
        )
        self.assertEqual(img.getpixel((0, 0)), (240, 240, 245))

    def test_image_width_accounts_for_warnings(self):
        """Image width must be wide enough to fit the longest warning string.

        This is the fix for issue 3.5 — warnings were being clipped.
        """
        # A short text with a long warning message
        text = ")"
        char_info, warnings, _ = self.node.parse_enclosures(text)
        self.assertTrue(len(warnings) > 0)

        img = self.node.create_visualization_image(
            text, char_info, warnings, font_size=24, line_height=1.5,
            padding=20, background_color="dark"
        )

        # The warning text "- Hanging closing parenthesis at position 0" is ~45 chars.
        # At font_size 20 (warning font), char_width ~12px → ~540px + padding.
        # Image width must be at least that, not just the 1-char text width.
        self.assertGreater(img.width, 100)

    def test_multi_line_image_height(self):
        """Multi-line text should produce taller image than single-line."""
        char_info1, warnings1, _ = self.node.parse_enclosures("hello")
        img1 = self.node.create_visualization_image(
            "hello", char_info1, warnings1, font_size=24, line_height=1.5,
            padding=20, background_color="dark"
        )
        char_info2, warnings2, _ = self.node.parse_enclosures("hello\nworld\nfoo")
        img2 = self.node.create_visualization_image(
            "hello\nworld\nfoo", char_info2, warnings2, font_size=24, line_height=1.5,
            padding=20, background_color="dark"
        )
        self.assertGreater(img2.height, img1.height)

    def test_warnings_add_height(self):
        """Image with warnings should be taller than without (for same text)."""
        text_clean = "hello"
        text_bad = "hello)"

        char_info_c, warnings_c, _ = self.node.parse_enclosures(text_clean)
        img_clean = self.node.create_visualization_image(
            text_clean, char_info_c, warnings_c, font_size=24, line_height=1.5,
            padding=20, background_color="dark"
        )

        char_info_b, warnings_b, _ = self.node.parse_enclosures(text_bad)
        img_bad = self.node.create_visualization_image(
            text_bad, char_info_b, warnings_b, font_size=24, line_height=1.5,
            padding=20, background_color="dark"
        )
        self.assertGreater(img_bad.height, img_clean.height)


class TestVisualize(unittest.TestCase):
    """Tests for the main visualize() entry point."""

    def setUp(self):
        self.node = illumoraeEnclosureVisualizerNode()

    def test_returns_three_outputs(self):
        result = self.node.visualize("hello")
        self.assertEqual(len(result), 3)

    def test_tensor_shape(self):
        """Image tensor must be [1, H, W, C] (ComfyUI IMAGE format)."""
        img_tensor, report, issues = self.node.visualize("hello")
        self.assertEqual(img_tensor.ndim, 4)
        self.assertEqual(img_tensor.shape[0], 1)  # batch
        self.assertEqual(img_tensor.shape[3], 3)  # RGB channels

    def test_tensor_dtype_and_range(self):
        """Tensor should be float32 in [0, 1]."""
        img_tensor, _, _ = self.node.visualize("hello")
        self.assertEqual(img_tensor.dtype, torch.float32)
        self.assertGreaterEqual(img_tensor.min().item(), 0.0)
        self.assertLessEqual(img_tensor.max().item(), 1.0)

    def test_report_contains_header(self):
        _, report, _ = self.node.visualize("hello")
        self.assertIn("=== ENCLOSURE ANALYSIS ===", report)

    def test_report_contains_max_depth(self):
        _, report, _ = self.node.visualize("(a(b)c)")
        self.assertIn("Max nesting depth: 2", report)

    def test_report_no_issues(self):
        _, report, issues = self.node.visualize("(a)")
        self.assertIn("No issues found", report)
        self.assertEqual(issues, "No issues found.")

    def test_report_with_hanging_paren(self):
        _, report, issues = self.node.visualize("a)")
        self.assertIn("WARNINGS", report)
        self.assertIn("Hanging", report)
        self.assertIn("HANGING/UNCLOSED PARENTHESIS", issues)

    def test_report_with_unclosed_paren(self):
        _, report, issues = self.node.visualize("a(b")
        self.assertIn("Unclosed", report)
        self.assertIn("HANGING/UNCLOSED PARENTHESIS", issues)

    def test_report_with_deep_nesting(self):
        _, report, issues = self.node.visualize("((((a))))")
        self.assertIn("Deep nesting", report)
        self.assertIn("MULTIPLE NESTED ENCLOSURES", issues)

    def test_issues_text_categorization_hanging(self):
        """Hanging/unclosed parens go to the hanging category."""
        _, _, issues = self.node.visualize("a) b(c")
        self.assertIn("HANGING/UNCLOSED PARENTHESIS", issues)
        self.assertIn("Hanging closing", issues)
        self.assertIn("Unclosed opening", issues)

    def test_issues_text_categorization_nesting(self):
        """Deep nesting goes to the nesting category."""
        _, _, issues = self.node.visualize("((((a))))")
        self.assertIn("MULTIPLE NESTED ENCLOSURES", issues)
        self.assertIn("Deep nesting", issues)

    def test_empty_text(self):
        """Empty text should not crash and produce valid outputs."""
        img_tensor, report, issues = self.node.visualize("")
        self.assertEqual(img_tensor.ndim, 4)
        self.assertIn("Max nesting depth: 0", report)
        self.assertEqual(issues, "No issues found.")

    def test_background_color_param(self):
        """Both background_color options should work without error."""
        for bg in ("dark", "light"):
            img_tensor, _, _ = self.node.visualize("hello", background_color=bg)
            self.assertEqual(img_tensor.ndim, 4)

    def test_custom_font_size(self):
        """Various font sizes within range should work."""
        for size in [12, 24, 48, 72]:
            img_tensor, _, _ = self.node.visualize("hello", font_size=size)
            self.assertEqual(img_tensor.ndim, 4)

    def test_no_var_args_in_signature(self):
        """visualize() should not accept *args or **kwargs (issue 3.9)."""
        import inspect
        sig = inspect.signature(self.node.visualize)
        params = sig.parameters
        self.assertNotIn("args", params)
        self.assertNotIn("kwargs", params)

    def test_multi_line_text(self):
        """Multi-line text with parens across lines should work."""
        text = "(hello\nworld)"
        img_tensor, report, issues = self.node.visualize(text)
        self.assertEqual(img_tensor.ndim, 4)
        self.assertIn("Max nesting depth: 1", report)


class TestNodeRegistration(unittest.TestCase):
    """Tests for ComfyUI node registration metadata."""

    def test_node_class_mappings_has_node(self):
        self.assertIn("illumoraeEnclosureVisualizerNode", NODE_CLASS_MAPPINGS)

    def test_node_display_name_mappings_has_node(self):
        self.assertIn("illumoraeEnclosureVisualizerNode", NODE_DISPLAY_NAME_MAPPINGS)

    def test_display_name(self):
        self.assertEqual(
            NODE_DISPLAY_NAME_MAPPINGS["illumoraeEnclosureVisualizerNode"],
            "Enclosure Visualizer",
        )

    def test_output_node_is_true(self):
        """OUTPUT_NODE should be True for a standalone visualizer (issue 3.14)."""
        self.assertTrue(illumoraeEnclosureVisualizerNode.OUTPUT_NODE)

    def test_return_types_and_names_length_match(self):
        cls = illumoraeEnclosureVisualizerNode
        self.assertEqual(len(cls.RETURN_TYPES), len(cls.RETURN_NAMES))
        self.assertEqual(len(cls.RETURN_TYPES), 3)

    def test_function_name(self):
        self.assertEqual(illumoraeEnclosureVisualizerNode.FUNCTION, "visualize")

    def test_category(self):
        self.assertEqual(illumoraeEnclosureVisualizerNode.CATEGORY, "illumorae")

    def test_input_types_structure(self):
        """INPUT_TYPES should have required, optional, and no empty hidden dict."""
        inputs = illumoraeEnclosureVisualizerNode.INPUT_TYPES()
        self.assertIn("required", inputs)
        self.assertIn("optional", inputs)
        self.assertNotIn("hidden", inputs)

    def test_required_inputs(self):
        inputs = illumoraeEnclosureVisualizerNode.INPUT_TYPES()
        required = inputs["required"]
        self.assertIn("text", required)
        self.assertIn("font_size", required)
        self.assertIn("line_height", required)
        self.assertIn("padding", required)

    def test_optional_background_color(self):
        inputs = illumoraeEnclosureVisualizerNode.INPUT_TYPES()
        bg = inputs["optional"]["background_color"]
        self.assertEqual(bg[0], ["dark", "light"])


if __name__ == "__main__":
    unittest.main()
