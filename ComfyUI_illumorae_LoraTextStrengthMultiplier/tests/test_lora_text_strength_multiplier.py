"""Regression tests for illumoraeLoRAStrengthMultiplierOnTextNode.

Covers the bugs fixed in the 20260816 review pass:

- parse_lora_syntax: integer/decimal/leading-dot/trailing-dot strengths,
  negative strengths, two-strength <lora:name:unet:clip> syntax, duplicate
  names preserved as separate records, names preserved verbatim (no strip),
  malformed tags skipped, empty input.
- format_lora_tag: single vs two-strength emission, :.4f precision.
- process: surrounding text preserved (M1), duplicate names preserved (M2),
  two-strength syntax multiplied on both unet and clip (M3), negative
  strengths accepted (M4), individual cap clamping (upper bound, negatives
  unaffected), total cap proportional scaling, cap=0 zeros out (m4),
  no-tag input returned unchanged, empty input, debug_prints does not crash,
  output is a 1-tuple of str.
- Registration: NODE_CLASS_MAPPINGS / NODE_DISPLAY_NAME_MAPPINGS / class
  attributes match expectations.

Usage:
    python -m tests.test_lora_text_strength_multiplier
    python -m pytest tests/test_lora_text_strength_multiplier.py -v
"""
from __future__ import annotations

import os
import re
import sys
import unittest

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PARENT = os.path.dirname(_THIS_DIR)
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)

from lora_text_strength_multiplier import (  # noqa: E402
    NODE_CLASS_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS,
    illumoraeLoRAStrengthMultiplierOnTextNode,
)

Node = illumoraeLoRAStrengthMultiplierOnTextNode

# Convenience kwargs for the no-cap case (the common path in tests).
_NO_CAPS = dict(
    multiplier=1.0,
    individual_cap_enabled=False,
    individual_cap=1.0,
    total_cap_enabled=False,
    total_cap=1.0,
)


def _strengths(text: str):
    """Extract all strength tuples (unet[, clip]) from a rewritten string."""
    out = []
    for m in re.finditer(r"<lora:([^:<>]+):([0-9.]+)(?::([0-9.]+))?>", text):
        vals = (float(m.group(2)), float(m.group(3)) if m.group(3) else None)
        out.append((m.group(1), vals))
    return out


class TestRegistration(unittest.TestCase):
    """Node is registered with the expected id and display name."""

    def test_node_class_mapping(self):
        self.assertIn("illumoraeLoRAStrengthMultiplierOnTextNode", NODE_CLASS_MAPPINGS)
        self.assertIs(
            NODE_CLASS_MAPPINGS["illumoraeLoRAStrengthMultiplierOnTextNode"],
            illumoraeLoRAStrengthMultiplierOnTextNode,
        )

    def test_node_display_name_mapping(self):
        self.assertIn("illumoraeLoRAStrengthMultiplierOnTextNode", NODE_DISPLAY_NAME_MAPPINGS)
        self.assertEqual(
            NODE_DISPLAY_NAME_MAPPINGS["illumoraeLoRAStrengthMultiplierOnTextNode"],
            "LoRA Strength Multiplier on Text",
        )

    def test_class_attributes(self):
        self.assertEqual(Node.RETURN_TYPES, ("STRING",))
        self.assertEqual(Node.RETURN_NAMES, ("modified_text",))
        self.assertEqual(Node.FUNCTION, "process")
        self.assertEqual(Node.CATEGORY, "illumorae")
        self.assertFalse(Node.OUTPUT_NODE)

    def test_no_init_required(self):
        # The class no longer defines __init__; instantiation takes no args.
        node = Node()
        self.assertIsInstance(node, Node)


class TestParseLoraSyntax(unittest.TestCase):
    """parse_lora_syntax: strength formats, two-strength, negatives, duplicates, names."""

    def setUp(self):
        self.node = Node()

    def test_integer_strength(self):
        self.assertEqual(self.node.parse_lora_syntax("<lora:foo:1>"), [("foo", 1.0, None)])

    def test_decimal_strength(self):
        self.assertEqual(self.node.parse_lora_syntax("<lora:foo:0.5>"), [("foo", 0.5, None)])

    def test_leading_dot_strength(self):
        self.assertEqual(self.node.parse_lora_syntax("<lora:foo:.5>"), [("foo", 0.5, None)])

    def test_trailing_dot_strength(self):
        # Regression (m2): the old regex rejected "1." (trailing dot).
        self.assertEqual(self.node.parse_lora_syntax("<lora:foo:1.>"), [("foo", 1.0, None)])

    def test_negative_strength(self):
        # Regression (M4): the old regex rejected negative strengths.
        self.assertEqual(self.node.parse_lora_syntax("<lora:foo:-0.5>"), [("foo", -0.5, None)])

    def test_negative_two_strength(self):
        self.assertEqual(
            self.node.parse_lora_syntax("<lora:foo:-0.5:-1.0>"),
            [("foo", -0.5, -1.0)],
        )

    def test_two_strength_syntax(self):
        # Regression (M3): the old regex did not recognize <lora:name:unet:clip>.
        self.assertEqual(
            self.node.parse_lora_syntax("<lora:foo:0.7:0.3>"),
            [("foo", 0.7, 0.3)],
        )

    def test_two_strength_trailing_dot(self):
        self.assertEqual(
            self.node.parse_lora_syntax("<lora:foo:1.:0.5>"),
            [("foo", 1.0, 0.5)],
        )

    def test_duplicate_names_preserved(self):
        # Regression (M2): the old dict-based parser collapsed duplicates.
        out = self.node.parse_lora_syntax("<lora:foo:0.3> <lora:foo:0.8>")
        self.assertEqual(out, [("foo", 0.3, None), ("foo", 0.8, None)])

    def test_name_preserved_verbatim(self):
        # Regression (m3): the old parser stripped whitespace from names,
        # silently renaming "<lora: foo :0.5>" to "foo". Names are now kept
        # verbatim so the emitted tag round-trips.
        out = self.node.parse_lora_syntax("<lora: foo :0.5>")
        self.assertEqual(out, [(" foo ", 0.5, None)])

    def test_name_with_dot(self):
        self.assertEqual(
            self.node.parse_lora_syntax("<lora:add_detail_xl.safetensors:0.7>"),
            [("add_detail_xl.safetensors", 0.7, None)],
        )

    def test_multiple_loras(self):
        out = self.node.parse_lora_syntax("<lora:a:0.3> <lora:b:0.8>")
        self.assertEqual(out, [("a", 0.3, None), ("b", 0.8, None)])

    def test_empty_text(self):
        self.assertEqual(self.node.parse_lora_syntax(""), [])

    def test_text_without_lora_tags(self):
        self.assertEqual(self.node.parse_lora_syntax("a cat on a chair"), [])

    def test_malformed_tag_skipped(self):
        # A tag whose strength does not match is skipped; valid tags survive.
        out = self.node.parse_lora_syntax("<lora:bad:abc> <lora:good:0.5>")
        self.assertEqual(out, [("good", 0.5, None)])

    def test_order_preserved(self):
        text = "<lora:z:0.1> <lora:a:0.2> <lora:m:0.3>"
        names = [r[0] for r in self.node.parse_lora_syntax(text)]
        self.assertEqual(names, ["z", "a", "m"])


class TestFormatLoraTag(unittest.TestCase):
    """format_lora_tag: single vs two-strength, precision."""

    def setUp(self):
        self.node = Node()

    def test_single_strength(self):
        self.assertEqual(self.node.format_lora_tag("foo", 0.5, None), "<lora:foo:0.5000>")

    def test_two_strength(self):
        self.assertEqual(self.node.format_lora_tag("foo", 0.7, 0.3), "<lora:foo:0.7000:0.3000>")

    def test_precision_four_decimals(self):
        # Regression (m1): the old emitter used :.2f, truncating sub-hundredth
        # precision despite the widget step being 0.0001.
        self.assertEqual(self.node.format_lora_tag("foo", 0.185175, None), "<lora:foo:0.1852>")

    def test_negative_strength(self):
        self.assertEqual(self.node.format_lora_tag("foo", -0.5, None), "<lora:foo:-0.5000>")

    def test_negative_two_strength(self):
        self.assertEqual(self.node.format_lora_tag("foo", -0.5, -1.0), "<lora:foo:-0.5000:-1.0000>")


class TestProcessMultiplier(unittest.TestCase):
    """process: multiplier applied to unet and clip."""

    def setUp(self):
        self.node = Node()

    def test_multiplier_applied(self):
        (out,) = self.node.process("<lora:foo:0.5>", multiplier=2.0,
                                   individual_cap_enabled=False, individual_cap=1.0,
                                   total_cap_enabled=False, total_cap=1.0)
        self.assertEqual(out, "<lora:foo:1.0000>")

    def test_multiplier_one_is_identity(self):
        text = "<lora:foo:0.5> <lora:bar:0.8:0.3>"
        (out,) = self.node.process(text, multiplier=1.0,
                                   individual_cap_enabled=False, individual_cap=1.0,
                                   total_cap_enabled=False, total_cap=1.0)
        self.assertEqual(out, "<lora:foo:0.5000> <lora:bar:0.8000:0.3000>")

    def test_multiplier_applied_to_both_strengths(self):
        # Regression (M3): both unet and clip are multiplied.
        (out,) = self.node.process("<lora:foo:0.5:0.2>", multiplier=2.0,
                                   individual_cap_enabled=False, individual_cap=1.0,
                                   total_cap_enabled=False, total_cap=1.0)
        self.assertEqual(out, "<lora:foo:1.0000:0.4000>")

    def test_multiplier_applied_to_negative(self):
        # Regression (M4): negative strengths are multiplied like any other.
        (out,) = self.node.process("<lora:foo:-0.5>", multiplier=2.0,
                                   individual_cap_enabled=False, individual_cap=1.0,
                                   total_cap_enabled=False, total_cap=1.0)
        self.assertEqual(out, "<lora:foo:-1.0000>")


class TestProcessPreservesSurroundingText(unittest.TestCase):
    """process: non-LoRA text is preserved (regression for M1)."""

    def setUp(self):
        self.node = Node()

    def test_text_around_tag_preserved(self):
        text = "a cute cat, <lora:style:1.0>, masterpiece"
        (out,) = self.node.process(text, multiplier=2.0,
                                   individual_cap_enabled=False, individual_cap=1.0,
                                   total_cap_enabled=False, total_cap=1.0)
        self.assertEqual(out, "a cute cat, <lora:style:2.0000>, masterpiece")

    def test_text_between_tags_preserved(self):
        text = "intro <lora:a:0.3> middle <lora:b:0.8> end"
        (out,) = self.node.process(text, multiplier=1.0,
                                   individual_cap_enabled=False, individual_cap=1.0,
                                   total_cap_enabled=False, total_cap=1.0)
        self.assertEqual(out, "intro <lora:a:0.3000> middle <lora:b:0.8000> end")

    def test_prefix_and_suffix_byte_for_byte(self):
        text = "PREFIX <lora:foo:0.5> SUFFIX"
        (out,) = self.node.process(text, multiplier=3.0,
                                   individual_cap_enabled=False, individual_cap=1.0,
                                   total_cap_enabled=False, total_cap=1.0)
        self.assertTrue(out.startswith("PREFIX "))
        self.assertTrue(out.endswith(" SUFFIX"))

    def test_newlines_preserved(self):
        text = "<lora:a:0.3>\n<lora:b:0.8>"
        (out,) = self.node.process(text, multiplier=1.0,
                                   individual_cap_enabled=False, individual_cap=1.0,
                                   total_cap_enabled=False, total_cap=1.0)
        self.assertEqual(out, "<lora:a:0.3000>\n<lora:b:0.8000>")

    def test_no_tags_returns_input_unchanged(self):
        text = "just a prompt with no loras"
        (out,) = self.node.process(text, multiplier=2.0,
                                   individual_cap_enabled=True, individual_cap=0.5,
                                   total_cap_enabled=True, total_cap=0.5)
        self.assertEqual(out, text)

    def test_empty_text_returns_empty(self):
        (out,) = self.node.process("", multiplier=2.0,
                                   individual_cap_enabled=False, individual_cap=1.0,
                                   total_cap_enabled=False, total_cap=1.0)
        self.assertEqual(out, "")


class TestProcessDuplicateNames(unittest.TestCase):
    """process: duplicate LoRA names are preserved as separate tags (M2)."""

    def setUp(self):
        self.node = Node()

    def test_duplicate_names_both_emitted(self):
        text = "<lora:foo:0.3> <lora:foo:0.8>"
        (out,) = self.node.process(text, multiplier=1.0,
                                   individual_cap_enabled=False, individual_cap=1.0,
                                   total_cap_enabled=False, total_cap=1.0)
        self.assertEqual(out, "<lora:foo:0.3000> <lora:foo:0.8000>")

    def test_duplicate_names_with_text_between(self):
        text = "<lora:foo:0.3> middle <lora:foo:0.8>"
        (out,) = self.node.process(text, multiplier=2.0,
                                   individual_cap_enabled=False, individual_cap=1.0,
                                   total_cap_enabled=False, total_cap=1.0)
        self.assertEqual(out, "<lora:foo:0.6000> middle <lora:foo:1.6000>")

    def test_duplicate_names_count_preserved(self):
        text = "<lora:foo:0.1> <lora:foo:0.2> <lora:foo:0.3>"
        (out,) = self.node.process(text, multiplier=1.0,
                                   individual_cap_enabled=False, individual_cap=1.0,
                                   total_cap_enabled=False, total_cap=1.0)
        self.assertEqual(out.count("<lora:foo:"), 3)


class TestProcessIndividualCap(unittest.TestCase):
    """process: individual cap clamps the upper bound."""

    def setUp(self):
        self.node = Node()

    def test_cap_clamps_high_values(self):
        (out,) = self.node.process("<lora:foo:0.8>", multiplier=2.0,
                                   individual_cap_enabled=True, individual_cap=1.0,
                                   total_cap_enabled=False, total_cap=1.0)
        self.assertEqual(out, "<lora:foo:1.0000>")

    def test_cap_does_not_raise_low_values(self):
        (out,) = self.node.process("<lora:foo:0.3>", multiplier=2.0,
                                   individual_cap_enabled=True, individual_cap=1.0,
                                   total_cap_enabled=False, total_cap=1.0)
        self.assertEqual(out, "<lora:foo:0.6000>")

    def test_cap_applies_to_both_strengths(self):
        (out,) = self.node.process("<lora:foo:0.8:0.9>", multiplier=2.0,
                                   individual_cap_enabled=True, individual_cap=1.0,
                                   total_cap_enabled=False, total_cap=1.0)
        self.assertEqual(out, "<lora:foo:1.0000:1.0000>")

    def test_cap_does_not_affect_negatives(self):
        # The cap is an upper bound (min(value, cap)); a negative value is
        # already below the cap and passes through unchanged.
        (out,) = self.node.process("<lora:foo:-0.5>", multiplier=2.0,
                                   individual_cap_enabled=True, individual_cap=1.0,
                                   total_cap_enabled=False, total_cap=1.0)
        self.assertEqual(out, "<lora:foo:-1.0000>")

    def test_cap_zero_zeros_out(self):
        # Regression (m4): individual_cap=0 with enforcement clamps to 0.
        (out,) = self.node.process("<lora:foo:0.8>", multiplier=2.0,
                                   individual_cap_enabled=True, individual_cap=0.0,
                                   total_cap_enabled=False, total_cap=1.0)
        self.assertEqual(out, "<lora:foo:0.0000>")

    def test_cap_disabled_leaves_values(self):
        (out,) = self.node.process("<lora:foo:0.8>", multiplier=2.0,
                                   individual_cap_enabled=False, individual_cap=1.0,
                                   total_cap_enabled=False, total_cap=1.0)
        self.assertEqual(out, "<lora:foo:1.6000>")


class TestProcessTotalCap(unittest.TestCase):
    """process: total cap scales all strengths proportionally."""

    def setUp(self):
        self.node = Node()

    def test_scales_down_when_over_cap(self):
        # total = 0.6 + 0.8 = 1.4; cap = 1.0 -> scale = 1.0/1.4
        (out,) = self.node.process("<lora:a:0.6> <lora:b:0.8>", multiplier=1.0,
                                   individual_cap_enabled=False, individual_cap=1.0,
                                   total_cap_enabled=True, total_cap=1.0)
        vals = [v for _, (v, _) in _strengths(out)]
        # Output is formatted to 4 decimals, so compare at 4 places.
        self.assertAlmostEqual(sum(vals), 1.0, places=4)
        # Proportional: a=0.6/1.4, b=0.8/1.4 (rounded to 4 decimals on output).
        self.assertAlmostEqual(vals[0], 0.6 / 1.4, places=4)
        self.assertAlmostEqual(vals[1], 0.8 / 1.4, places=4)

    def test_no_scaling_when_under_cap(self):
        (out,) = self.node.process("<lora:a:0.3> <lora:b:0.4>", multiplier=1.0,
                                   individual_cap_enabled=False, individual_cap=1.0,
                                   total_cap_enabled=True, total_cap=1.0)
        self.assertEqual(out, "<lora:a:0.3000> <lora:b:0.4000>")

    def test_scales_both_unet_and_clip(self):
        # total = (0.6+0.2) + (0.8+0.4) = 2.0; cap = 1.0 -> scale = 0.5
        (out,) = self.node.process("<lora:a:0.6:0.2> <lora:b:0.8:0.4>", multiplier=1.0,
                                   individual_cap_enabled=False, individual_cap=1.0,
                                   total_cap_enabled=True, total_cap=1.0)
        recs = _strengths(out)
        total = sum(u + (c or 0.0) for _, (u, c) in recs)
        self.assertAlmostEqual(total, 1.0, places=6)
        self.assertAlmostEqual(recs[0][1][0], 0.3, places=6)
        self.assertAlmostEqual(recs[0][1][1], 0.1, places=6)

    def test_total_cap_zero_zeros_out(self):
        # Regression (m4): total_cap=0 scales by 0/total = 0.
        (out,) = self.node.process("<lora:a:0.6> <lora:b:0.8>", multiplier=1.0,
                                   individual_cap_enabled=False, individual_cap=1.0,
                                   total_cap_enabled=True, total_cap=0.0)
        vals = [v for _, (v, _) in _strengths(out)]
        self.assertEqual(vals, [0.0, 0.0])

    def test_total_cap_with_multiplier(self):
        # multiplier=2 -> 1.2 + 1.6 = 2.8; cap=1.0 -> scale=1/2.8
        (out,) = self.node.process("<lora:a:0.6> <lora:b:0.8>", multiplier=2.0,
                                   individual_cap_enabled=False, individual_cap=1.0,
                                   total_cap_enabled=True, total_cap=1.0)
        vals = [v for _, (v, _) in _strengths(out)]
        self.assertAlmostEqual(sum(vals), 1.0, places=6)

    def test_total_cap_disabled_no_scaling(self):
        (out,) = self.node.process("<lora:a:0.6> <lora:b:0.8>", multiplier=1.0,
                                   individual_cap_enabled=False, individual_cap=1.0,
                                   total_cap_enabled=False, total_cap=1.0)
        self.assertEqual(out, "<lora:a:0.6000> <lora:b:0.8000>")


class TestProcessCombined(unittest.TestCase):
    """process: individual cap then total cap compose correctly."""

    def setUp(self):
        self.node = Node()

    def test_individual_then_total(self):
        # multiplier=2 -> a=1.2, b=1.6; individual cap=1.0 -> a=1.0, b=1.0;
        # total = 2.0; total cap=1.0 -> scale=0.5 -> a=0.5, b=0.5.
        (out,) = self.node.process("<lora:a:0.6> <lora:b:0.8>", multiplier=2.0,
                                   individual_cap_enabled=True, individual_cap=1.0,
                                   total_cap_enabled=True, total_cap=1.0)
        vals = [v for _, (v, _) in _strengths(out)]
        self.assertAlmostEqual(sum(vals), 1.0, places=6)
        self.assertAlmostEqual(vals[0], 0.5, places=6)
        self.assertAlmostEqual(vals[1], 0.5, places=6)

    def test_all_tags_rewritten(self):
        text = "<lora:a:0.3> <lora:b:0.8> <lora:c:0.5>"
        (out,) = self.node.process(text, multiplier=1.0,
                                   individual_cap_enabled=False, individual_cap=1.0,
                                   total_cap_enabled=False, total_cap=1.0)
        names = [n for n, _ in _strengths(out)]
        self.assertEqual(names, ["a", "b", "c"])


class TestProcessOutputShape(unittest.TestCase):
    """process: return type and debug path sanity."""

    def setUp(self):
        self.node = Node()

    def test_returns_one_tuple_of_str(self):
        result = self.node.process("<lora:foo:0.5>", multiplier=1.0,
                                   individual_cap_enabled=False, individual_cap=1.0,
                                   total_cap_enabled=False, total_cap=1.0)
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 1)
        self.assertIsInstance(result[0], str)

    def test_debug_prints_does_not_crash(self):
        import contextlib
        import io

        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            (out,) = self.node.process("a cat, <lora:foo:0.5>, dog", multiplier=2.0,
                                       individual_cap_enabled=True, individual_cap=1.0,
                                       total_cap_enabled=True, total_cap=1.0,
                                       debug_prints=True)
        self.assertEqual(out, "a cat, <lora:foo:1.0000>, dog")
        # Debug output should mention key stages.
        text = buf.getvalue()
        self.assertIn("Input Text:", text)
        self.assertIn("Final output text:", text)


if __name__ == "__main__":
    unittest.main()
