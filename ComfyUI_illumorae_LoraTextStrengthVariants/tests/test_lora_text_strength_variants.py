"""Regression tests for illumoraeLoRARandomizeStrengthOnTextNode.

Covers the bugs fixed in the 20260816 review pass:

- parse_lora_syntax: integer strengths, ``.NN`` strengths, ``N.`` trailing-dot
  strengths, names containing dots, malformed tags skipped (and reported in
  debug), empty input.
- randomize_strengths: total stays under TOTAL_STRENGTH, individual values
  capped at MAX_INDIVIDUAL_STRENGTH, every input name present in output,
  zeroed LoRAs emitted at 0.00, reproducible with the same rng, uses a local
  rng (does not perturb the global random module).
- highlight_random_lora: exactly one LoRA at HIGHLIGHT_STRENGTH, the rest at
  DIM_STRENGTH, empty input returns empty dict, reproducible, local rng.
- process: pass-through returns input unchanged; surrounding prompt text is
  preserved (only <lora:...> tags rewritten); randomize+highlight mutually
  exclusive (randomize wins); no <lora:...> tags returns input unchanged;
  seeded runs are reproducible; output is a 1-tuple of str.
- Registration: NODE_CLASS_MAPPINGS / NODE_DISPLAY_NAME_MAPPINGS / class
  attributes match expectations.

Usage:
    python -m tests.test_lora_text_strength_variants
    python -m pytest tests/test_lora_text_strength_variants.py -v
"""
from __future__ import annotations

import os
import random
import re
import sys
import unittest

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PARENT = os.path.dirname(_THIS_DIR)
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)

from lora_text_strength_variants import (  # noqa: E402
    NODE_CLASS_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS,
    illumoraeLoRARandomizeStrengthOnTextNode,
)

Node = illumoraeLoRARandomizeStrengthOnTextNode


class TestRegistration(unittest.TestCase):
    """Node is registered with the expected id and display name."""

    def test_node_class_mapping(self):
        self.assertIn("illumoraeLoRARandomizeStrengthOnTextNode", NODE_CLASS_MAPPINGS)
        self.assertIs(
            NODE_CLASS_MAPPINGS["illumoraeLoRARandomizeStrengthOnTextNode"],
            illumoraeLoRARandomizeStrengthOnTextNode,
        )

    def test_node_display_name_mapping(self):
        self.assertIn("illumoraeLoRARandomizeStrengthOnTextNode", NODE_DISPLAY_NAME_MAPPINGS)
        self.assertEqual(
            NODE_DISPLAY_NAME_MAPPINGS["illumoraeLoRARandomizeStrengthOnTextNode"],
            "LoRA Randomize Strength on Text",
        )

    def test_class_attributes(self):
        self.assertEqual(Node.RETURN_TYPES, ("STRING",))
        self.assertEqual(Node.RETURN_NAMES, ("modified_text",))
        self.assertEqual(Node.FUNCTION, "process")
        self.assertEqual(Node.CATEGORY, "illumorae")
        # OUTPUT_NODE was removed (defaults to False); ensure it is not set True.
        self.assertFalse(getattr(Node, "OUTPUT_NODE", False))

    def test_strength_constants_present(self):
        # Constants are class-level (no __init__ instance state).
        self.assertEqual(Node.TOTAL_STRENGTH, 1.5)
        self.assertEqual(Node.MAX_INDIVIDUAL_STRENGTH, 0.9)
        self.assertEqual(Node.HIGHLIGHT_STRENGTH, 0.9)
        self.assertEqual(Node.DIM_STRENGTH, 0.01)

    def test_no_init_required(self):
        # The class no longer defines __init__; instantiation must not require
        # any args and must not rely on instance-stored config.
        node = Node()
        self.assertEqual(node.TOTAL_STRENGTH, Node.TOTAL_STRENGTH)
        self.assertEqual(node.MAX_INDIVIDUAL_STRENGTH, Node.MAX_INDIVIDUAL_STRENGTH)


class TestParseLoraSyntax(unittest.TestCase):
    """parse_lora_syntax: strength formats, names, malformed tags, empty input."""

    def setUp(self):
        self.node = Node()

    def test_integer_strength(self):
        # Regression: the old regex required a fractional digit and rejected "1".
        self.assertEqual(self.node.parse_lora_syntax("<lora:foo:1>"), {"foo": 1.0})

    def test_decimal_strength(self):
        self.assertEqual(self.node.parse_lora_syntax("<lora:foo:0.5>"), {"foo": 0.5})

    def test_leading_dot_strength(self):
        self.assertEqual(self.node.parse_lora_syntax("<lora:foo:.5>"), {"foo": 0.5})

    def test_trailing_dot_strength(self):
        # Regression: the old regex rejected "1." (trailing dot, no fractional digit).
        self.assertEqual(self.node.parse_lora_syntax("<lora:foo:1.>"), {"foo": 1.0})

    def test_name_with_dot(self):
        self.assertEqual(
            self.node.parse_lora_syntax("<lora:add_detail_xl.safetensors:0.7>"),
            {"add_detail_xl.safetensors": 0.7},
        )

    def test_multiple_loras(self):
        parsed = self.node.parse_lora_syntax("<lora:a:0.3> <lora:b:0.8>")
        self.assertEqual(parsed, {"a": 0.3, "b": 0.8})

    def test_name_whitespace_stripped(self):
        self.assertEqual(self.node.parse_lora_syntax("<lora: foo :0.5>"), {"foo": 0.5})

    def test_empty_text(self):
        self.assertEqual(self.node.parse_lora_syntax(""), {})

    def test_text_without_lora_tags(self):
        self.assertEqual(self.node.parse_lora_syntax("a cat on a chair"), {})

    def test_malformed_tag_skipped(self):
        # A tag whose strength does not match the pattern is skipped (not in
        # the parsed dict). Surrounding valid tags are still parsed.
        parsed = self.node.parse_lora_syntax("<lora:bad:abc> <lora:good:0.5>")
        self.assertEqual(parsed, {"good": 0.5})

    def test_malformed_tag_reported_in_debug(self):
        # When debug_prints is on, a malformed <lora:...> tag is logged. We
        # capture stdout to verify the warning is emitted.
        import io
        import contextlib

        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            self.node.parse_lora_syntax("<lora:bad:abc>", debug_prints=True)
        out = buf.getvalue()
        self.assertIn("Skipped malformed", out)


class TestRandomizeStrengths(unittest.TestCase):
    """randomize_strengths: budget, caps, completeness, reproducibility, local rng."""

    def setUp(self):
        self.node = Node()

    def test_total_under_budget(self):
        names = ["a", "b", "c", "d", "e"]
        rng = random.Random(42)
        out = self.node.randomize_strengths(names, rng)
        self.assertLessEqual(sum(out.values()), Node.TOTAL_STRENGTH + 1e-9)

    def test_individual_capped(self):
        names = ["a", "b", "c"]
        # Try many seeds to exercise the cap across draws.
        for seed in range(20):
            out = self.node.randomize_strengths(names, random.Random(seed))
            for v in out.values():
                self.assertLessEqual(v, Node.MAX_INDIVIDUAL_STRENGTH + 1e-9)

    def test_all_names_present(self):
        names = ["a", "b", "c", "d"]
        out = self.node.randomize_strengths(names, random.Random(1))
        self.assertEqual(set(out.keys()), set(names))

    def test_zeroed_loras_emitted_at_zero(self):
        # With many LoRAs the budget exhausts; remaining ones get 0.0.
        names = [f"lora{i}" for i in range(20)]
        out = self.node.randomize_strengths(names, random.Random(7))
        # At least one should be 0.0 (budget 1.5 across 20 LoRAs).
        self.assertIn(0.0, out.values())
        # And all zeroed entries are exactly 0.0, not missing.
        for name in names:
            self.assertIn(name, out)

    def test_reproducible_with_same_seed(self):
        names = ["a", "b", "c", "d"]
        out1 = self.node.randomize_strengths(names, random.Random(123))
        out2 = self.node.randomize_strengths(names, random.Random(123))
        self.assertEqual(out1, out2)

    def test_different_seeds_differ(self):
        # Not guaranteed for every pair, but two distinct seeds should usually
        # produce different allocations for a non-trivial name set.
        names = ["a", "b", "c", "d", "e"]
        out1 = self.node.randomize_strengths(names, random.Random(1))
        out2 = self.node.randomize_strengths(names, random.Random(2))
        self.assertNotEqual(out1, out2)

    def test_does_not_perturb_global_rng(self):
        # Regression for the global random.seed() bug: calling the node must
        # not affect the process-global random module's state.
        random.seed(999)
        expected = random.random()
        # Reset and call the node several times.
        random.seed(999)
        self.node.randomize_strengths(["a", "b", "c"], random.Random(50))
        self.node.randomize_strengths(["x", "y"], random.Random(51))
        after = random.random()
        self.assertEqual(after, expected)

    def test_empty_names_returns_empty(self):
        out = self.node.randomize_strengths([], random.Random(0))
        self.assertEqual(out, {})

    def test_single_lora(self):
        out = self.node.randomize_strengths(["solo"], random.Random(3))
        self.assertEqual(set(out.keys()), {"solo"})
        self.assertLessEqual(out["solo"], Node.MAX_INDIVIDUAL_STRENGTH)


class TestHighlightRandomLora(unittest.TestCase):
    """highlight_random_lora: one high, rest dim, empty, reproducible, local rng."""

    def setUp(self):
        self.node = Node()

    def test_one_highlighted_rest_dimmed(self):
        names = ["a", "b", "c", "d"]
        out = self.node.highlight_random_lora(names, random.Random(10))
        highs = [n for n, v in out.items() if v == Node.HIGHLIGHT_STRENGTH]
        dims = [n for n, v in out.items() if v == Node.DIM_STRENGTH]
        self.assertEqual(len(highs), 1)
        self.assertEqual(len(dims), len(names) - 1)
        self.assertEqual(set(out.keys()), set(names))

    def test_uses_class_constants(self):
        names = ["a", "b"]
        out = self.node.highlight_random_lora(names, random.Random(0))
        self.assertIn(Node.HIGHLIGHT_STRENGTH, out.values())
        self.assertIn(Node.DIM_STRENGTH, out.values())

    def test_empty_names_returns_empty(self):
        out = self.node.highlight_random_lora([], random.Random(0))
        self.assertEqual(out, {})

    def test_reproducible_with_same_seed(self):
        names = ["a", "b", "c", "d"]
        out1 = self.node.highlight_random_lora(names, random.Random(321))
        out2 = self.node.highlight_random_lora(names, random.Random(321))
        self.assertEqual(out1, out2)

    def test_does_not_perturb_global_rng(self):
        random.seed(1234)
        expected = random.random()
        random.seed(1234)
        self.node.highlight_random_lora(["a", "b", "c"], random.Random(50))
        after = random.random()
        self.assertEqual(after, expected)


class TestProcessPassthrough(unittest.TestCase):
    """process: pass-through and no-tag paths return input unchanged."""

    def setUp(self):
        self.node = Node()

    def test_both_flags_off_returns_input_unchanged(self):
        text = "a cat, <lora:foo:0.5>, on a chair"
        (out,) = self.node.process(text, randomize=False, highlight=False, seed=1)
        self.assertEqual(out, text)

    def test_empty_text_passthrough(self):
        (out,) = self.node.process("", randomize=False, highlight=False, seed=1)
        self.assertEqual(out, "")

    def test_no_lora_tags_returns_input_unchanged(self):
        # Even with a mode on, no <lora:...> tags means no work to do.
        text = "a cat on a chair"
        (out,) = self.node.process(text, randomize=True, seed=1)
        self.assertEqual(out, text)

    def test_no_lora_tags_highlight_returns_input_unchanged(self):
        text = "a cat on a chair"
        (out,) = self.node.process(text, highlight=True, seed=1)
        self.assertEqual(out, text)


class TestProcessPreservesSurroundingText(unittest.TestCase):
    """process: surrounding prompt text is preserved (regression for 2.5)."""

    def setUp(self):
        self.node = Node()

    def test_randomize_preserves_surrounding_text(self):
        text = "a cat, <lora:foo:0.5>, sitting on a chair"
        (out,) = self.node.process(text, randomize=True, seed=42)
        self.assertIn("a cat,", out)
        self.assertIn("sitting on a chair", out)
        # The tag is still present, rewritten with a numeric strength.
        self.assertIn("<lora:foo:", out)

    def test_highlight_preserves_surrounding_text(self):
        text = "a cat, <lora:foo:0.5>, sitting on a chair"
        (out,) = self.node.process(text, highlight=True, seed=42)
        self.assertIn("a cat,", out)
        self.assertIn("sitting on a chair", out)
        self.assertIn("<lora:foo:", out)

    def test_multiple_tags_preserve_text_between_them(self):
        text = "intro <lora:a:0.3> middle <lora:b:0.8> end"
        (out,) = self.node.process(text, randomize=True, seed=5)
        self.assertIn("intro", out)
        self.assertIn("middle", out)
        self.assertIn("end", out)
        self.assertIn("<lora:a:", out)
        self.assertIn("<lora:b:", out)

    def test_text_before_and_after_tag_unchanged(self):
        text = "PREFIX <lora:foo:0.5> SUFFIX"
        (out,) = self.node.process(text, highlight=True, seed=99)
        # The non-tag portions are byte-for-byte preserved.
        self.assertTrue(out.startswith("PREFIX "))
        self.assertTrue(out.endswith(" SUFFIX"))


class TestProcessMutualExclusivity(unittest.TestCase):
    """process: randomize + highlight -> randomize wins (regression for 2.1)."""

    def setUp(self):
        self.node = Node()

    def test_both_flags_randomize_takes_precedence(self):
        # With randomize winning, the output is NOT the highlight pattern
        # (one at 0.9, rest at 0.01). We check that the output differs from
        # the pure-highlight output, and matches the pure-randomize output.
        text = "<lora:a:0.3> <lora:b:0.8> <lora:c:0.5>"
        (both,) = self.node.process(text, randomize=True, highlight=True, seed=77)
        (rand_only,) = self.node.process(text, randomize=True, highlight=False, seed=77)
        (high_only,) = self.node.process(text, randomize=False, highlight=True, seed=77)
        self.assertEqual(both, rand_only)
        # And it should not equal the highlight-only output (which would indicate
        # highlight clobbered the randomized values).
        self.assertNotEqual(both, high_only)

    def test_both_flags_does_not_produce_highlight_pattern(self):
        # The highlight pattern has exactly one LoRA at 0.90 and the rest at 0.01.
        # The randomize-winning output should not match that pattern.
        text = "<lora:a:0.3> <lora:b:0.8> <lora:c:0.5> <lora:d:0.2>"
        (out,) = self.node.process(text, randomize=True, highlight=True, seed=12)
        strengths = re.findall(r"<lora:[^:<>]+:([0-9.]+)>", out)
        values = [float(s) for s in strengths]
        # Highlight pattern would be {0.9, 0.01, 0.01, 0.01}.
        self.assertNotEqual(sorted(values), [0.01, 0.01, 0.01, 0.9])


class TestProcessReproducibility(unittest.TestCase):
    """process: seeded runs are reproducible (regression for 2.2/2.3)."""

    def setUp(self):
        self.node = Node()

    def test_randomize_reproducible_same_seed(self):
        text = "<lora:a:0.3> <lora:b:0.8> <lora:c:0.5>"
        (out1,) = self.node.process(text, randomize=True, seed=555)
        (out2,) = self.node.process(text, randomize=True, seed=555)
        self.assertEqual(out1, out2)

    def test_highlight_reproducible_same_seed(self):
        text = "<lora:a:0.3> <lora:b:0.8> <lora:c:0.5>"
        (out1,) = self.node.process(text, highlight=True, seed=555)
        (out2,) = self.node.process(text, highlight=True, seed=555)
        self.assertEqual(out1, out2)

    def test_seed_zero_is_reproducible(self):
        # Regression for 2.3: seed=0 used to mean "unseeded" (non-reproducible).
        # Now seed=0 is a valid deterministic seed.
        text = "<lora:a:0.3> <lora:b:0.8>"
        (out1,) = self.node.process(text, randomize=True, seed=0)
        (out2,) = self.node.process(text, randomize=True, seed=0)
        self.assertEqual(out1, out2)

    def test_process_does_not_perturb_global_rng(self):
        # Calling process must not seed the global random module.
        random.seed(4242)
        expected = random.random()
        random.seed(4242)
        self.node.process("<lora:a:0.3> <lora:b:0.8>", randomize=True, seed=10)
        self.node.process("<lora:a:0.3> <lora:b:0.8>", highlight=True, seed=10)
        after = random.random()
        self.assertEqual(after, expected)


class TestProcessOutputShape(unittest.TestCase):
    """process: return type and value sanity."""

    def setUp(self):
        self.node = Node()

    def test_returns_one_tuple_of_str(self):
        result = self.node.process("<lora:foo:0.5>", randomize=True, seed=1)
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 1)
        self.assertIsInstance(result[0], str)

    def test_randomize_output_strengths_in_range(self):
        text = "<lora:a:0.3> <lora:b:0.8> <lora:c:0.5>"
        (out,) = self.node.process(text, randomize=True, seed=3)
        strengths = re.findall(r"<lora:[^:<>]+:([0-9.]+)>", out)
        values = [float(s) for s in strengths]
        for v in values:
            self.assertGreaterEqual(v, 0.0)
            self.assertLessEqual(v, Node.MAX_INDIVIDUAL_STRENGTH + 1e-9)
        self.assertLessEqual(sum(values), Node.TOTAL_STRENGTH + 1e-9)

    def test_highlight_output_has_one_high(self):
        text = "<lora:a:0.3> <lora:b:0.8> <lora:c:0.5>"
        (out,) = self.node.process(text, highlight=True, seed=3)
        strengths = re.findall(r"<lora:[^:<>]+:([0-9.]+)>", out)
        values = [float(s) for s in strengths]
        self.assertEqual(values.count(Node.HIGHLIGHT_STRENGTH), 1)
        self.assertEqual(values.count(Node.DIM_STRENGTH), len(values) - 1)

    def test_all_tags_rewritten(self):
        # Every <lora:...> tag in the input should still be present (as a tag)
        # in the output, just with a new strength.
        text = "<lora:a:0.3> <lora:b:0.8> <lora:c:0.5>"
        (out,) = self.node.process(text, randomize=True, seed=8)
        out_tags = re.findall(r"<lora:([^:<>]+):", out)
        self.assertEqual(sorted(out_tags), ["a", "b", "c"])


if __name__ == "__main__":
    unittest.main()
