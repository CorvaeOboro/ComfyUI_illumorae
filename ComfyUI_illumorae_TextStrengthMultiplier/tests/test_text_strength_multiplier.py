"""Regression tests for illumoraeTextStrengthMultiplierNode.

Covers the bugs fixed in the 20260816 review pass:

- M1: LoRA placeholders must not be re-enclosed by wrap_plain_text. A LoRA
  tag on its own line, or a paragraph containing a LoRA tag, is left
  unwrapped so the restored <lora:...> tag is not nested inside weighted
  syntax.
- M2: Nested parentheses such as "(a (b:1.5):2.0)" are handled by a
  depth-aware scanner; the outer weight is multiplied and the inner group
  is preserved (the old non-greedy regex truncated at the first ")").
- M3: Total-cap rescaling respects the individual minimum as a hard floor
  and the total cap as a hard ceiling; sections pinned at the minimum are
  fixed there and the remaining sections are rescaled against the
  remaining budget, so the final sum does not exceed the total cap.
- m1: Emitted weights use "(text:1.5)" with no space before the colon.
- m2: CRLF/CR paragraph breaks are normalized to LF before splitting.
- m4: float() parse failures are caught as ValueError, not Exception.
- m5: LoRA placeholders use a UUID-based prefix so literal "@@LORA_n@@"
  text in the prompt cannot collide with a placeholder.
- m6: cap-then-min ordering means cap < minimum resolves to the minimum.

Also covers the module-level helpers _find_weighted_groups and
_rewrite_weighted_groups, the plain-text wrapping path, individual cap/min
without total cap, and the no-op cases (empty input, multiplier 1.0).

Usage:
    python -m tests.test_text_strength_multiplier
    python -m pytest tests/test_text_strength_multiplier.py -v
"""
from __future__ import annotations

import os
import sys
import unittest

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PARENT = os.path.dirname(_THIS_DIR)
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)

from text_strength_multiplier import (  # noqa: E402
    NODE_CLASS_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS,
    _find_weighted_groups,
    _rewrite_weighted_groups,
    illumoraeTextStrengthMultiplierNode,
)

Node = illumoraeTextStrengthMultiplierNode


def run(
    text: str,
    multiplier: float = 1.0,
    individual_cap_enabled: bool = False,
    individual_cap: float = 1.0,
    individual_min_enabled: bool = False,
    individual_min: float = 0.0,
    total_cap_enabled: bool = False,
    total_cap: float = 1.0,
) -> str:
    """Call process() with the given args and return the single output string."""
    out = Node().process(
        text,
        multiplier,
        individual_cap_enabled,
        individual_cap,
        individual_min_enabled,
        individual_min,
        total_cap_enabled,
        total_cap,
    )
    assert isinstance(out, tuple) and len(out) == 1
    return out[0]


def strengths(text: str):
    """Extract the numeric strengths from every weighted group in text."""
    return [float(s) for _, _, _, s in _find_weighted_groups(text) if s]


class TestFindWeightedGroups(unittest.TestCase):
    """_find_weighted_groups: depth-aware top-level group discovery."""

    def test_simple_group_with_strength(self):
        groups = _find_weighted_groups("(fruit:1.5)")
        self.assertEqual(len(groups), 1)
        start, end, content, strength = groups[0]
        self.assertEqual(start, 0)
        # "(fruit:1.5)" is 11 chars; the closing ")" is at index 10.
        self.assertEqual(end, 10)
        self.assertEqual(content, "fruit")
        self.assertEqual(strength, "1.5")

    def test_simple_group_without_strength(self):
        groups = _find_weighted_groups("(fruit)")
        self.assertEqual(len(groups), 1)
        _, _, content, strength = groups[0]
        self.assertEqual(content, "fruit")
        self.assertEqual(strength, "")

    def test_nested_group_outer_only(self):
        # Only the outer group is returned; the inner group is part of the
        # outer content and is not re-parsed.
        groups = _find_weighted_groups("(a (b:1.5):2.0)")
        self.assertEqual(len(groups), 1)
        _, _, content, strength = groups[0]
        self.assertEqual(content, "a (b:1.5)")
        self.assertEqual(strength, "2.0")

    def test_multiple_top_level_groups(self):
        groups = _find_weighted_groups("(a:1.0)(b:2.0) text (c:3.0)")
        self.assertEqual(len(groups), 3)
        self.assertEqual([g[2] for g in groups], ["a", "b", "c"])
        self.assertEqual([g[3] for g in groups], ["1.0", "2.0", "3.0"])

    def test_unbalanced_open_paren_skipped(self):
        # An opening paren with no matching close is not emitted as a group.
        groups = _find_weighted_groups("before (unbalanced after")
        self.assertEqual(groups, [])

    def test_empty_string(self):
        self.assertEqual(_find_weighted_groups(""), [])

    def test_no_parens(self):
        self.assertEqual(_find_weighted_groups("plain text"), [])

    def test_strength_with_spaces(self):
        groups = _find_weighted_groups("(fruit: 1.5 )")
        self.assertEqual(len(groups), 1)
        _, _, content, strength = groups[0]
        self.assertEqual(content, "fruit")
        self.assertEqual(strength, "1.5")

    def test_inner_text_with_colon_not_trailing_is_content(self):
        # A colon that is not a trailing strength suffix stays in content.
        groups = _find_weighted_groups("(see figure 3: ref)")
        self.assertEqual(len(groups), 1)
        _, _, content, strength = groups[0]
        # "ref" is not a number, so no strength is parsed.
        self.assertEqual(strength, "")
        self.assertEqual(content, "see figure 3: ref")


class TestRewriteWeightedGroups(unittest.TestCase):
    """_rewrite_weighted_groups: rebuilds text via a transform callback."""

    def test_identity_transform_preserves_text(self):
        text = "(a:1.0) and (b:2.0)"
        out, parsed = _rewrite_weighted_groups(text, lambda c, s: f"{c}:{s}")
        self.assertEqual(out, text)
        self.assertEqual(parsed, [("a", "1.0"), ("b", "2.0")])

    def test_no_groups_returns_unchanged(self):
        text = "no groups here"
        out, parsed = _rewrite_weighted_groups(text, lambda c, s: "x")
        self.assertEqual(out, text)
        self.assertEqual(parsed, [])

    def test_transform_replaces_group_inner(self):
        text = "(a:1.0)"
        out, _ = _rewrite_weighted_groups(text, lambda c, s: f"{c}:9.0")
        self.assertEqual(out, "(a:9.0)")

    def test_preserves_text_between_groups(self):
        text = "x(a:1.0)y(b:2.0)z"
        out, _ = _rewrite_weighted_groups(text, lambda c, s: f"{c}:{s}")
        self.assertEqual(out, text)


class TestBasicMultiplication(unittest.TestCase):
    """Step 2: weighted-group strength multiplication."""

    def test_multiply_existing_strength(self):
        self.assertEqual(run("(fruit:1.5)", multiplier=2.0), "(fruit:3.00)")

    def test_multiply_default_strength(self):
        # A group with no explicit strength gets base 1.0 * multiplier.
        self.assertEqual(run("(fruit)", multiplier=2.0), "(fruit:2.00)")

    def test_multiplier_one_preserves_strength(self):
        self.assertEqual(run("(fruit:1.5)", multiplier=1.0), "(fruit:1.50)")

    def test_no_space_before_colon(self):
        # m1: emitted weights must not have a space before the colon.
        out = run("(fruit:1.0)", multiplier=2.0)
        self.assertNotIn(" :", out)
        self.assertEqual(out, "(fruit:2.00)")

    def test_multiple_groups(self):
        self.assertEqual(
            run("(a:1.0)(b:2.0)", multiplier=2.0),
            "(a:2.00)(b:4.00)",
        )

    def test_empty_input(self):
        self.assertEqual(run("", multiplier=2.0), "")

    def test_plain_text_no_groups_unchanged_when_mult_one(self):
        # multiplier 1.0 wraps plain text at 1.0 strength.
        self.assertEqual(run("a cat", multiplier=1.0), "(a cat:1.00)")


class TestNestedParens(unittest.TestCase):
    """M2: nested parentheses are handled by the depth-aware scanner."""

    def test_nested_outer_weight_multiplied_inner_preserved(self):
        out = run("(a (b:1.5):2.0)", multiplier=2.0)
        self.assertEqual(out, "(a (b:1.5):4.00)")

    def test_nested_no_outer_strength(self):
        out = run("(a (b:1.5))", multiplier=2.0)
        # Outer has no explicit strength -> base 1.0 * 2.0 = 2.0.
        self.assertEqual(out, "(a (b:1.5):2.00)")

    def test_deeply_nested(self):
        out = run("(a (b (c:1.0):2.0):3.0)", multiplier=2.0)
        self.assertEqual(out, "(a (b (c:1.0):2.0):6.00)")

    def test_nested_not_truncated_at_first_close(self):
        # The old non-greedy regex would produce "(a (b:1.5 :1.50)" here.
        out = run("(a (b:1.5):2.0)", multiplier=1.0)
        self.assertTrue(out.endswith(")"))
        self.assertEqual(out.count("("), out.count(")"))


class TestLoRAPreservation(unittest.TestCase):
    """M1/m5: LoRA tags are preserved and not re-enclosed."""

    def test_lora_tag_alone_not_wrapped(self):
        # A LoRA tag on its own line must be restored verbatim, not wrapped.
        self.assertEqual(
            run("<lora:style:1.0>", multiplier=1.5),
            "<lora:style:1.0>",
        )

    def test_lora_tag_in_own_paragraph_not_wrapped(self):
        # Paragraph consisting solely of a LoRA tag is left unwrapped.
        out = run("a cat\n\n<lora:style:1.0>\n\na dog", multiplier=1.5)
        self.assertIn("<lora:style:1.0>", out)
        self.assertNotIn("(<lora:style:1.0>", out)
        self.assertIn("(a cat:1.50)", out)
        self.assertIn("(a dog:1.50)", out)

    def test_lora_tag_inline_not_wrapped(self):
        # A paragraph containing a LoRA tag is left unwrapped so the tag is
        # not nested inside weighted syntax.
        out = run("a cute cat <lora:style:1.0>", multiplier=1.5)
        self.assertNotIn("(", out)
        self.assertIn("<lora:style:1.0>", out)

    def test_lora_tag_preserved_alongside_weighted_group(self):
        out = run("(fruit:1.0) <lora:style:1.0>", multiplier=2.0)
        self.assertIn("(fruit:2.00)", out)
        self.assertIn("<lora:style:1.0>", out)

    def test_multiple_lora_tags_preserved(self):
        out = run("<lora:a:1.0> and <lora:b:0.5>", multiplier=1.5)
        self.assertIn("<lora:a:1.0>", out)
        self.assertIn("<lora:b:0.5>", out)

    def test_placeholder_does_not_collide_with_literal_text(self):
        # m5: a literal "@@LORA_0@@" in the prompt must survive unchanged.
        out = run("@@LORA_0@@ and <lora:style:1.0>", multiplier=1.5)
        self.assertIn("@@LORA_0@@", out)
        self.assertIn("<lora:style:1.0>", out)


class TestCRLFNormalization(unittest.TestCase):
    """m2: CRLF/CR paragraph breaks are normalized to LF."""

    def test_crlf_paragraph_break_normalized(self):
        out = run("a cat\r\n\r\na dog", multiplier=1.5)
        self.assertNotIn("\r", out)
        self.assertEqual(out, "(a cat:1.50)\n\n(a dog:1.50)")

    def test_cr_only_paragraph_break_normalized(self):
        out = run("a cat\r\ra dog", multiplier=1.5)
        self.assertNotIn("\r", out)

    def test_lf_unchanged(self):
        out = run("a cat\n\na dog", multiplier=1.5)
        self.assertEqual(out, "(a cat:1.50)\n\n(a dog:1.50)")


class TestPlainTextWrapping(unittest.TestCase):
    """Step 3: plain-text paragraphs are wrapped in weighted groups."""

    def test_plain_paragraph_wrapped(self):
        self.assertEqual(run("a cat", multiplier=1.5), "(a cat:1.50)")

    def test_paragraph_with_group_not_wrapped(self):
        out = run("(a:1.0) text", multiplier=1.5)
        # The paragraph already contains a weighted group, so it is left
        # unchanged (the group itself was processed in Step 2).
        self.assertNotIn("((a:1.50) text", out)

    def test_empty_paragraph_not_wrapped(self):
        self.assertEqual(run("", multiplier=1.5), "")

    def test_whitespace_paragraph_not_wrapped(self):
        self.assertEqual(run("   ", multiplier=1.5), "   ")

    def test_multiple_paragraphs_each_wrapped(self):
        out = run("a\n\nb", multiplier=2.0)
        self.assertEqual(out, "(a:2.00)\n\n(b:2.00)")


class TestIndividualCapAndMin(unittest.TestCase):
    """Individual cap/min without total cap."""

    def test_cap_clamps_high_value(self):
        out = run("(a:5.0)", multiplier=1.0, individual_cap_enabled=True, individual_cap=2.0)
        self.assertEqual(out, "(a:2.00)")

    def test_min_raises_low_value(self):
        out = run("(a:0.5)", multiplier=1.0, individual_min_enabled=True, individual_min=1.0)
        self.assertEqual(out, "(a:1.00)")

    def test_cap_then_min_resolves_to_min_when_cap_below_min(self):
        # m6: cap applied first, min second; cap < min -> min wins.
        out = run(
            "(a:5.0)",
            multiplier=1.0,
            individual_cap_enabled=True,
            individual_cap=1.0,
            individual_min_enabled=True,
            individual_min=2.0,
        )
        self.assertEqual(out, "(a:2.00)")

    def test_cap_and_min_both_off_no_clamp(self):
        out = run("(a:5.0)", multiplier=2.0)
        self.assertEqual(out, "(a:10.00)")

    def test_cap_applies_to_wrapped_plain_text(self):
        out = run("a cat", multiplier=5.0, individual_cap_enabled=True, individual_cap=2.0)
        self.assertEqual(out, "(a cat:2.00)")


class TestTotalCap(unittest.TestCase):
    """M3: total-cap rescaling with individual minimum precedence."""

    def test_total_cap_scales_proportionally(self):
        # 3 groups summing to 6.0, cap 3.0 -> each halved to 1.0.
        out = run("(a:2.0)(b:2.0)(c:2.0)", total_cap_enabled=True, total_cap=3.0)
        self.assertEqual(strengths(out), [1.0, 1.0, 1.0])
        self.assertAlmostEqual(sum(strengths(out)), 3.0)

    def test_total_cap_not_applied_when_under_cap(self):
        out = run("(a:1.0)(b:1.0)", total_cap_enabled=True, total_cap=5.0)
        self.assertEqual(strengths(out), [1.0, 1.0])

    def test_total_cap_respects_individual_minimum(self):
        # total 12.0, cap 3.0, min 0.8. a and b would scale to 0.25 (< 0.8),
        # so they are pinned at 0.8 and c absorbs the remaining budget.
        out = run(
            "(a:1.0)(b:1.0)(c:10.0)",
            individual_min_enabled=True,
            individual_min=0.8,
            total_cap_enabled=True,
            total_cap=3.0,
        )
        s = strengths(out)
        self.assertEqual(s[0], 0.8)
        self.assertEqual(s[1], 0.8)
        # Remaining budget 3.0 - 1.6 = 1.4 goes to c.
        self.assertAlmostEqual(s[2], 1.4)
        self.assertAlmostEqual(sum(s), 3.0)

    def test_total_cap_sum_does_not_exceed_cap(self):
        out = run(
            "(a:3.0)(b:3.0)(c:3.0)",
            individual_min_enabled=True,
            individual_min=0.5,
            total_cap_enabled=True,
            total_cap=4.0,
        )
        self.assertLessEqual(sum(strengths(out)), 4.0 + 1e-9)

    def test_minimums_exceeding_cap_all_set_to_minimum(self):
        # If minimums alone meet or exceed the cap, every section is set to
        # its minimum (the closest feasible point).
        out = run(
            "(a:1.0)(b:1.0)",
            individual_min_enabled=True,
            individual_min=3.0,
            total_cap_enabled=True,
            total_cap=2.0,
        )
        self.assertEqual(strengths(out), [3.0, 3.0])

    def test_total_cap_with_individual_cap(self):
        # Individual cap clamps high values during rescale.
        out = run(
            "(a:10.0)(b:10.0)",
            individual_cap_enabled=True,
            individual_cap=1.5,
            total_cap_enabled=True,
            total_cap=3.0,
        )
        s = strengths(out)
        self.assertLessEqual(max(s), 1.5 + 1e-9)

    def test_total_cap_zero_groups_no_error(self):
        out = run("plain text", total_cap_enabled=True, total_cap=1.0)
        # No weighted groups with explicit strength -> no rescale, plain
        # text is wrapped.
        self.assertIn("plain text", out)


class TestNodeMetadata(unittest.TestCase):
    """Node class metadata and mappings are correct."""

    def test_return_types(self):
        self.assertEqual(Node.RETURN_TYPES, ("STRING",))

    def test_return_names(self):
        self.assertEqual(Node.RETURN_NAMES, ("modified_text",))

    def test_function(self):
        self.assertEqual(Node.FUNCTION, "process")

    def test_category(self):
        self.assertEqual(Node.CATEGORY, "illumorae")

    def test_output_node(self):
        self.assertFalse(Node.OUTPUT_NODE)

    def test_input_types_required_keys(self):
        required = Node.INPUT_TYPES()["required"]
        expected = {
            "text",
            "multiplier",
            "individual_cap_enabled",
            "individual_cap",
            "individual_min_enabled",
            "individual_min",
            "total_cap_enabled",
            "total_cap",
        }
        self.assertEqual(set(required.keys()), expected)

    def test_input_types_no_empty_optional_or_hidden(self):
        # n1: empty optional/hidden dicts were removed.
        self.assertNotIn("optional", Node.INPUT_TYPES())
        self.assertNotIn("hidden", Node.INPUT_TYPES())

    def test_node_class_mappings(self):
        self.assertIn("illumoraeTextStrengthMultiplierNode", NODE_CLASS_MAPPINGS)
        self.assertIs(
            NODE_CLASS_MAPPINGS["illumoraeTextStrengthMultiplierNode"], Node
        )

    def test_node_display_name_mappings(self):
        self.assertEqual(
            NODE_DISPLAY_NAME_MAPPINGS["illumoraeTextStrengthMultiplierNode"],
            "Text Strength Multiplier",
        )


class TestEdgeCases(unittest.TestCase):
    """Miscellaneous edge cases."""

    def test_unbalanced_open_paren_in_input(self):
        # An unbalanced "(" is skipped by the scanner (no group emitted);
        # the surrounding text is preserved and Step 3 wraps the paragraph.
        # The key guarantee is no crash and the text survives intact.
        out = run("before (unbalanced after", multiplier=2.0)
        self.assertIn("unbalanced", out)
        self.assertIn("before", out)
        self.assertIn("after", out)
        # The wrap adds one balanced pair; the unbalanced "(" is still
        # present as text but the output has balanced outer parens.
        self.assertTrue(out.startswith("("))
        self.assertTrue(out.endswith(")"))

    def test_prose_parenthetical_with_colon_not_treated_as_weight(self):
        # "(see figure 3: ref)" has a colon but no trailing number, so it is
        # treated as a group with no explicit strength and gets the default.
        out = run("(see figure 3: ref)", multiplier=2.0)
        self.assertEqual(out, "(see figure 3: ref:2.00)")

    def test_zero_multiplier(self):
        out = run("(a:5.0)", multiplier=0.0)
        self.assertEqual(out, "(a:0.00)")

    def test_return_is_tuple_of_one_string(self):
        out = Node().process("(a:1.0)", 2.0, False, 1.0, False, 0.0, False, 1.0)
        self.assertIsInstance(out, tuple)
        self.assertEqual(len(out), 1)
        self.assertIsInstance(out[0], str)


if __name__ == "__main__":
    unittest.main()
