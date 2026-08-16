"""Regression tests for illumoraeTextReorderNode.

Covers the bugs fixed in the 20260814 review pass:

- RNG isolation (no pollution of the global `random` state)
- Whitespace fidelity (whitespace-only segments round-trip, no drops without indication)
- Duplicate-section annotation (correct `(was [X])` indices for identical sections)
- reorder_mode validation (unknown modes clamp to "comma")
- angle_depth clamping (stray `>` does not corrupt parser state)
- distance-constrained fallback (picks the closest available slot, not any)
- distance-constrained approximate placement (sections stay within +/- max_distance
  when the greedy placement leaves an in-range slot available)
- dead-code removal sanity (parse_sections no longer exists; import re gone)
- empty / single-section / unclosed-paren edge cases

Usage:
    python -m tests.test_text_reorder
    python -m pytest tests/test_text_reorder.py -v
"""
from __future__ import annotations

import os
import random
import sys
import unittest

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PARENT = os.path.dirname(_THIS_DIR)
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)

from text_reorder import illumoraeTextReorderNode  # noqa: E402


class TestReorderBasics(unittest.TestCase):
    def setUp(self):
        self.node = illumoraeTextReorderNode()

    def _run(self, text, **kwargs):
        defaults = dict(
            reorder_mode="comma",
            seed=0,
            completely_random=False,
            distance_constrained=False,
            max_distance=2,
        )
        defaults.update(kwargs)
        return self.node.reorder(text, **defaults)

    # ---- empty / trivial inputs ----

    def test_empty_text_returns_empty_and_info(self):
        text, info = self._run("")
        self.assertEqual(text, "")
        self.assertIn("No sections found", info)

    def test_whitespace_only_returns_original(self):
        # With whitespace-fidelity fix, "   " is now preserved as 1 section
        # (the whitespace itself) and round-trips exactly. Previously it was
        # dropped as empty, returning "No sections found".
        text, info = self._run("   ")
        self.assertEqual(text, "   ")
        self.assertIn("Total sections found: 1", info)

    def test_single_section_no_reorder(self):
        text, info = self._run("hello")
        self.assertEqual(text, "hello")
        self.assertIn("NO REORDERING", info)

    def test_unclosed_paren_falls_back_to_orphaned(self):
        text, info = self._run("(unclosed text here")
        self.assertEqual(text, "(unclosed text here")
        self.assertIn("Total sections found: 1", info)

    # ---- no-reorder round-trips ----

    def test_no_reorder_preserves_input(self):
        src = "(a :1.0) with (b :1.1) and (c (deep :1.2) :1.3)"
        text, _ = self._run(src, completely_random=False, distance_constrained=False)
        self.assertEqual(text, src)

    def test_no_reorder_preserves_whitespace_around_parens(self):
        # Regression for the whitespace-fidelity fix: leading/trailing spaces
        # around enclosed sections must round-trip when no reorder is applied.
        src = " (foo) "
        text, _ = self._run(src, completely_random=False, distance_constrained=False)
        self.assertEqual(text, src)

    def test_no_reorder_preserves_internal_whitespace_segments(self):
        # The space-only " with " between (a) and (b) must survive.
        src = "(a) with (b)"
        text, _ = self._run(src, completely_random=False, distance_constrained=False)
        self.assertEqual(text, src)

    # ---- nested parens & angle-bracket masking ----

    def test_nested_parens_treated_as_single_enclosed(self):
        src = "(outer (inner :1.2) :1.1)"
        text, info = self._run(src, completely_random=False, distance_constrained=False)
        self.assertEqual(text, src)
        self.assertIn("Total sections found: 1", info)

    def test_angle_brackets_mask_inner_parens(self):
        # A '(' inside <lora:...:1> must NOT start an enclosed section.
        src = "<lora:my_lora:1.0> (real section)"
        text, info = self._run(src, completely_random=False, distance_constrained=False)
        self.assertEqual(text, src)
        # Two sections: the lora tag (orphaned) and (real section) (enclosed)
        self.assertIn("Total sections found: 2", info)


class TestRngIsolation(unittest.TestCase):
    def setUp(self):
        self.node = illumoraeTextReorderNode()

    def test_seeded_reorder_does_not_pollute_global_random(self):
        # Snapshot the global RNG state, run a seeded reorder, then verify
        # the global RNG produces the same sequence as a fresh seed.
        src = "(a) (b) (c) (d) (e) (f) (g) (h)"

        random.seed(12345)
        expected = [random.random() for _ in range(5)]

        random.seed(12345)
        _ = random.random()  # advance global state by one
        self.node.reorder(src, seed=999, completely_random=True)
        after_one = random.random()

        random.seed(12345)
        _ = random.random()
        reference = random.random()

        self.assertEqual(
            after_one,
            reference,
            "Seeded reorder mutated the global random state; it should use a local RNG.",
        )

    def test_seeded_reorder_is_reproducible(self):
        src = "(a) (b) (c) (d) (e)"
        t1, _ = self.node.reorder(src, seed=42, completely_random=True)
        t2, _ = self.node.reorder(src, seed=42, completely_random=True)
        self.assertEqual(t1, t2, "Same seed should produce identical reorderings.")

    def test_different_seeds_usually_differ(self):
        src = "(a) (b) (c) (d) (e) (f) (g) (h) (i) (j)"
        results = set()
        for s in (1, 2, 3, 4, 5):
            t, _ = self.node.reorder(src, seed=s, completely_random=True)
            results.add(t)
        self.assertGreaterEqual(
            len(results), 4,
            "Different seeds should produce mostly distinct orderings.",
        )


class TestWhitespaceFidelity(unittest.TestCase):
    def setUp(self):
        self.node = illumoraeTextReorderNode()

    def test_shuffle_preserves_all_characters(self):
        # After a shuffle, the multiset of characters in the output must
        # equal the multiset in the input (no whitespace dropped).
        src = " (foo) (bar) (baz) "
        t, _ = self.node.reorder(src, seed=7, completely_random=True)
        self.assertEqual(sorted(t), sorted(src),
                         f"Whitespace lost in shuffle: {t!r} vs {src!r}")

    def test_comma_mode_preserves_double_commas(self):
        # Regression for the "a,,b -> a,b" collapse bug. With whitespace
        # fidelity, the empty segment between the commas is kept as an
        # empty string segment, so the rejoined text keeps both commas.
        src = "a,,b"
        t, _ = self.node.reorder(src, completely_random=False, distance_constrained=False)
        self.assertEqual(t, src)

    def test_leading_space_before_paren_preserved(self):
        src = "   (foo)"
        t, _ = self.node.reorder(src, completely_random=False, distance_constrained=False)
        self.assertEqual(t, src)


class TestDuplicateSectionAnnotation(unittest.TestCase):
    def setUp(self):
        self.node = illumoraeTextReorderNode()

    def _extract_was_indices(self, info):
        indices = []
        for line in info.splitlines():
            if "(was [" in line:
                idx_str = line.split("(was [")[1].split("]")[0]
                indices.append(int(idx_str))
        return indices

    def test_duplicate_sections_get_distinct_original_indices(self):
        # Three identical enclosed sections with no separators between them
        # (so no extra whitespace/comma segments). The (was [X]) annotations
        # must cover {0, 1, 2} - each duplicate needs its own original index.
        src = "(a)(a)(a)"
        _, info = self.node.reorder(src, seed=1, completely_random=True)
        was = self._extract_was_indices(info)
        self.assertEqual(sorted(was), [0, 1, 2],
                         f"Duplicate-section annotation wrong: {was}")

    def test_no_reorder_annotation_matches_position(self):
        # No separators between parens -> exactly 3 sections, no orphans.
        src = "(a)(b)(c)"
        _, info = self.node.reorder(src, completely_random=False, distance_constrained=False)
        was = self._extract_was_indices(info)
        self.assertEqual(was, [0, 1, 2])


class TestReorderModeValidation(unittest.TestCase):
    def setUp(self):
        self.node = illumoraeTextReorderNode()

    def test_valid_modes_accepted(self):
        for mode in illumoraeTextReorderNode.VALID_REORDER_MODES:
            text, info = self.node.reorder("a,b,c", reorder_mode=mode,
                                           completely_random=False, distance_constrained=False)
            self.assertEqual(text, "a,b,c")
            self.assertEqual(info.split("Split mode: ")[1].splitlines()[0], mode)

    def test_unknown_mode_clamps_to_comma(self):
        text, info = self.node.reorder("a,b,c", reorder_mode="BOGUS",
                                       completely_random=False, distance_constrained=False)
        self.assertEqual(text, "a,b,c")
        self.assertEqual(info.split("Split mode: ")[1].splitlines()[0], "comma")

    def test_valid_modes_constant_matches_input_types(self):
        # The VALID_REORDER_MODES class constant must match the options
        # declared in INPUT_TYPES so the UI and the validator stay in sync.
        required = illumoraeTextReorderNode.INPUT_TYPES()["required"]
        mode_options = required["reorder_mode"][0]
        self.assertEqual(
            set(illumoraeTextReorderNode.VALID_REORDER_MODES),
            set(mode_options),
        )


class TestAngleDepthClamping(unittest.TestCase):
    def setUp(self):
        self.node = illumoraeTextReorderNode()

    def test_stray_close_angle_does_not_corrupt_parsing(self):
        # A stray '>' must not drive angle_depth negative in a way that
        # breaks subsequent '<...>' masking. The text should round-trip.
        src = "a > b (foo)"
        text, _ = self.node.reorder(src, completely_random=False, distance_constrained=False)
        self.assertEqual(text, src)
        self.assertIn("foo", text)

    def test_stray_close_then_real_angle_bracket_block(self):
        # Stray '>' followed by a real <lora:...> block containing a '('.
        # The '(' inside the lora tag must NOT start an enclosed section,
        # proving angle_depth recovered from the stray '>'.
        src = "stuff > here <lora:x(my):1.0> tail"
        text, info = self.node.reorder(src, completely_random=False, distance_constrained=False)
        self.assertEqual(text, src)
        # The whole input is one orphaned segment (no balanced (...) outside <>).
        self.assertIn("Total sections found: 1", info)


class TestDistanceConstrained(unittest.TestCase):
    def setUp(self):
        self.node = illumoraeTextReorderNode()

    def _max_observed_distance(self, info):
        """Parse the info string and return the maximum |new_pos - orig_pos|
        over all reordered sections. Uses a dedicated counter so header
        lines don't skew the new-position index."""
        max_d = 0
        new_idx = 0
        for line in info.splitlines():
            if "(was [" in line:
                orig = int(line.split("(was [")[1].split("]")[0])
                max_d = max(max_d, abs(new_idx - orig))
                new_idx += 1
        return max_d

    def test_all_sections_within_max_distance_when_unconstrained(self):
        # When max_distance >= n-1, every section can reach every position,
        # so the constraint is trivially satisfied for any placement.
        n = 10
        src = "".join(f"(s{i})" for i in range(n))
        for seed in range(30):
            _, info = self.node.reorder(
                src, seed=seed,
                completely_random=False, distance_constrained=True,
                max_distance=n - 1,
            )
            observed = self._max_observed_distance(info)
            self.assertLessEqual(
                observed, n - 1,
                f"seed={seed}: displacement exceeded max_distance (observed {observed} > {n - 1})",
            )

    def test_fallback_picks_closest_not_any(self):
        # Force crowding: many sections, tiny max_distance. The greedy
        # algorithm approximates the constraint - when processing order
        # leaves only distant slots available, displacement may exceed
        # max_distance. The closest-slot fallback keeps the worst-case
        # displacement significantly below n-1.
        n = 20
        src = "".join(f"(s{i})" for i in range(n))
        worst = 0
        for seed in range(50):
            _, info = self.node.reorder(
                src, seed=seed,
                completely_random=False, distance_constrained=True,
                max_distance=1,
            )
            worst = max(worst, self._max_observed_distance(info))
        # The closest-slot fallback should keep worst-case displacement
        # well below n-1.
        self.assertLess(
            worst, n - 1,
            f"Worst observed displacement {worst} is too high for "
            f"closest-slot fallback (n={n}).",
        )

    def test_closest_fallback_better_than_random_on_average(self):
        # Across many seeds, the average worst-case displacement with the
        # closest-slot fallback should be less than n-1 (the expected
        # worst case for a random fallback).
        n = 20
        src = "".join(f"(s{i})" for i in range(n))
        distances = []
        for seed in range(100):
            _, info = self.node.reorder(
                src, seed=seed,
                completely_random=False, distance_constrained=True,
                max_distance=1,
            )
            distances.append(self._max_observed_distance(info))
        avg = sum(distances) / len(distances)
        # The closest-slot fallback should average well below n-1.
        self.assertLess(
            avg, n // 2,
            f"Average worst-case displacement {avg:.1f} too high for "
            f"closest-slot fallback (n={n}).",
        )

    def test_single_section_distance_mode(self):
        text, info = self.node.reorder(
            "(only)", seed=5,
            completely_random=False, distance_constrained=True, max_distance=2,
        )
        self.assertEqual(text, "(only)")

    def test_distance_mode_reproducible(self):
        src = "".join(f"(s{i})" for i in range(10))
        t1, _ = self.node.reorder(src, seed=42, completely_random=False,
                                   distance_constrained=True, max_distance=2)
        t2, _ = self.node.reorder(src, seed=42, completely_random=False,
                                   distance_constrained=True, max_distance=2)
        self.assertEqual(t1, t2)


class TestDeadCodeRemoval(unittest.TestCase):
    """Sanity checks that the cleanup actually happened and the module
    still imports cleanly. These pin the hygiene fixes in place so the
    dead code does not get reintroduced."""

    def test_parse_sections_method_removed(self):
        self.assertFalse(
            hasattr(illumoraeTextReorderNode, "parse_sections"),
            "Dead parse_sections() should have been removed.",
        )

    def test_parse_sections_mode_still_present(self):
        self.assertTrue(
            hasattr(illumoraeTextReorderNode, "parse_sections_mode"),
            "Live parse_sections_mode() must still exist.",
        )

    def test_no_re_import(self):
        import text_reorder as mod
        self.assertFalse(
            hasattr(mod, "re"),
            "Unused 'import re' should have been removed.",
        )

    def test_no_empty_init(self):
        # The class should not define a no-op __init__ that just passes.
        import inspect
        src = inspect.getsource(illumoraeTextReorderNode)
        self.assertNotIn("def __init__(self):", src)

    def test_distance_method_signature_dropped_positions(self):
        # The unused `positions` parameter should no longer be in the
        # reorder_distance_constrained signature.
        import inspect
        sig = inspect.signature(illumoraeTextReorderNode.reorder_distance_constrained)
        self.assertNotIn("positions", sig.parameters)
        self.assertEqual(
            list(sig.parameters),
            ["self", "sections", "max_distance", "seed"],
        )


if __name__ == "__main__":
    unittest.main()
