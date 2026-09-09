"""
Regression tests for illumoraeTextTruncateProtectNode.

Run via:
    python -m ComfyUI_illumorae_TextTruncateProtect.tests.test_text_truncate_protect
or:
    python -m unittest ComfyUI_illumorae_TextTruncateProtect.tests.test_text_truncate_protect
"""
import os
import sys
import unittest

# Allow running this file directly without package install.
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(THIS_DIR)
GRANDPARENT_DIR = os.path.dirname(PARENT_DIR)
for p in (PARENT_DIR, GRANDPARENT_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

try:
    from ComfyUI_illumorae_TextTruncateProtect.text_truncate_protect import (
        illumoraeTextTruncateProtectNode,
    )
except ImportError:
    from text_truncate_protect import illumoraeTextTruncateProtectNode  # type: ignore


class TestTextTruncateProtect(unittest.TestCase):
    def setUp(self):
        self.node = illumoraeTextTruncateProtectNode()

    # ---- Sanity baselines ----

    def test_under_limit_unchanged(self):
        text = "hello world"
        out, _stats, final_len, removed = self.node.truncate_text(
            text=text, character_limit=100, threshold=10,
        )
        self.assertEqual(out, text)
        self.assertEqual(final_len, len(text))
        self.assertEqual(removed, 0)

    def test_basic_truncation_no_protection(self):
        text = "abcdefghij"
        out, _stats, final_len, _ = self.node.truncate_text(
            text=text, character_limit=5, threshold=0,
        )
        self.assertEqual(out, "abcde")
        self.assertEqual(final_len, 5)

    # ---- Enclosure protection regression ----

    def test_enclosure_includes_closing_paren_at_boundary(self):
        """Bug: closing ')' was being dropped when limit fell exactly on it."""
        text = "abc(def)ghij"  # length 12, ')' at index 7
        out, _stats, _final_len, _ = self.node.truncate_text(
            text=text, character_limit=7, threshold=2,
            protect_enclosures=True, enclosure_types="all",
        )
        # Must include the full "(def)" enclosure, not "abc(def".
        self.assertIn("(def)", out, f"closing paren dropped: {out!r}")
        self.assertFalse(
            out.endswith("(def") or out.endswith("(def "),
            f"truncated mid-enclosure: {out!r}",
        )

    def test_enclosure_includes_closing_when_limit_inside(self):
        """When limit lands inside enclosure, threshold should extend to include closer."""
        text = "ab(cdefgh)ij"  # ')' at index 9
        out, _stats, _final_len, _ = self.node.truncate_text(
            text=text, character_limit=6, threshold=5,
            protect_enclosures=True,
        )
        self.assertTrue(
            out == "ab" or out.endswith(")"),
            f"expected to either skip enclosure or include closer, got {out!r}",
        )

    def test_enclosure_all_bracket_types(self):
        for opener, closer in [("(", ")"), ("{", "}"), ("[", "]"), ("<", ">")]:
            text = f"abc{opener}def{closer}ghij"
            out, _stats, _final_len, _ = self.node.truncate_text(
                text=text, character_limit=7, threshold=2,
                protect_enclosures=True, enclosure_types="all",
            )
            with self.subTest(pair=opener + closer):
                self.assertIn(f"{opener}def{closer}", out, f"dropped {closer}: {out!r}")

    def test_nested_enclosures_preserved(self):
        text = "ab(c(d)e)fghij"  # outer ')' at index 8
        out, _stats, _final_len, _ = self.node.truncate_text(
            text=text, character_limit=8, threshold=3,
            protect_enclosures=True,
        )
        # Either truncate before the outer '(' or include the full outer group.
        self.assertTrue(
            "(" not in out or out.count("(") == out.count(")"),
            f"unbalanced parens in output: {out!r}",
        )

    def test_enclosure_no_truncation_needed(self):
        text = "abc(def)"
        out, _stats, final_len, removed = self.node.truncate_text(
            text=text, character_limit=100, threshold=10,
            protect_enclosures=True,
        )
        self.assertEqual(out, text)
        self.assertEqual(final_len, len(text))
        self.assertEqual(removed, 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
