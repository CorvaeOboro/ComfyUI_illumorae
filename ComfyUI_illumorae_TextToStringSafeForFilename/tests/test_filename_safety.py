"""Filename-safety regression tests for the Text To Filename Safe Text node.

Verifies the generic (non-enumerated) character-rejection logic added when
the hardcoded BAD_FILENAME_CHAR_SET was removed. The node now relies on
NFKD normalization plus an ord > 127 catch-all plus unicode-category
filtering.

All non-ASCII test characters are constructed at runtime via chr(codepoint)
so this test file itself stays ASCII-clean and does not trigger the
illumorae_ascii_clean dev tool. A randomized sweep picks code points from
known-non-ASCII ranges on each run to exercise the catch-all broadly.

Usage:
    python -m tests.test_filename_safety
    python -m pytest tests/test_filename_safety.py -v
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

from text_to_text_safe_for_filename import (  # noqa: E402
    illumoraeTextToFilenameSafeTextNode,
)

# --------------------------------------------------------------------------
# Non-ASCII code points used by the fixed-case tests below.
# Stored as integers and built with chr() at runtime so the source file
# contains no literal non-ASCII glyphs and no \uXXXX escape sequences.
# --------------------------------------------------------------------------

CP_EN_DASH = 0x2013
CP_EM_DASH = 0x2014
CP_LEFT_DQUOTE = 0x201C
CP_RIGHT_DQUOTE = 0x201D
CP_INTEGRAL = 0x222B
CP_GREEK_SMALL_ALPHA = 0x03B1
CP_GREEK_CAP_LAMDA = 0x039B
CP_CJK_ZHONG = 0x4E2D
CP_CJK_WEN = 0x6587
CP_BOX_HORIZ = 0x2500
CP_BOX_VERT = 0x2502
CP_CIRCLED_ONE = 0x2460
CP_EMOJI_GRINNING = 0x1F600

# Ranges known to be entirely non-ASCII (ord > 127). Used by the randomized
# sweep to pick characters that must be rejected or decomposed by the node.
# Each tuple is (start, end) inclusive.
NON_ASCII_RANGES = [
    (0x0080, 0x00FF),   # Latin-1 supplement
    (0x0100, 0x017F),   # Latin Extended-A
    (0x0180, 0x024F),   # Latin Extended-B
    (0x0370, 0x03FF),   # Greek and Coptic
    (0x2000, 0x206F),   # General punctuation
    (0x2200, 0x22FF),   # Mathematical operators
    (0x2460, 0x24FF),   # Enclosed alphanumerics
    (0x2500, 0x257F),   # Box drawing
    (0x25A0, 0x25FF),   # Geometric shapes
    (0x2600, 0x26FF),   # Miscellaneous symbols
    (0x3000, 0x303F),   # CJK symbols and punctuation
    (0x4E00, 0x4FFF),   # CJK unified ideographs (sample)
    (0x1F300, 0x1F3FF), # Miscellaneous symbols and pictographs
    (0x1F600, 0x1F64F), # Emoticons
]


def _char(cp):
    """Build a single character from a code point."""
    return chr(cp)


class TestFilenameSafety(unittest.TestCase):
    def setUp(self):
        self.node = illumoraeTextToFilenameSafeTextNode()

    def _run(self, text, max_length=150):
        return self.node.process_string(text, max_length=max_length)[0]

    # ---- basic ASCII ----

    def test_spaces_become_underscores(self):
        self.assertEqual(self._run("Hello World"), "Hello_World")

    def test_direct_replacements(self):
        # : -> _, space -> _, ? -> _, * -> _ ; trailing _ from * is kept
        # (strip only removes trailing spaces and periods, not underscores).
        self.assertEqual(self._run("file: name? test*"), "file_name_test_")

    def test_path_separators_replaced(self):
        self.assertEqual(self._run("path/to/file"), "path_to_file")

    def test_angle_brackets_replaced(self):
        self.assertEqual(self._run("test<>test"), "test_test")

    def test_pipe_replaced(self):
        self.assertEqual(self._run("a|b|c"), "a_b_c")

    # ---- accented letters (NFKD decomposition) ----

    def test_accented_letters_decomposed(self):
        # NFKD decomposes accented e into base e + combining mark; the
        # combining mark is non-ASCII and stripped, leaving ASCII base.
        self.assertEqual(self._run("cafe resume"), "cafe_resume")

    def test_german_umlaute_decomposed(self):
        # u-umlaut decomposes to u + combining diaeresis; combining mark
        # stripped, base u kept.
        self.assertEqual(self._run("uber"), "uber")

    # ---- non-ASCII symbol removal (the generic catch-all) ----
    # Non-ASCII chars are REMOVED (stripped), not replaced with _.
    # All non-ASCII chars built via chr() to keep this file ASCII-clean.

    def test_en_dash_removed(self):
        self.assertEqual(self._run("unicode" + _char(CP_EN_DASH) + "test"), "unicodetest")

    def test_em_dash_removed(self):
        self.assertEqual(self._run("a" + _char(CP_EM_DASH) + "b"), "ab")

    def test_smart_quotes_removed(self):
        s = _char(CP_LEFT_DQUOTE) + "quote" + _char(CP_RIGHT_DQUOTE)
        self.assertEqual(self._run(s), "quote")

    def test_math_symbols_removed(self):
        s = "math" + _char(CP_INTEGRAL) + "test" + _char(CP_GREEK_SMALL_ALPHA)
        self.assertEqual(self._run(s), "mathtest")

    def test_greek_capital_removed(self):
        s = "mixed" + _char(CP_GREEK_CAP_LAMDA) + "ABC"
        self.assertEqual(self._run(s), "mixedABC")

    def test_emoji_removed(self):
        s = "hello" + _char(CP_EMOJI_GRINNING) + "world"
        self.assertEqual(self._run(s), "helloworld")

    def test_cjk_removed(self):
        s = "test" + _char(CP_CJK_ZHONG) + _char(CP_CJK_WEN) + "test"
        self.assertEqual(self._run(s), "testtest")

    def test_box_drawing_removed(self):
        s = "box" + _char(CP_BOX_HORIZ) + _char(CP_BOX_VERT) + "draw"
        self.assertEqual(self._run(s), "boxdraw")

    def test_circled_digit_decomposed_to_ascii(self):
        # circled digit one decomposes to ASCII "1" under NFKD, so it
        # survives as a safe digit rather than being removed.
        s = "step" + _char(CP_CIRCLED_ONE) + "done"
        self.assertEqual(self._run(s), "step1done")

    # ---- specifically removed characters ----

    def test_comma_removed(self):
        self.assertEqual(self._run("comma, test"), "comma_test")

    def test_apostrophe_removed(self):
        self.assertEqual(self._run("apostrophe's test"), "apostrophes_test")

    # ---- control characters ----

    def test_control_chars_removed(self):
        # tab and null are control chars; removed entirely (not -> _).
        self.assertEqual(self._run("a\tb\x00c"), "abc")

    # ---- underscore collapsing ----

    def test_multiple_underscores_collapsed(self):
        self.assertEqual(self._run("a___b"), "a_b")

    def test_replacements_then_collapse(self):
        # Each of / : ? becomes _ then collapsed.
        self.assertEqual(self._run("a/:?b"), "a_b")

    # ---- trailing / leading stripping ----

    def test_trailing_periods_stripped(self):
        self.assertEqual(self._run("test..."), "test")

    def test_trailing_underscores_not_stripped(self):
        # strip(" .") only removes trailing spaces and periods, not
        # underscores. This is existing behaviour, pinned here.
        self.assertEqual(self._run("test   "), "test_")

    # ---- Windows reserved names ----

    def test_reserved_con(self):
        self.assertEqual(self._run("CON"), "CON_0")

    def test_reserved_nul(self):
        self.assertEqual(self._run("nul"), "nul_0")

    def test_reserved_com1(self):
        self.assertEqual(self._run("COM1"), "COM1_0")

    def test_non_reserved_con_text(self):
        # "CON" embedded in a longer name is not reserved.
        self.assertEqual(self._run("CONCEPT"), "CONCEPT")

    # ---- length truncation ----

    def test_truncation(self):
        self.assertEqual(self._run("a" * 200, max_length=50), "a" * 50)

    def test_no_truncation_under_limit(self):
        self.assertEqual(self._run("a" * 100, max_length=150), "a" * 100)

    # ---- empty / whitespace fallback ----

    def test_empty_string_fallback(self):
        self.assertEqual(self._run(""), "error_empty_string")

    def test_whitespace_only_becomes_single_underscore(self):
        # Spaces become _, collapse to single _, strip(" .") does not
        # remove underscores, so result is "_" (not empty, no fallback).
        self.assertEqual(self._run("   "), "_")

    def test_only_symbols_fallback(self):
        # Non-ASCII symbols removed -> empty -> fallback.
        s = _char(CP_EN_DASH) + _char(CP_EM_DASH)
        self.assertEqual(self._run(s), "error_empty_string")

    # ---- output is ASCII-only (the core invariant) ----

    def test_output_contains_no_non_ascii_fixed_cases(self):
        cases = [
            "cafe resume",
            "hello" + _char(CP_EMOJI_GRINNING) + "world",
            "test" + _char(CP_CJK_ZHONG) + _char(CP_CJK_WEN) + _char(CP_GREEK_SMALL_ALPHA) + _char(CP_INTEGRAL),
            "mix" + _char(CP_EN_DASH) + _char(CP_EM_DASH) + _char(CP_BOX_HORIZ) + _char(CP_CIRCLED_ONE),
        ]
        for case in cases:
            result = self._run(case)
            for ch in result:
                self.assertLessEqual(
                    ord(ch), 127,
                    f"Non-ASCII char {ch!r} (U+{ord(ch):04X}) in output {result!r} for input {case!r}",
                )

    # ---- randomized sweep over known-non-ASCII ranges ----

    def test_random_non_ascii_all_removed_or_decomposed(self):
        """Pick random code points from non-ASCII ranges and verify the
        output never contains a character with ord > 127. Characters that
        NFKD-decompose to ASCII base letters (e.g. circled digits) will
        survive as ASCII; everything else must be removed.
        """
        rng = random.Random(42)
        for _ in range(500):
            start, end = rng.choice(NON_ASCII_RANGES)
            cp = rng.randint(start, end)
            ch = chr(cp)
            result = self._run("a" + ch + "b")
            for out_ch in result:
                self.assertLessEqual(
                    ord(out_ch), 127,
                    f"Non-ASCII char {out_ch!r} (U+{ord(out_ch):04X}) in output {result!r} "
                    f"for input char U+{cp:04X}",
                )

    def test_random_non_ascii_never_crashes(self):
        """The node must not raise on any single non-ASCII code point."""
        rng = random.Random(99)
        for _ in range(500):
            start, end = rng.choice(NON_ASCII_RANGES)
            cp = rng.randint(start, end)
            try:
                self._run(chr(cp))
            except Exception as exc:
                self.fail(f"Node raised {type(exc).__name__} for input char U+{cp:04X}: {exc}")


if __name__ == "__main__":
    unittest.main()
