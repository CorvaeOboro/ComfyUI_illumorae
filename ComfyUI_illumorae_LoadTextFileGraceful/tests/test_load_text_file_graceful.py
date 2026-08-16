"""Regression tests for illumoraeLoadTextFileGracefulNode.

Covers the bugs fixed in the 20260816 review pass:

- Return arity: load_file returns exactly 2 values matching RETURN_TYPES /
  RETURN_NAMES on every path (missing file, read error, success). The old
  code declared 3 outputs (STRING, DICT, LABEL) but only returned 2.
- LABEL dropped in favor of STRING for the status output.
- Comment rule: a line is a comment when its first non-whitespace character
  is '#'; inline trailing '#' is preserved; indented comments are skipped.
- Line preservation: original line text (including indentation) is kept;
  only the line terminator is removed. The old code strip()-ed every line.
- Non-UTF-8 fallback: a non-UTF-8 file is decoded lossily with
  errors='replace' instead of returning an empty string.
- Narrowed except: only OSError is treated as a read failure; programming
  errors are no longer masked behind a graceful status.
- IS_CHANGED: folds file mtime + size into the cache key so editing the
  text file refreshes the output; a missing file yields a distinct key.
- Status strings include the full path and distinguish a file that loaded
  with content from one that was empty after comment stripping.

Usage:
    python -m tests.test_load_text_file_graceful
    python -m pytest tests/test_load_text_file_graceful.py -v
"""
from __future__ import annotations

import os
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PARENT = os.path.dirname(_THIS_DIR)
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)

from load_text_file_graceful import (  # noqa: E402
    NODE_CLASS_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS,
    illumoraeLoadTextFileGracefulNode,
)

Node = illumoraeLoadTextFileGracefulNode


class TestLoadFile(unittest.TestCase):
    """load_file: end-to-end behavior and edge cases."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.node = Node()

    def tearDown(self):
        self.tmp.cleanup()

    def _write(self, name, text):
        p = self.root / name
        p.write_text(text, encoding="utf-8")
        return str(p)

    def _call(self, **kw):
        defaults = dict(file_path="", debug_mode=False)
        defaults.update(kw)
        return self.node.load_file(**defaults)

    def test_return_arity_is_two_on_success(self):
        # Regression: old code declared 3 outputs but returned 2.
        p = self._write("a.txt", "hello\nworld\n")
        out = self._call(file_path=p)
        self.assertEqual(len(out), 2)
        self.assertEqual(len(Node.RETURN_TYPES), 2)
        self.assertEqual(len(Node.RETURN_NAMES), 2)

    def test_return_arity_is_two_on_missing_file(self):
        out = self._call(file_path=str(self.root / "missing.txt"))
        self.assertEqual(len(out), 2)

    def test_return_arity_is_two_on_read_error(self):
        p = self._write("a.txt", "hello")
        real_open = open

        def fake_open(path, *args, **kwargs):
            if str(path) == p:
                raise PermissionError("simulated permission denied")
            return real_open(path, *args, **kwargs)

        with mock.patch("builtins.open", side_effect=fake_open):
            out = self._call(file_path=p)
        self.assertEqual(len(out), 2)

    def test_missing_file_returns_empty_and_status(self):
        missing = str(self.root / "missing.txt")
        text, status = self._call(file_path=missing)
        self.assertEqual(text, "")
        self.assertIn("File not found", status)
        self.assertIn(missing, status)

    def test_empty_path_returns_empty_and_status(self):
        text, status = self._call(file_path="")
        self.assertEqual(text, "")
        self.assertIn("File not found", status)

    def test_directory_path_treated_as_not_a_file(self):
        # isfile() is False for a directory; must not attempt to read it.
        text, status = self._call(file_path=str(self.root))
        self.assertEqual(text, "")
        self.assertIn("File not found", status)

    def test_simple_file_no_comments(self):
        p = self._write("a.txt", "line one\nline two\n")
        text, status = self._call(file_path=p)
        self.assertEqual(text, "line one\nline two")
        self.assertIn("Loaded", status)
        self.assertIn(p, status)

    def test_comment_lines_are_skipped(self):
        p = self._write("a.txt", "# header\nkeep me\n# trailer\ndone\n")
        text, status = self._call(file_path=p)
        self.assertEqual(text, "keep me\ndone")
        self.assertIn("Loaded", status)

    def test_indented_comment_is_skipped(self):
        # First non-whitespace character is '#', so this is a comment.
        p = self._write("a.txt", "    # indented comment\nkeep\n")
        text, _ = self._call(file_path=p)
        self.assertEqual(text, "keep")

    def test_inline_trailing_hash_is_preserved(self):
        # Inline '#' is not a comment; the line is kept verbatim.
        p = self._write("a.txt", "text # note\n")
        text, _ = self._call(file_path=p)
        self.assertEqual(text, "text # note")

    def test_indentation_is_preserved(self):
        # Regression: old code strip()-ed every line, dropping indentation.
        p = self._write("a.txt", "    indented\n        deeper\nflat\n")
        text, _ = self._call(file_path=p)
        self.assertEqual(text, "    indented\n        deeper\nflat")

    def test_blank_lines_are_preserved(self):
        p = self._write("a.txt", "a\n\nb\n")
        text, _ = self._call(file_path=p)
        self.assertEqual(text, "a\n\nb")

    def test_file_with_only_comments_returns_empty_with_loaded_status(self):
        p = self._write("a.txt", "# only\n# comments\n")
        text, status = self._call(file_path=p)
        self.assertEqual(text, "")
        # Distinct from the missing-file empty string: status says "Loaded".
        self.assertIn("Loaded", status)
        self.assertIn("empty after comment strip", status)

    def test_empty_file_returns_empty_with_loaded_status(self):
        p = self._write("a.txt", "")
        text, status = self._call(file_path=p)
        self.assertEqual(text, "")
        self.assertIn("Loaded", status)
        self.assertIn("empty after comment strip", status)

    def test_status_includes_line_count_when_content_present(self):
        p = self._write("a.txt", "a\nb\nc\n")
        _, status = self._call(file_path=p)
        self.assertIn("3 lines", status)

    def test_non_utf8_file_falls_back_to_lossy_decode(self):
        # Bytes that are invalid UTF-8; errors='replace' must still yield a
        # usable string instead of an empty-string error return.
        p = self.root / "bad.txt"
        p.write_bytes(b"\xff\xfe bad \x00 bytes\nkeep\n")
        text, status = self._call(file_path=str(p))
        self.assertIsInstance(text, str)
        self.assertIn("keep", text)
        self.assertIn("Loaded", status)

    def test_oserror_during_read_returns_status(self):
        # Narrowed except: OSError surfaces as a status; the path is included.
        p = self._write("a.txt", "hello")
        real_open = open

        def fake_open(path, *args, **kwargs):
            if str(path) == p:
                raise PermissionError("simulated permission denied")
            return real_open(path, *args, **kwargs)

        with mock.patch("builtins.open", side_effect=fake_open):
            text, status = self._call(file_path=p)
        self.assertEqual(text, "")
        self.assertIn("Error reading", status)
        self.assertIn(p, status)
        self.assertIn("simulated permission denied", status)

    def test_programming_error_is_not_masked(self):
        # Narrowed except: a non-OSError exception must propagate, not be
        # swallowed into a graceful status string.
        p = self._write("a.txt", "hello")

        def fake_open(*args, **kwargs):
            raise ValueError("not an os error")

        with mock.patch("builtins.open", side_effect=fake_open):
            with self.assertRaises(ValueError):
                self._call(file_path=p)

    def test_debug_mode_does_not_raise(self):
        p = self._write("a.txt", "# c\nkeep\n")
        text, status = self._call(file_path=p, debug_mode=True)
        self.assertEqual(text, "keep")
        self.assertIn("Loaded", status)


class TestIsChanged(unittest.TestCase):
    """IS_CHANGED: cache invalidation on file state changes."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.f = self.root / "a.txt"
        self.f.write_text("original", encoding="utf-8")
        self.path = str(self.f)

    def tearDown(self):
        self.tmp.cleanup()

    def test_missing_file_returns_distinct_key(self):
        key = Node.IS_CHANGED(file_path=str(self.root / "missing.txt"))
        self.assertIsInstance(key, str)
        self.assertIn("<missing>", key)

    def test_existing_file_returns_string_with_state(self):
        key = Node.IS_CHANGED(file_path=self.path)
        self.assertIsInstance(key, str)
        self.assertIn(self.path, key)
        self.assertNotIn("<missing>", key)

    def test_key_changes_when_file_content_changes(self):
        before = Node.IS_CHANGED(file_path=self.path)
        self.f.write_text("modified", encoding="utf-8")
        _bump_mtime(self.f)
        after = Node.IS_CHANGED(file_path=self.path)
        self.assertNotEqual(before, after)

    def test_key_stable_when_file_unchanged(self):
        k1 = Node.IS_CHANGED(file_path=self.path)
        k2 = Node.IS_CHANGED(file_path=self.path)
        self.assertEqual(k1, k2)

    def test_key_reflects_path_change(self):
        other = self.root / "b.txt"
        other.write_text("original", encoding="utf-8")
        self.assertNotEqual(
            Node.IS_CHANGED(file_path=self.path),
            Node.IS_CHANGED(file_path=str(other)),
        )


class TestNodeMappings(unittest.TestCase):
    """Sanity checks on the exported mappings and class attributes."""

    def test_class_mappings_present(self):
        self.assertIn("illumoraeLoadTextFileGracefulNode", NODE_CLASS_MAPPINGS)
        self.assertIs(
            NODE_CLASS_MAPPINGS["illumoraeLoadTextFileGracefulNode"],
            Node,
        )

    def test_display_name_mappings_present(self):
        self.assertIn("illumoraeLoadTextFileGracefulNode", NODE_DISPLAY_NAME_MAPPINGS)
        self.assertEqual(
            NODE_DISPLAY_NAME_MAPPINGS["illumoraeLoadTextFileGracefulNode"],
            "Load Text File Graceful",
        )

    def test_class_attributes(self):
        self.assertEqual(Node.CATEGORY, "illumorae")
        self.assertEqual(Node.FUNCTION, "load_file")
        self.assertEqual(Node.OUTPUT_NODE, False)
        # Regression for the arity bug: counts must match.
        self.assertEqual(len(Node.RETURN_TYPES), len(Node.RETURN_NAMES))
        self.assertEqual(Node.RETURN_TYPES, ("STRING", "STRING"))
        self.assertEqual(Node.RETURN_NAMES, ("text", "status"))

    def test_return_types_are_core_comfyui_types(self):
        # Regression: old code used non-core "LABEL" and "DICT".
        for t in Node.RETURN_TYPES:
            self.assertEqual(t, "STRING")

    def test_input_types_shape(self):
        it = Node.INPUT_TYPES()
        self.assertIn("required", it)
        self.assertIn("optional", it)
        self.assertIn("file_path", it["required"])
        self.assertIn("debug_mode", it["optional"])
        # Regression: old code carried an empty "hidden" dict.
        self.assertNotIn("hidden", it)

    def test_is_changed_is_callable_classmethod(self):
        # Must be callable without an instance (ComfyUI calls it on the class).
        key = Node.IS_CHANGED(file_path="")
        self.assertIsInstance(key, str)


def _bump_mtime(path, delay=0.02):
    """Sleep briefly then touch ``path`` so its mtime advances.

    Some filesystems (notably Windows NTFS in some configurations) have
    coarse mtime resolution; the sleep makes the change observable.
    """
    time.sleep(delay)
    os.utime(path, None)


if __name__ == "__main__":
    unittest.main()
