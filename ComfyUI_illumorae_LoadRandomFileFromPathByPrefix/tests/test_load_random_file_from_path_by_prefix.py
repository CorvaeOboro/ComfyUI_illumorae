"""Regression tests for illumoraeLoadRandomFileFromPathByPrefixNode.

Covers the bugs fixed in the 20260814 review pass:

- _parse_extensions: None / "" / "*" all disable the filter (aligned);
  comma lists, missing leading dot, surrounding whitespace, mixed case.
- _collect_matches: prefix filter, extension filter, case_sensitive
  applies to prefix only (extension always case-insensitive), recursive
  walk, redundant stem branch removed without behavior change, missing
  folder returns empty.
- load_random_file: empty/missing folder, no matches, deterministic
  index_override, reproducible seed, override out-of-range surfaces in
  the status report, non-UTF-8 fallback, OSError path.
- RNG isolation: the global random state is not perturbed by seeding.
- _folder_signature / IS_CHANGED: seed<0 returns nan; seeded key includes
  a folder signature that changes when the folder contents change.

Usage:
    python -m tests.test_load_random_file_from_path_by_prefix
    python -m pytest tests/test_load_random_file_from_path_by_prefix.py -v
"""
from __future__ import annotations

import math
import os
import random
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

from load_random_file_from_path_by_prefix import (  # noqa: E402
    NODE_CLASS_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS,
    illumoraeLoadRandomFileFromPathByPrefixNode,
)

Node = illumoraeLoadRandomFileFromPathByPrefixNode


class TestParseExtensions(unittest.TestCase):
    """_parse_extensions: normalization and the disable-filter sentinel."""

    def test_none_disables_filter(self):
        self.assertIsNone(Node._parse_extensions(None))

    def test_empty_string_disables_filter(self):
        # Aligned with None: both mean "any extension".
        self.assertIsNone(Node._parse_extensions(""))

    def test_whitespace_only_disables_filter(self):
        self.assertIsNone(Node._parse_extensions("   "))

    def test_star_disables_filter(self):
        self.assertIsNone(Node._parse_extensions("*"))

    def test_star_among_others_disables_filter(self):
        # A single "*" anywhere in the list disables the filter.
        self.assertIsNone(Node._parse_extensions(".md,*"))

    def test_single_extension_with_dot(self):
        self.assertEqual(Node._parse_extensions(".md"), [".md"])

    def test_single_extension_without_dot_gets_dot_prefix(self):
        self.assertEqual(Node._parse_extensions("md"), [".md"])

    def test_comma_separated_list(self):
        self.assertEqual(Node._parse_extensions(".md,.txt"), [".md", ".txt"])

    def test_comma_separated_without_dots(self):
        self.assertEqual(Node._parse_extensions("md,txt"), [".md", ".txt"])

    def test_surrounding_whitespace_and_blanks_ignored(self):
        self.assertEqual(
            Node._parse_extensions("  .md , , .txt  "),
            [".md", ".txt"],
        )

    def test_uppercase_extensions_lowercased(self):
        self.assertEqual(Node._parse_extensions(".MD,.TXT"), [".md", ".txt"])


class TestCollectMatches(unittest.TestCase):
    """_collect_matches: filtering, case sensitivity, recursion."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        # Top-level files.
        (self.root / "prompt_wan22_english.md").write_text("english", encoding="utf-8")
        (self.root / "prompt_wan22_japanese.md").write_text("japanese", encoding="utf-8")
        (self.root / "prompt_wan22_notes.txt").write_text("notes", encoding="utf-8")
        (self.root / "prompt_sdxl_english.md").write_text("sdxl", encoding="utf-8")
        (self.root / "readme.md").write_text("readme", encoding="utf-8")
        # Nested file for the recursive test.
        sub = self.root / "sub"
        sub.mkdir()
        (sub / "prompt_wan22_nested.md").write_text("nested", encoding="utf-8")
        self.node = Node()

    def tearDown(self):
        self.tmp.cleanup()

    def _collect(self, **kw):
        defaults = dict(
            prefix="prompt_wan22_",
            extensions=[".md"],
            recursive=False,
            case_sensitive=False,
            debug_mode=False,
        )
        defaults.update(kw)
        return self.node._collect_matches(self.root, **defaults)

    def test_prefix_and_extension_filter(self):
        matches, skipped = self._collect()
        names = sorted(p.name for p in matches)
        self.assertEqual(names, ["prompt_wan22_english.md", "prompt_wan22_japanese.md"])
        skipped_names = sorted(p.name for p, _ in skipped)
        self.assertIn("prompt_wan22_notes.txt", skipped_names)
        self.assertIn("prompt_sdxl_english.md", skipped_names)

    def test_extension_star_disables_filter(self):
        matches, _ = self._collect(extensions=None)
        names = sorted(p.name for p in matches)
        self.assertEqual(
            names,
            ["prompt_wan22_english.md", "prompt_wan22_japanese.md", "prompt_wan22_notes.txt"],
        )

    def test_multiple_extensions(self):
        matches, _ = self._collect(extensions=[".md", ".txt"])
        names = sorted(p.name for p in matches)
        self.assertEqual(
            names,
            ["prompt_wan22_english.md", "prompt_wan22_japanese.md", "prompt_wan22_notes.txt"],
        )

    def test_recursive_walk(self):
        matches, _ = self._collect(recursive=True)
        names = sorted(p.name for p in matches)
        self.assertIn("prompt_wan22_nested.md", names)
        self.assertEqual(
            names,
            ["prompt_wan22_english.md", "prompt_wan22_japanese.md", "prompt_wan22_nested.md"],
        )

    def test_non_recursive_excludes_subdirs(self):
        matches, _ = self._collect(recursive=False)
        names = sorted(p.name for p in matches)
        self.assertNotIn("prompt_wan22_nested.md", names)

    def test_case_sensitive_prefix(self):
        # With case_sensitive=True, a lowercase prefix must not match an
        # uppercase filename.
        (self.root / "PROMPT_WAN22_caps.md").write_text("caps", encoding="utf-8")
        matches_ci, _ = self._collect(case_sensitive=False)
        matches_cs, _ = self._collect(case_sensitive=True)
        names_ci = sorted(p.name for p in matches_ci)
        names_cs = sorted(p.name for p in matches_cs)
        self.assertIn("PROMPT_WAN22_caps.md", names_ci)
        self.assertNotIn("PROMPT_WAN22_caps.md", names_cs)

    def test_extension_always_case_insensitive_even_with_case_sensitive(self):
        # Per the documented behavior: case_sensitive governs only the prefix.
        (self.root / "prompt_wan22_caps.MD").write_text("capsmd", encoding="utf-8")
        matches, _ = self._collect(case_sensitive=True, extensions=[".md"])
        names = sorted(p.name for p in matches)
        self.assertIn("prompt_wan22_caps.MD", names)

    def test_empty_prefix_matches_all_files(self):
        matches, _ = self._collect(prefix="", extensions=[".md"])
        names = sorted(p.name for p in matches)
        self.assertEqual(
            names,
            ["prompt_sdxl_english.md", "prompt_wan22_english.md",
             "prompt_wan22_japanese.md", "readme.md"],
        )

    def test_missing_folder_returns_empty(self):
        node = Node()
        matches, skipped = node._collect_matches(
            Path(self.tmp.name) / "does_not_exist",
            prefix="prompt_wan22_",
            extensions=[".md"],
            recursive=False,
            case_sensitive=False,
            debug_mode=False,
        )
        self.assertEqual(matches, [])
        self.assertEqual(skipped, [])

    def test_matches_are_sorted_deterministically(self):
        matches, _ = self._collect()
        names = [p.name for p in matches]
        self.assertEqual(names, sorted(names, key=str.lower))


class TestLoadRandomFile(unittest.TestCase):
    """load_random_file: end-to-end behavior and edge cases."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        for i in range(3):
            (self.root / f"prompt_wan22_v{i + 1}.md").write_text(
                f"content-{i + 1}", encoding="utf-8",
            )
        (self.root / "prompt_sdxl_v1.md").write_text("sdxl", encoding="utf-8")
        self.node = Node()

    def tearDown(self):
        self.tmp.cleanup()

    def _call(self, **kw):
        defaults = dict(
            folder=str(self.root),
            prefix="prompt_wan22_",
            extension=".md",
            recursive=False,
            case_sensitive=False,
            seed=-1,
            index_override=-1,
            debug_mode=False,
        )
        defaults.update(kw)
        return self.node.load_random_file(**defaults)

    def test_empty_folder_input(self):
        self.assertEqual(self._call(folder=""), ("", "", "", "", "Folder path is empty"))

    def test_none_folder_input(self):
        self.assertEqual(self._call(folder=None), ("", "", "", "", "Folder path is empty"))

    def test_missing_folder(self):
        out = self._call(folder=str(self.root / "missing"))
        text, name, path, folder, status = out
        self.assertEqual(text, "")
        self.assertEqual(name, "")
        self.assertEqual(path, "")
        self.assertEqual(folder, "")
        self.assertIn("Folder not found", status)

    def test_no_matches_returns_report(self):
        out = self._call(prefix="nomatch_")
        text, name, path, folder, status = out
        self.assertEqual(text, "")
        self.assertEqual(name, "")
        self.assertEqual(path, "")
        self.assertEqual(folder, str(self.root))
        self.assertIn("No files match", status)

    def test_index_override_is_deterministic(self):
        out1 = self._call(index_override=1, seed=42)
        out2 = self._call(index_override=1, seed=99)
        self.assertEqual(out1, out2)
        self.assertEqual(out1[1], "prompt_wan22_v1")
        self.assertEqual(out1[0], "content-1")

    def test_index_override_second(self):
        out = self._call(index_override=2, seed=-1)
        self.assertEqual(out[1], "prompt_wan22_v2")
        self.assertEqual(out[0], "content-2")

    def test_index_override_out_of_range_falls_back_and_surfaces(self):
        out = self._call(index_override=999, seed=42)
        text, name, path, folder, status = out
        self.assertNotEqual(text, "")
        self.assertIn("fell back to random", status)
        self.assertIn("index_override 999", status)

    def test_seed_is_reproducible(self):
        out1 = self._call(seed=12345)
        out2 = self._call(seed=12345)
        self.assertEqual(out1, out2)

    def test_different_seeds_pick_valid_files(self):
        out1 = self._call(seed=1)
        out2 = self._call(seed=2)
        valid = {"prompt_wan22_v1", "prompt_wan22_v2", "prompt_wan22_v3"}
        self.assertIn(out1[1], valid)
        self.assertIn(out2[1], valid)

    def test_returned_paths_are_absolute(self):
        out = self._call(index_override=1)
        _, _, path, folder, _ = out
        self.assertTrue(Path(path).is_absolute())
        self.assertTrue(Path(folder).is_absolute())
        self.assertEqual(Path(path).parent, Path(folder))

    def test_non_utf8_file_falls_back_to_replace(self):
        bad = self.root / "prompt_wan22_bad.md"
        bad.write_bytes(b"\xff\xfe binary \x00 bytes")
        out = self._call(prefix="prompt_wan22_bad", index_override=1)
        text, name, path, folder, status = out
        self.assertEqual(name, "prompt_wan22_bad")
        self.assertIsInstance(text, str)
        self.assertIn("Loaded", status)

    def test_oserror_during_read_returns_status(self):
        # Force open() to raise OSError (e.g. permission denied) for the
        # chosen file, exercising the narrowed `except OSError` branch.
        real_open = open

        def fake_open(path, *args, **kwargs):
            if str(path).endswith("prompt_wan22_v1.md"):
                raise PermissionError("simulated permission denied")
            return real_open(path, *args, **kwargs)

        with mock.patch("builtins.open", side_effect=fake_open):
            out = self._call(index_override=1)
        text, name, path, folder, status = out
        self.assertEqual(text, "")
        self.assertEqual(name, "prompt_wan22_v1")
        self.assertIn("Error reading", status)
        self.assertIn("simulated permission denied", status)


class TestRngIsolation(unittest.TestCase):
    """Seeding must not perturb the process-global random state."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        for i in range(3):
            (self.root / f"prompt_wan22_v{i + 1}.md").write_text(
                f"c{i + 1}", encoding="utf-8",
            )
        self.node = Node()

    def tearDown(self):
        self.tmp.cleanup()

    def _call(self, seed):
        return self.node.load_random_file(
            folder=str(self.root),
            prefix="prompt_wan22_",
            extension=".md",
            recursive=False,
            case_sensitive=False,
            seed=seed,
            index_override=-1,
            debug_mode=False,
        )

    def test_global_rng_state_unchanged_by_seeded_call(self):
        random.seed(0)
        before = random.getstate()
        self._call(seed=12345)
        self.assertEqual(random.getstate(), before)

    def test_global_rng_state_unchanged_by_unseeded_call(self):
        random.seed(0)
        before = random.getstate()
        self._call(seed=-1)
        # Unseeded uses a fresh Random() instance, so the global state is
        # still untouched.
        self.assertEqual(random.getstate(), before)


class TestFolderSignatureAndIsChanged(unittest.TestCase):
    """_folder_signature + IS_CHANGED: cache invalidation on disk changes."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        (self.root / "a.md").write_text("a", encoding="utf-8")

    def tearDown(self):
        self.tmp.cleanup()

    def test_missing_folder_signature(self):
        self.assertEqual(Node._folder_signature(str(self.root / "missing"), False), "<missing>")

    def test_empty_folder_signature(self):
        self.assertEqual(Node._folder_signature("", False), "<missing>")

    def test_existing_folder_signature_is_string(self):
        sig = Node._folder_signature(str(self.root), False)
        self.assertIsInstance(sig, str)
        self.assertNotIn(sig, ("<missing>", "<unreadable>"))
        self.assertIn(":", sig)

    def test_signature_changes_when_file_added(self):
        sig_before = Node._folder_signature(str(self.root), False)
        (self.root / "b.md").write_text("b", encoding="utf-8")
        # Force an mtime change (some filesystems have coarse resolution).
        _bump_mtime(self.root)
        sig_after = Node._folder_signature(str(self.root), False)
        self.assertNotEqual(sig_before, sig_after)

    def test_is_changed_seed_negative_returns_nan(self):
        for s in (-1, -2, None):
            self.assertTrue(
                _is_nan(Node.IS_CHANGED(folder=str(self.root), seed=s)),
                f"seed={s!r} should return nan",
            )

    def test_is_changed_seeded_returns_string_with_signature(self):
        key = Node.IS_CHANGED(folder=str(self.root), seed=42)
        self.assertIsInstance(key, str)
        sig = Node._folder_signature(str(self.root), False)
        self.assertIn(sig, key)

    def test_is_changed_changes_when_folder_changes(self):
        key_before = Node.IS_CHANGED(folder=str(self.root), seed=42)
        (self.root / "c.md").write_text("c", encoding="utf-8")
        _bump_mtime(self.root)
        key_after = Node.IS_CHANGED(folder=str(self.root), seed=42)
        self.assertNotEqual(key_before, key_after)

    def test_is_changed_stable_when_folder_unchanged(self):
        k1 = Node.IS_CHANGED(folder=str(self.root), seed=42, index_override=2)
        k2 = Node.IS_CHANGED(folder=str(self.root), seed=42, index_override=2)
        self.assertEqual(k1, k2)

    def test_is_changed_reflects_input_changes(self):
        base = dict(folder=str(self.root), seed=42)
        self.assertNotEqual(
            Node.IS_CHANGED(prefix="a", **base),
            Node.IS_CHANGED(prefix="b", **base),
        )
        self.assertNotEqual(
            Node.IS_CHANGED(extension=".md", **base),
            Node.IS_CHANGED(extension=".txt", **base),
        )
        self.assertNotEqual(
            Node.IS_CHANGED(index_override=1, **base),
            Node.IS_CHANGED(index_override=2, **base),
        )


class TestNodeMappings(unittest.TestCase):
    """Sanity checks on the exported mappings and class attributes."""

    def test_class_mappings_present(self):
        self.assertIn("illumoraeLoadRandomFileFromPathByPrefixNode", NODE_CLASS_MAPPINGS)
        self.assertIs(
            NODE_CLASS_MAPPINGS["illumoraeLoadRandomFileFromPathByPrefixNode"],
            Node,
        )

    def test_display_name_mappings_present(self):
        self.assertIn("illumoraeLoadRandomFileFromPathByPrefixNode", NODE_DISPLAY_NAME_MAPPINGS)
        self.assertEqual(
            NODE_DISPLAY_NAME_MAPPINGS["illumoraeLoadRandomFileFromPathByPrefixNode"],
            "Load Random File From Path By Prefix",
        )

    def test_class_attributes(self):
        self.assertEqual(Node.CATEGORY, "illumorae")
        self.assertEqual(Node.FUNCTION, "load_random_file")
        self.assertEqual(Node.OUTPUT_NODE, False)
        self.assertEqual(len(Node.RETURN_TYPES), len(Node.RETURN_NAMES))
        self.assertEqual(
            Node.RETURN_NAMES, ("text", "file_name", "file_path", "folder_path", "status"),
        )

    def test_input_types_shape(self):
        it = Node.INPUT_TYPES()
        self.assertIn("required", it)
        self.assertIn("optional", it)
        for name in ("folder", "prefix", "extension", "recursive",
                     "case_sensitive", "seed", "index_override"):
            self.assertIn(name, it["required"])
        self.assertIn("debug_mode", it["optional"])


def _is_nan(value):
    return isinstance(value, float) and math.isnan(value)


def _bump_mtime(path, delay=0.02):
    """Sleep briefly then touch ``path`` so its mtime advances.

    Some filesystems (notably Windows NTFS in some configurations) have
    coarse mtime resolution; the sleep makes the change observable.
    """
    time.sleep(delay)
    os.utime(path, None)


if __name__ == "__main__":
    unittest.main()
