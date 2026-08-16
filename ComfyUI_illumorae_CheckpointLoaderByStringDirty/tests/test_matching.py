"""Regression tests for illumoraeCheckpointLoaderByStringDirtyNode.find_matching_filename.

Covers the matching strategies (exact path, exact filename, base name,
partial filename, partial path), the fixed ambiguity detection (M1), the
tightened partial-filename strategy (M2), the empty-input guard (m11),
extension preference ranking, and case-insensitivity.

Run from the repo root:
    python -m pytest tests/test_matching.py
    python -m tests.test_matching
"""
from __future__ import annotations

import os
import sys
import unittest

# conftest.py (same directory) installs the folder_paths/nodes stubs and puts
# the package directory on sys.path before collection.

from checkpoint_loader_by_string_dirty import illumoraeCheckpointLoaderByStringDirtyNode as Node  # noqa: E402


class TestFindMatchingFilename(unittest.TestCase):
    """Black-box tests over the staticmethod find_matching_filename."""

    def _files(self, *paths, base_dir="/fake/ckpts"):
        # Each entry is (rel_path, base_dir).
        return [(p, base_dir) for p in paths]

    # --- exact strategies ---

    def test_exact_relative_path_match(self):
        files = self._files("sub/modelA.safetensors", "sub/modelB.safetensors")
        rel, base = Node.find_matching_filename("sub/modelA.safetensors", files)
        self.assertEqual(rel, "sub/modelA.safetensors")

    def test_exact_relative_path_case_insensitive(self):
        files = self._files("sub/ModelA.safetensors")
        rel, _ = Node.find_matching_filename("SUB/modela.safetensors", files)
        self.assertEqual(rel, "sub/ModelA.safetensors")

    def test_exact_relative_path_backslash_normalized(self):
        files = self._files("sub/ModelA.safetensors")
        rel, _ = Node.find_matching_filename("sub\\ModelA.safetensors", files)
        self.assertEqual(rel, "sub/ModelA.safetensors")

    def test_exact_filename_match(self):
        files = self._files("dir1/modelA.safetensors", "dir2/modelB.safetensors")
        rel, _ = Node.find_matching_filename("modelB.safetensors", files)
        self.assertEqual(rel, "dir2/modelB.safetensors")

    def test_base_name_match(self):
        """Input without extension matches the file's base name."""
        files = self._files("dir1/modelA.safetensors", "dir2/modelB.safetensors")
        rel, _ = Node.find_matching_filename("modelA", files)
        self.assertEqual(rel, "dir1/modelA.safetensors")

    # --- extension preference ---

    def test_preferred_extension_wins_on_tie(self):
        """When the same base name exists as both .safetensors and .sft, the
        .safetensors file (first in preferred_exts) is selected."""
        files = self._files("modelA.sft", "modelA.safetensors")
        rel, _ = Node.find_matching_filename("modelA", files, preferred_exts=(".safetensors", ".sft"))
        self.assertTrue(rel.endswith(".safetensors"))

    def test_preferred_extension_order_respected(self):
        """If .sft is preferred over .safetensors, the .sft file wins."""
        files = self._files("modelA.sft", "modelA.safetensors")
        rel, _ = Node.find_matching_filename("modelA", files, preferred_exts=(".sft", ".safetensors"))
        self.assertTrue(rel.endswith(".sft"))

    # --- partial matching (M2) ---

    def test_partial_filename_matches_basename_only(self):
        """Strategy 4 must match the input basename against the file's
        *basename*, not the full path. 'modelA' is a substring of
        'modelA.safetensors' (basename) so it matches."""
        files = self._files("dir1/modelA.safetensors")
        rel, _ = Node.find_matching_filename("modelA.safet", files)
        self.assertEqual(rel, "dir1/modelA.safetensors")

    def test_partial_filename_does_not_match_unrelated_nested_path(self):
        """Strategy 4 (partial filename) matches against the basename only,
        so 'extra' (a directory name) does not match via strategy 4. However
        strategy 5 (partial path) still matches it against the full path,
        so the overall result is a match - confirming strategy 4 was
        skipped and strategy 5 handled it."""
        files = self._files("models/extra/alpha.safetensors", "other/beta.safetensors")
        rel, _ = Node.find_matching_filename("extra", files)
        self.assertEqual(rel, "models/extra/alpha.safetensors")

    def test_partial_path_strategy_still_matches_directory_fragment(self):
        """Strategy 5 (partial path) still allows matching against the full
        path, so 'extra' matches 'models/extra/alpha.safetensors'."""
        files = self._files("models/extra/alpha.safetensors", "other/beta.safetensors")
        rel, _ = Node.find_matching_filename("extra", files)
        self.assertEqual(rel, "models/extra/alpha.safetensors")

    # --- ambiguity detection (M1) ---

    def test_ambiguous_base_name_raises(self):
        """Two files with the same base name in different directories have
        equal rank (same ext, same length) so the matcher must raise
        ValueError instead of silently picking one."""
        files = self._files("dir1/modelA.safetensors", "dir2/modelA.safetensors")
        with self.assertRaises(ValueError) as ctx:
            Node.find_matching_filename("modelA", files)
        self.assertIn("Ambiguous", str(ctx.exception))

    def test_ambiguous_base_name_resolved_by_extension_preference(self):
        """If the two candidates differ in extension preference, the tie is
        broken by ext_rank and no ambiguity error is raised."""
        files = self._files("dir1/modelA.sft", "dir2/modelA.safetensors")
        rel, _ = Node.find_matching_filename("modelA", files, preferred_exts=(".safetensors", ".sft"))
        self.assertTrue(rel.endswith(".safetensors"))

    def test_different_length_paths_not_ambiguous(self):
        """Two files with different path lengths are ranked by length and
        do not raise even if they share the same extension."""
        files = self._files("a/modelA.safetensors", "a/b/c/modelA.safetensors")
        # Both match base name 'modelA'; shorter path wins, no ambiguity.
        rel, _ = Node.find_matching_filename("modelA", files)
        self.assertEqual(rel, "a/modelA.safetensors")

    # --- empty input (m11) ---

    def test_empty_input_raises_value_error(self):
        files = self._files("modelA.safetensors")
        with self.assertRaises(ValueError) as ctx:
            Node.find_matching_filename("", files)
        self.assertIn("empty", str(ctx.exception).lower())

    def test_whitespace_only_input_raises_value_error(self):
        files = self._files("modelA.safetensors")
        with self.assertRaises(ValueError) as ctx:
            Node.find_matching_filename("   ", files)
        self.assertIn("empty", str(ctx.exception).lower())

    # --- no match ---

    def test_no_match_raises_file_not_found_error(self):
        files = self._files("modelA.safetensors")
        with self.assertRaises(FileNotFoundError):
            Node.find_matching_filename("does_not_exist", files)

    def test_no_files_raises_file_not_found_error(self):
        with self.assertRaises(FileNotFoundError):
            Node.find_matching_filename("modelA", [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
