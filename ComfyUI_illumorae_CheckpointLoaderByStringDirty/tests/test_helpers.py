"""Regression tests for the helper staticmethods.

Covers:
- _parse_extensions (m2): normalization, leading-dot insertion, lowercasing,
  empty-entry dropping.
- _get_all_checkpoints_recursive_all_dirs (m7): case-insensitive extension
  filtering, recursive discovery, forward-slash normalization.
- _validate_safetensors_header (M3): header-only validation accepts a
  well-formed safetensors file and rejects corrupt / non-safetensors files
  without loading tensor data.

Run from the repo root:
    python -m pytest tests/test_helpers.py
    python -m tests.test_helpers
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest

from checkpoint_loader_by_string_dirty import illumoraeCheckpointLoaderByStringDirtyNode as Node  # noqa: E402


class TestParseExtensions(unittest.TestCase):
    def test_default_extensions(self):
        exts = Node._parse_extensions(".safetensors,.sft")
        self.assertEqual(exts, (".safetensors", ".sft"))

    def test_adds_leading_dot(self):
        exts = Node._parse_extensions("safetensors,sft")
        self.assertEqual(exts, (".safetensors", ".sft"))

    def test_lowercases(self):
        exts = Node._parse_extensions(".SafeTensors,.SFT")
        self.assertEqual(exts, (".safetensors", ".sft"))

    def test_drops_empty_entries(self):
        exts = Node._parse_extensions(".safetensors,, ,.sft")
        self.assertEqual(exts, (".safetensors", ".sft"))

    def test_strips_whitespace(self):
        exts = Node._parse_extensions("  .safetensors  ,  .sft  ")
        self.assertEqual(exts, (".safetensors", ".sft"))

    def test_single_extension(self):
        exts = Node._parse_extensions(".ckpt")
        self.assertEqual(exts, (".ckpt",))


class TestRecursiveWalk(unittest.TestCase):
    def _make_file(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as fh:
            fh.write(b"\x00")

    def test_finds_files_recursively(self, *args):
        with tempfile.TemporaryDirectory() as tmp:
            self._make_file(os.path.join(tmp, "top.safetensors"))
            self._make_file(os.path.join(tmp, "sub", "nested.safetensors"))
            self._make_file(os.path.join(tmp, "sub", "deep", "deep.sft"))
            files = Node._get_all_checkpoints_recursive_all_dirs([tmp], exts=(".safetensors", ".sft"))
            rels = sorted(f for f, _ in files)
            self.assertEqual(rels, [
                "sub/deep/deep.sft",
                "sub/nested.safetensors",
                "top.safetensors",
            ])

    def test_case_insensitive_extension_filter(self):
        """m7: uppercase extensions must be discovered."""
        with tempfile.TemporaryDirectory() as tmp:
            self._make_file(os.path.join(tmp, "Model.SAFETENSORS"))
            self._make_file(os.path.join(tmp, "other.Sft"))
            files = Node._get_all_checkpoints_recursive_all_dirs([tmp], exts=(".safetensors", ".sft"))
            rels = sorted(f for f, _ in files)
            self.assertEqual(rels, ["Model.SAFETENSORS", "other.Sft"])

    def test_ignores_non_matching_extensions(self):
        with tempfile.TemporaryDirectory() as tmp:
            self._make_file(os.path.join(tmp, "model.ckpt"))
            self._make_file(os.path.join(tmp, "model.safetensors"))
            files = Node._get_all_checkpoints_recursive_all_dirs([tmp], exts=(".safetensors", ".sft"))
            rels = [f for f, _ in files]
            self.assertEqual(rels, ["model.safetensors"])

    def test_forward_slash_normalization(self):
        with tempfile.TemporaryDirectory() as tmp:
            self._make_file(os.path.join(tmp, "sub", "model.safetensors"))
            files = Node._get_all_checkpoints_recursive_all_dirs([tmp], exts=(".safetensors",))
            rel = files[0][0]
            self.assertNotIn("\\", rel)
            self.assertIn("/", rel)


class TestValidateSafetensorsHeader(unittest.TestCase):
    def _write_safetensors_header(self, path, header_obj):
        """Write a minimal valid safetensors file (header only, no tensors)."""
        header_bytes = json.dumps(header_obj).encode("utf-8")
        with open(path, "wb") as fh:
            fh.write(len(header_bytes).to_bytes(8, "little"))
            fh.write(header_bytes)

    def test_valid_safetensors_passes(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "model.safetensors")
            self._write_safetensors_header(path, {"__metadata__": {}})
            # Should not raise.
            Node._validate_safetensors_header(path)

    def test_empty_header_obj_passes(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "model.safetensors")
            self._write_safetensors_header(path, {})
            Node._validate_safetensors_header(path)

    def test_pickle_file_rejected(self):
        """A non-safetensors file (e.g. a pickle) must be rejected."""
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "model.ckpt")
            with open(path, "wb") as fh:
                fh.write(b"\x80\x02cos\nsystem\nq\x00X\x05\x00\x00\x00echo\x01q\x01\x85q\x02Rq\x03.")
            with self.assertRaises(ValueError):
                Node._validate_safetensors_header(path)

    def test_truncated_file_rejected(self):
        """A file smaller than 8 bytes cannot contain a safetensors header."""
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "tiny.safetensors")
            with open(path, "wb") as fh:
                fh.write(b"\x00\x00")
            with self.assertRaises(ValueError):
                Node._validate_safetensors_header(path)

    def test_corrupt_header_rejected(self):
        """A file with a valid 8-byte length but non-JSON header is rejected."""
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "corrupt.safetensors")
            with open(path, "wb") as fh:
                fh.write((10).to_bytes(8, "little"))
                fh.write(b"not json!!")
            with self.assertRaises(ValueError):
                Node._validate_safetensors_header(path)

    def test_truncated_header_rejected(self):
        """A file whose declared header length exceeds the remaining bytes."""
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "trunc.safetensors")
            with open(path, "wb") as fh:
                fh.write((1000).to_bytes(8, "little"))
                fh.write(b"{}")
            with self.assertRaises(ValueError):
                Node._validate_safetensors_header(path)

    def test_nonexistent_file_rejected(self):
        with self.assertRaises(ValueError):
            Node._validate_safetensors_header("/nonexistent/path/model.safetensors")


if __name__ == "__main__":
    unittest.main(verbosity=2)
