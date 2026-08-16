"""Integration tests for load_checkpoint / load_checkpoint_safe.

Covers:
- C1: the file validated in safe mode is the same file passed to the loader
  (the validated path must equal the loaded path).
- M3: safe mode uses header-only validation (no safetensors dependency).
- m1: output_vae / output_clip are no longer accepted parameters.
- m2: file_extensions is intersected with the safetensors-only set in safe
  mode, so non-safe extensions are dropped while the input remains
  observable.
- m11: empty ckpt_name raises ValueError early.

These tests stub folder_paths.get_folder_paths / get_full_path and the
nodes.CheckpointLoaderSimple loader so the full call path can be exercised
without a running ComfyUI instance.

Run from the repo root:
    python -m pytest tests/test_safe_mode.py
    python -m tests.test_safe_mode
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import types
import unittest

# conftest.py installs the stubs and puts the package dir on sys.path.
import folder_paths  # noqa: E402
import nodes  # noqa: E402
from checkpoint_loader_by_string_dirty import illumoraeCheckpointLoaderByStringDirtyNode as Node  # noqa: E402


def _write_safetensors(path, header_obj=None):
    """Write a minimal valid safetensors file (header only)."""
    header_obj = header_obj or {"__metadata__": {}}
    header_bytes = json.dumps(header_obj).encode("utf-8")
    with open(path, "wb") as fh:
        fh.write(len(header_bytes).to_bytes(8, "little"))
        fh.write(header_bytes)


class TestSafeModePathResolution(unittest.TestCase):
    """C1: the validated path must be the same path the loader receives."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = self._tmp.name
        # Create a valid safetensors file in a registered checkpoint dir.
        self.ckpt_dir = os.path.join(self.tmp, "checkpoints")
        os.makedirs(self.ckpt_dir, exist_ok=True)
        self.ckpt_name = "modelA.safetensors"
        self.ckpt_full_path = os.path.join(self.ckpt_dir, self.ckpt_name)
        _write_safetensors(self.ckpt_full_path)

        # Patch folder_paths to point at our temp dir.
        self._orig_get_folder_paths = folder_paths.get_folder_paths
        self._orig_get_full_path = folder_paths.get_full_path
        folder_paths.get_folder_paths = lambda key: [self.ckpt_dir] if key == "checkpoints" else []
        folder_paths.get_full_path = lambda key, name: (
            os.path.join(self.ckpt_dir, name) if key == "checkpoints" else None
        )

        # Reset the loader's recorded names.
        nodes.CheckpointLoaderSimple.loaded_names = []

    def tearDown(self):
        folder_paths.get_folder_paths = self._orig_get_folder_paths
        folder_paths.get_full_path = self._orig_get_full_path
        self._tmp.cleanup()

    def test_safe_mode_loads_validated_path(self):
        """The rel_path passed to the loader must resolve (via
        get_full_path) to the same file that was header-validated."""
        node = Node()
        model, clip, vae, fname = node.load_checkpoint(
            self.ckpt_name, DEBUG_MODE=False, safe_mode=True
        )
        # The loader was called exactly once with the resolved rel_path.
        self.assertEqual(nodes.CheckpointLoaderSimple.loaded_names, [self.ckpt_name])
        self.assertEqual(fname, self.ckpt_name)
        self.assertEqual(model, "MODEL_SENTINEL")

    def test_safe_mode_rejects_non_safetensors_file(self):
        """A pickle-based file in the checkpoints dir must be rejected by
        the header validation, even if it has a .safetensors extension."""
        bad_path = os.path.join(self.ckpt_dir, "evil.safetensors")
        with open(bad_path, "wb") as fh:
            fh.write(b"\x80\x02cos\nsystem\n.")
        node = Node()
        with self.assertRaises(ValueError):
            node.load_checkpoint("evil.safetensors", safe_mode=True)
        # The loader must NOT have been called for the rejected file.
        self.assertEqual(nodes.CheckpointLoaderSimple.loaded_names, [])

    def test_safe_mode_no_safetensors_dependency(self):
        """M3: safe mode must not import the safetensors package. The
        header-only check uses only json + open."""
        # Ensure safetensors is not in sys.modules (it wouldn't be in the
        # test env anyway, but be explicit).
        sys.modules.pop("safetensors", None)
        sys.modules.pop("safetensors.torch", None)
        node = Node()
        # Should succeed without importing safetensors.
        node.load_checkpoint(self.ckpt_name, safe_mode=True)
        self.assertNotIn("safetensors", sys.modules)


class TestSafeModeExtensions(unittest.TestCase):
    """m2: file_extensions is intersected with the safetensors-only set."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = self._tmp.name
        self.ckpt_dir = os.path.join(self.tmp, "checkpoints")
        os.makedirs(self.ckpt_dir, exist_ok=True)
        # Create both a .safetensors and a .ckpt file with the same base name.
        _write_safetensors(os.path.join(self.ckpt_dir, "modelA.safetensors"))
        with open(os.path.join(self.ckpt_dir, "modelA.ckpt"), "wb") as fh:
            fh.write(b"\x80\x02cos\nsystem\n.")

        self._orig_get_folder_paths = folder_paths.get_folder_paths
        self._orig_get_full_path = folder_paths.get_full_path
        folder_paths.get_folder_paths = lambda key: [self.ckpt_dir] if key == "checkpoints" else []
        folder_paths.get_full_path = lambda key, name: (
            os.path.join(self.ckpt_dir, name) if key == "checkpoints" else None
        )
        nodes.CheckpointLoaderSimple.loaded_names = []

    def tearDown(self):
        folder_paths.get_folder_paths = self._orig_get_folder_paths
        folder_paths.get_full_path = self._orig_get_full_path
        self._tmp.cleanup()

    def test_unsafe_extension_dropped_in_safe_mode(self):
        """Even if file_extensions includes .ckpt, safe mode must only
        discover .safetensors files and must not load the .ckpt file."""
        node = Node()
        model, clip, vae, fname = node.load_checkpoint(
            "modelA", safe_mode=True, file_extensions=".safetensors,.ckpt"
        )
        # The .safetensors file was loaded, not the .ckpt.
        self.assertTrue(fname.endswith(".safetensors"))
        self.assertEqual(nodes.CheckpointLoaderSimple.loaded_names, ["modelA.safetensors"])

    def test_safe_extension_subset_respected(self):
        """If file_extensions is only .sft, safe mode discovers only .sft
        files (the .safetensors file is excluded)."""
        # Add a .sft file.
        _write_safetensors(os.path.join(self.ckpt_dir, "modelB.sft"))
        node = Node()
        model, clip, vae, fname = node.load_checkpoint(
            "modelB", safe_mode=True, file_extensions=".sft"
        )
        self.assertEqual(fname, "modelB.sft")

    def test_all_unsafe_extensions_falls_back_to_safe_set(self):
        """If every requested extension is unsafe, safe mode falls back to
        the full safetensors set rather than discovering nothing."""
        node = Node()
        model, clip, vae, fname = node.load_checkpoint(
            "modelA", safe_mode=True, file_extensions=".ckpt,.bin"
        )
        # Falls back to .safetensors/.sft, finds modelA.safetensors.
        self.assertTrue(fname.endswith(".safetensors"))


class TestParameterCleanup(unittest.TestCase):
    """m1: output_vae / output_clip are no longer accepted parameters."""

    def test_load_checkpoint_rejects_output_vae(self):
        node = Node()
        with self.assertRaises(TypeError):
            node.load_checkpoint("modelA", output_vae=True)

    def test_load_checkpoint_rejects_output_clip(self):
        node = Node()
        with self.assertRaises(TypeError):
            node.load_checkpoint("modelA", output_clip=True)

    def test_load_checkpoint_safe_rejects_output_vae(self):
        node = Node()
        with self.assertRaises(TypeError):
            node.load_checkpoint_safe("modelA", output_vae=True)


class TestEmptyInput(unittest.TestCase):
    """m11: empty ckpt_name raises ValueError before any filesystem access."""

    def test_empty_string_raises_value_error(self):
        node = Node()
        with self.assertRaises(ValueError) as ctx:
            node.load_checkpoint("", safe_mode=True)
        self.assertIn("empty", str(ctx.exception).lower())

    def test_whitespace_string_raises_value_error(self):
        node = Node()
        with self.assertRaises(ValueError) as ctx:
            node.load_checkpoint("   ", safe_mode=False)
        self.assertIn("empty", str(ctx.exception).lower())


if __name__ == "__main__":
    unittest.main(verbosity=2)
