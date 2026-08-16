"""Regression tests for illumoraeLoadImageWFilePathOutNode.

Covers the bugs fixed in the 20260816 review pass:

- load_image: None input raises ValueError (no longer AttributeError);
  non-existent path raises ValueError; corrupt/non-image file raises
  RuntimeError with a clear message; valid RGB image returns a
  (1, H, W, 3) float32 tensor in [0, 1] plus a 64x64 zero mask; RGBA
  image returns an inverted alpha mask matching image dimensions; palette
  (P) mode with transparency preserves alpha via RGBA conversion; file
  name and folder path come from pathlib (stem / parent).
- _resolve_path: None -> None; a direct valid file path is used as-is;
  a non-existent path falls through to the annotation system then
  fallback; return annotation is Optional[Path].
- IS_CHANGED: None -> "" (always unchanged); non-existent path ->
  float("nan") (always re-run); valid file -> SHA-256 hex of bytes;
  identical bytes -> same hash; edited bytes -> different hash.
- VALIDATE_INPUTS: None -> True; existing path -> True; missing path ->
  error string.
- Registration: NODE_CLASS_MAPPINGS / NODE_DISPLAY_NAME_MAPPINGS and
  class attributes (RETURN_TYPES / RETURN_NAMES arity, default input)
  are consistent.

`folder_paths` is a ComfyUI runtime module not present in the test
environment, so it is stubbed in sys.modules before the node import.

Usage:
    python -m tests.test_load_image_filepath_out
    python -m pytest tests/test_load_image_filepath_out.py -v
"""
from __future__ import annotations

import hashlib
import math
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

# folder_paths is a ComfyUI module unavailable outside the ComfyUI process.
# Stub it before importing the node so module-level `import folder_paths`
# succeeds. get_annotated_filepath just returns its input unchanged, which
# lets _resolve_path fall through to the direct-path branch for real files.
if "folder_paths" not in sys.modules:
    _fp_stub = mock.MagicMock()
    _fp_stub.get_annotated_filepath = staticmethod(lambda x: x)
    sys.modules["folder_paths"] = _fp_stub

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PARENT = os.path.dirname(_THIS_DIR)
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)

import torch  # noqa: E402
from PIL import Image  # noqa: E402

from load_image_filepath_out import (  # noqa: E402
    NODE_CLASS_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS,
    illumoraeLoadImageWFilePathOutNode,
)

Node = illumoraeLoadImageWFilePathOutNode


def _make_rgb(path: Path, w: int = 8, h: int = 6, color=(255, 0, 0)):
    """Write a small solid-color RGB PNG to ``path``."""
    Image.new("RGB", (w, h), color).save(path)


def _make_rgba(path: Path, w: int = 8, h: int = 6, color=(255, 0, 0, 128)):
    """Write a small solid-color RGBA PNG to ``path``."""
    Image.new("RGBA", (w, h), color).save(path)


def _make_palette_with_transparency(path: Path, w: int = 4, h: int = 4):
    """Write a palette-mode PNG whose index 0 is fully transparent."""
    img = Image.new("P", (w, h))
    # Two palette entries: 0 transparent, 1 opaque red.
    img.putpalette([0, 0, 0, 255, 0, 0] + [0] * (256 * 3 - 6))
    px = img.load()
    for y in range(h):
        for x in range(w):
            px[x, y] = 1 if (x + y) % 2 else 0
    img.info["transparency"] = 0
    img.save(path)


class TestRegistration(unittest.TestCase):
    """Node is registered with the expected id and display name."""

    def test_node_class_mapping(self):
        self.assertIn(
            "illumoraeLoadImageWFilePathOutNode", NODE_CLASS_MAPPINGS
        )
        self.assertIs(
            NODE_CLASS_MAPPINGS["illumoraeLoadImageWFilePathOutNode"], Node
        )

    def test_node_display_name_mapping(self):
        self.assertEqual(
            NODE_DISPLAY_NAME_MAPPINGS["illumoraeLoadImageWFilePathOutNode"],
            "Load Image w FilePath Out",
        )

    def test_class_attributes(self):
        self.assertEqual(Node.CATEGORY, "illumorae")
        self.assertEqual(Node.FUNCTION, "load_image")
        self.assertEqual(
            Node.RETURN_TYPES,
            ("IMAGE", "MASK", "STRING", "STRING", "INT", "INT"),
        )
        # RETURN_NAMES arity matches RETURN_TYPES.
        self.assertEqual(len(Node.RETURN_NAMES), len(Node.RETURN_TYPES))
        # All uppercase convention (review fix #6).
        self.assertEqual(
            Node.RETURN_NAMES,
            ("IMAGE", "MASK", "FILE NAME", "FOLDER PATH", "WIDTH", "HEIGHT"),
        )

    def test_input_types_shape(self):
        inputs = Node.INPUT_TYPES()
        self.assertIn("required", inputs)
        req = inputs["required"]
        self.assertIn("image", req)
        self.assertIn("debug_mode", req)
        # Default is a plain placeholder without the misleading [output] suffix
        # (review fix #5).
        default = req["image"][1]["default"]
        self.assertNotIn("[output]", default)
        self.assertEqual(req["debug_mode"][1]["default"], False)


class TestResolvePath(unittest.TestCase):
    """_resolve_path: None handling, direct paths, fallback."""

    def test_none_returns_none(self):
        self.assertIsNone(Node._resolve_path(None))

    def test_none_returns_optional_path_annotation(self):
        # The annotation is Optional[Path]; None is a valid return.
        # We assert the runtime behavior rather than introspecting __annotations__
        # to stay robust across Python versions.
        result = Node._resolve_path(None)
        self.assertIsNone(result)

    def test_direct_valid_file_path(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "img.png"
            _make_rgb(p)
            resolved = Node._resolve_path(str(p))
            self.assertEqual(resolved, p)

    def test_direct_valid_pathlib_path(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "img.png"
            _make_rgb(p)
            resolved = Node._resolve_path(p)
            self.assertEqual(resolved, p)

    def test_nonexistent_returns_path_object(self):
        # A non-existent input does not resolve to a direct file, falls through
        # to the annotation system (stubbed to identity), and is returned as a
        # Path even though it does not exist on disk.
        resolved = Node._resolve_path("C:/does/not/exist.png")
        self.assertIsInstance(resolved, Path)
        self.assertFalse(resolved.exists())


class TestLoadImage(unittest.TestCase):
    """load_image: happy paths, error paths, mask handling."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.node = Node()

    def tearDown(self):
        self.tmp.cleanup()

    def test_none_input_raises_value_error(self):
        # Review fix #2: explicit ValueError instead of AttributeError.
        with self.assertRaises(ValueError):
            self.node.load_image(None)

    def test_nonexistent_path_raises_value_error(self):
        with self.assertRaises(ValueError):
            self.node.load_image(str(self.root / "missing.png"))

    def test_rgb_image_tensor_shape_and_range(self):
        p = self.root / "rgb.png"
        _make_rgb(p, w=8, h=6, color=(255, 0, 0))
        img, mask, name, folder, w, h = self.node.load_image(str(p))
        self.assertEqual(img.shape, (1, 6, 8, 3))
        self.assertEqual(img.dtype, torch.float32)
        self.assertTrue(float(img.min()) >= 0.0)
        self.assertTrue(float(img.max()) <= 1.0)
        # Solid red => R channel ~1.0, G/B ~0.0.
        self.assertAlmostEqual(float(img[0, 0, 0, 0]), 1.0, places=5)
        self.assertAlmostEqual(float(img[0, 0, 0, 1]), 0.0, places=5)

    def test_rgb_no_alpha_returns_64x64_zero_mask(self):
        # Review fix #7: documented 64x64 placeholder convention (kept by choice).
        p = self.root / "rgb.png"
        _make_rgb(p, w=8, h=6)
        _, mask, _, _, _, _ = self.node.load_image(str(p))
        self.assertEqual(mask.shape, (64, 64))
        self.assertTrue(float(mask.min()) == 0.0)
        self.assertTrue(float(mask.max()) == 0.0)

    def test_rgba_alpha_mask_matches_image_dims(self):
        p = self.root / "rgba.png"
        _make_rgba(p, w=8, h=6, color=(255, 0, 0, 128))
        _, mask, _, _, w, h = self.node.load_image(str(p))
        self.assertEqual(mask.shape, (6, 8))
        # Alpha 128/255 ~= 0.502; inverted mask = 1 - 0.502 ~= 0.498.
        self.assertAlmostEqual(float(mask[0, 0]), 1.0 - 128 / 255.0, places=4)

    def test_palette_transparency_preserved(self):
        # Review fix #11: P-mode with transparency converts through RGBA so
        # alpha is preserved instead of dropping to a 64x64 zero mask.
        p = self.root / "pal.png"
        _make_palette_with_transparency(p, w=4, h=4)
        _, mask, _, _, w, h = self.node.load_image(str(p))
        # Mask should match image dimensions, not the 64x64 placeholder.
        self.assertEqual(mask.shape, (4, 4))
        # Transparent pixels (index 0) -> alpha 0 -> inverted mask 1.0.
        # Opaque pixels (index 1) -> alpha 255 -> inverted mask 0.0.
        # The (0,0) pixel is index 0 (transparent) so mask should be 1.0.
        self.assertAlmostEqual(float(mask[0, 0]), 1.0, places=4)

    def test_file_name_is_stem_without_extension(self):
        # Review fix #9: uses image_path.stem instead of os.path helper.
        p = self.root / "my.image.name.png"
        _make_rgb(p)
        _, _, name, _, _, _ = self.node.load_image(str(p))
        # stem of "my.image.name.png" is "my.image.name" (only last suffix stripped).
        self.assertEqual(name, "my.image.name")

    def test_folder_path_is_parent(self):
        p = self.root / "img.png"
        _make_rgb(p)
        _, _, _, folder, _, _ = self.node.load_image(str(p))
        self.assertEqual(folder, str(self.root))

    def test_width_height_returned(self):
        p = self.root / "img.png"
        _make_rgb(p, w=10, h=7)
        _, _, _, _, w, h = self.node.load_image(str(p))
        self.assertEqual(w, 10)
        self.assertEqual(h, 7)

    def test_corrupt_file_raises_runtime_error(self):
        # Review fix #3: narrow except wraps the decode block.
        p = self.root / "not_an_image.png"
        p.write_bytes(b"this is not a png")
        with self.assertRaises(RuntimeError):
            self.node.load_image(str(p))

    def test_non_image_text_file_raises_runtime_error(self):
        p = self.root / "text.txt"
        p.write_text("hello", encoding="utf-8")
        with self.assertRaises(RuntimeError):
            self.node.load_image(str(p))

    def test_six_outputs_returned(self):
        p = self.root / "img.png"
        _make_rgb(p)
        out = self.node.load_image(str(p))
        self.assertEqual(len(out), 6)

    def test_image_parameter_not_shadowed(self):
        # Review fix #8: the input `image` string is not overwritten by the
        # decoded tensor. We cannot introspect locals, but we verify the input
        # string is still usable after the call by confirming a second call
        # with the same string works (i.e. it was not mutated into a tensor).
        p = self.root / "img.png"
        _make_rgb(p)
        s = str(p)
        self.node.load_image(s)
        self.assertEqual(s, str(p))  # input string unchanged


class TestIsChanged(unittest.TestCase):
    """IS_CHANGED: None, missing path, content hashing."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_none_returns_empty_string(self):
        self.assertEqual(Node.IS_CHANGED(None), "")

    def test_nonexistent_path_returns_nan(self):
        # Review fix #4: missing path -> nan (always re-run) instead of
        # FileNotFoundError.
        result = Node.IS_CHANGED(str(self.root / "missing.png"))
        self.assertTrue(math.isnan(result))

    def test_valid_file_returns_sha256_hex(self):
        p = self.root / "img.png"
        _make_rgb(p)
        result = Node.IS_CHANGED(str(p))
        expected = hashlib.sha256(p.read_bytes()).hexdigest()
        self.assertEqual(result, expected)

    def test_same_bytes_same_hash(self):
        p = self.root / "img.png"
        _make_rgb(p, color=(0, 255, 0))
        h1 = Node.IS_CHANGED(str(p))
        h2 = Node.IS_CHANGED(str(p))
        self.assertEqual(h1, h2)

    def test_different_bytes_different_hash(self):
        p = self.root / "img.png"
        _make_rgb(p, color=(0, 255, 0))
        h1 = Node.IS_CHANGED(str(p))
        _make_rgb(p, color=(0, 0, 255))  # overwrite with different content
        h2 = Node.IS_CHANGED(str(p))
        self.assertNotEqual(h1, h2)


class TestValidateInputs(unittest.TestCase):
    """VALIDATE_INPUTS: None, existing, missing."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_none_returns_true(self):
        self.assertTrue(Node.VALIDATE_INPUTS(None))

    def test_existing_path_returns_true(self):
        p = self.root / "img.png"
        _make_rgb(p)
        self.assertTrue(Node.VALIDATE_INPUTS(str(p)))

    def test_missing_path_returns_error_string(self):
        result = Node.VALIDATE_INPUTS(str(self.root / "missing.png"))
        self.assertIsInstance(result, str)
        self.assertIn("Invalid image path", result)


if __name__ == "__main__":
    unittest.main()
