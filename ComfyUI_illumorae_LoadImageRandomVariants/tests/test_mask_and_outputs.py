"""Regression tests for mask shape and output metadata (review items 2.3, 2.9, 3.6).

Covers:
- 2.3: The default no-alpha mask must match the returned image's H/W, not a
  hardcoded 64x64 tensor.
- 2.9: ``file_name`` is the stem (no extension) and ``folder_path`` is the
  parent directory of the chosen file.
- 3.6: ``RETURN_NAMES`` uses lowercase ``"mask"`` (not ``"MASK"``).
- Alpha-channel images produce a mask derived from the alpha channel with
  the expected inversion (1 - alpha/255).

Usage:
    python -m tests.test_mask_and_outputs
    pytest tests/test_mask_and_outputs.py
"""
from __future__ import annotations

import os
import sys

# conftest.py (same directory) installs the folder_paths stub and puts the
# package directory on sys.path before collection.

from load_image_random_variant import illumoraeLoadImageRandomVariantNode  # noqa: E402

_NODE = illumoraeLoadImageRandomVariantNode()


def _make_rgb(path: str, size=(20, 16)):
    from PIL import Image

    Image.new("RGB", size, (128, 64, 32)).save(path)


def _make_rgba(path: str, size=(20, 16), alpha_value=128):
    from PIL import Image

    img = Image.new("RGBA", size, (128, 64, 32, alpha_value))
    img.save(path)


def test_no_alpha_mask_matches_image_shape(tmp_path):
    """Default mask H/W must equal the image H/W (2.3)."""
    folder = str(tmp_path)
    _make_rgb(os.path.join(folder, "image.png"), size=(20, 16))
    image, mask, fname, fpath = _NODE.load_image(
        folder=folder,
        base_filename="image",
        extension=".png",
        variant_suffixes="_CAM_ORTHO_PROJ_",
        search_variants=False,
        seed=1,
        variant_index_override=1,
        debug_mode=False,
    )
    # image shape: (1, H, W, 3)
    assert image.shape[1] == 16 and image.shape[2] == 20, f"image shape {image.shape!r}"
    # mask should match H, W (not 64x64)
    assert mask.shape[0] == 16 and mask.shape[1] == 20, (
        f"mask shape {mask.shape!r} does not match image H/W (16, 20)"
    )
    # default mask is all zeros
    assert mask.abs().max().item() == 0.0, "default mask should be all zeros"


def test_alpha_mask_is_inverted_alpha(tmp_path):
    """An RGBA image produces a mask = 1 - alpha/255."""
    folder = str(tmp_path)
    _make_rgba(os.path.join(folder, "image.png"), size=(10, 10), alpha_value=128)
    image, mask, fname, fpath = _NODE.load_image(
        folder=folder,
        base_filename="image",
        extension=".png",
        variant_suffixes="_CAM_ORTHO_PROJ_",
        search_variants=False,
        seed=1,
        variant_index_override=1,
        debug_mode=False,
    )
    expected = 1.0 - (128 / 255.0)
    # mask values should all be ~expected
    assert abs(mask.float().max().item() - expected) < 1e-5, (
        f"alpha mask max {mask.max().item()!r} != expected {expected!r}"
    )


def test_file_name_is_stem_without_extension(tmp_path):
    """file_name output is the stem, no extension (2.9)."""
    folder = str(tmp_path)
    _make_rgb(os.path.join(folder, "my_image.png"), size=(8, 8))
    image, mask, fname, fpath = _NODE.load_image(
        folder=folder,
        base_filename="my_image",
        extension=".png",
        variant_suffixes="_CAM_ORTHO_PROJ_",
        search_variants=False,
        seed=1,
        variant_index_override=1,
        debug_mode=False,
    )
    assert fname == "my_image", f"expected stem 'my_image', got {fname!r}"
    assert fpath == folder, f"expected folder path {folder!r}, got {fpath!r}"


def test_base_filename_with_extension_is_stripped(tmp_path):
    """If the user passes 'my_image.png' as base_filename, the node strips it."""
    folder = str(tmp_path)
    _make_rgb(os.path.join(folder, "my_image.png"), size=(8, 8))
    image, mask, fname, fpath = _NODE.load_image(
        folder=folder,
        base_filename="my_image.png",  # accidental extension
        extension=".png",
        variant_suffixes="_CAM_ORTHO_PROJ_",
        search_variants=False,
        seed=1,
        variant_index_override=1,
        debug_mode=False,
    )
    assert fname == "my_image", f"extension not stripped; got {fname!r}"


def test_return_names_lowercase_mask():
    """RETURN_NAMES must use lowercase 'mask' (3.6)."""
    cls = illumoraeLoadImageRandomVariantNode
    assert "mask" in cls.RETURN_NAMES, f"RETURN_NAMES={cls.RETURN_NAMES!r}"
    assert "MASK" not in cls.RETURN_NAMES, (
        f"RETURN_NAMES still has uppercase 'MASK': {cls.RETURN_NAMES!r}"
    )


def test_extension_without_dot_is_normalized(tmp_path):
    """Passing extension='png' (no dot) should still load the image."""
    folder = str(tmp_path)
    _make_rgb(os.path.join(folder, "image.png"), size=(8, 8))
    image, mask, fname, fpath = _NODE.load_image(
        folder=folder,
        base_filename="image",
        extension="png",  # no dot
        variant_suffixes="_CAM_ORTHO_PROJ_",
        search_variants=False,
        seed=1,
        variant_index_override=1,
        debug_mode=False,
    )
    assert fname == "image", f"extension normalization failed; got {fname!r}"


if __name__ == "__main__":
    import pytest

    sys.exit(pytest.main([__file__, "-v"]))
