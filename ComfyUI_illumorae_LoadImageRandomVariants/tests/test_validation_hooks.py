"""Regression tests for the ComfyUI lifecycle hooks (review item 2.1).

Covers:
- 2.1: ``IS_CHANGED`` and ``VALIDATE_INPUTS`` must reference the node's real
  inputs (``folder``, ``base_filename``, ``variant_index_override``, ...),
  not a non-existent ``image`` argument. Previously both hooks had
  ``image=None`` signatures and were effectively no-ops.
  - ``IS_CHANGED`` returns ``float("nan")`` (always re-execute) when no
    explicit override is set, and ``False`` (cacheable) when a 1-based
    override is set.
  - ``VALIDATE_INPUTS`` returns ``True`` for a valid folder and an error
    string for a missing folder or empty base_filename.

Usage:
    python -m tests.test_validation_hooks
    pytest tests/test_validation_hooks.py
"""
from __future__ import annotations

import math
import os
import sys

# conftest.py (same directory) installs the folder_paths stub and puts the
# package directory on sys.path before collection.

from load_image_random_variant import illumoraeLoadImageRandomVariantNode  # noqa: E402

cls = illumoraeLoadImageRandomVariantNode


def test_is_changed_returns_nan_without_override():
    """No override -> always re-execute (nan) so a new variant can be picked."""
    result = cls.IS_CHANGED(
        folder="C:/input",
        base_filename="image",
        extension=".png",
        variant_suffixes="_CAM_ORTHO_PROJ_",
        search_variants=True,
        seed=-1,
        variant_index_override=-1,
        debug_mode=False,
    )
    assert isinstance(result, float) and math.isnan(result), (
        f"expected nan for random mode, got {result!r}"
    )


def test_is_changed_returns_false_with_override():
    """A 1-based override makes the result stable -> cacheable (False)."""
    result = cls.IS_CHANGED(
        folder="C:/input",
        base_filename="image",
        extension=".png",
        variant_suffixes="_CAM_ORTHO_PROJ_",
        search_variants=True,
        seed=-1,
        variant_index_override=2,
        debug_mode=False,
    )
    assert result is False, f"expected False for override mode, got {result!r}"


def test_is_changed_does_not_take_image_argument():
    """The hook signature must not require an 'image' kwarg (regression for 2.1)."""
    # Calling with only the real inputs must not raise; in particular no
    # 'image' argument is required.
    result = cls.IS_CHANGED(variant_index_override=-1)
    assert isinstance(result, float) and math.isnan(result)


def test_validate_inputs_accepts_valid_folder(tmp_path):
    """VALIDATE_INPUTS returns True for an existing folder."""
    result = cls.VALIDATE_INPUTS(folder=str(tmp_path), base_filename="image")
    assert result is True, f"expected True for valid folder, got {result!r}"


def test_validate_inputs_rejects_missing_folder(tmp_path):
    """VALIDATE_INPUTS returns an error string for a non-existent folder."""
    missing = os.path.join(str(tmp_path), "does_not_exist")
    result = cls.VALIDATE_INPUTS(folder=missing, base_filename="image")
    assert isinstance(result, str), f"expected error string, got {result!r}"
    assert "Invalid folder path" in result, f"unexpected error message: {result!r}"


def test_validate_inputs_rejects_empty_base_filename(tmp_path):
    """VALIDATE_INPUTS returns an error string for an empty base_filename."""
    result = cls.VALIDATE_INPUTS(folder=str(tmp_path), base_filename="")
    assert isinstance(result, str), f"expected error string, got {result!r}"
    assert "base_filename" in result, f"unexpected error message: {result!r}"


def test_validate_inputs_does_not_take_image_argument():
    """VALIDATE_INPUTS must work without an 'image' kwarg (regression for 2.1)."""
    result = cls.VALIDATE_INPUTS(folder=".", base_filename="image")
    # '.' always exists, so this should be True
    assert result is True, f"expected True, got {result!r}"


def test_no_resolve_path_method_remains():
    """The dead _resolve_path helper should be gone (2.1)."""
    assert not hasattr(cls, "_resolve_path"), (
        "_resolve_path should have been removed alongside the dead image-based hooks"
    )


if __name__ == "__main__":
    import pytest

    sys.exit(pytest.main([__file__, "-v"]))
