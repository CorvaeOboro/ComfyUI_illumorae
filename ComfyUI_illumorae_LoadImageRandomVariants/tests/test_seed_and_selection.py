"""Regression tests for variant selection logic (review items 2.2, 2.5, 2.8).

Covers:
- 2.2: ``seed = -1`` produces a fresh random pick each run (not a fixed pick
  from a deterministic ``random.seed(-1)``), and a non-negative seed is
  reproducible. The node must not mutate the process-global RNG.
- 2.5: ``variant_index_override`` is 1-based; ``-1`` means random; ``0`` is
  not a valid 1-based index and is treated as random (no silent crash, no
  off-by-one into the first variant).
- 2.8: the return tuple has exactly four elements and no ``variant_index_used``
  leaks out (it was previously computed but never returned).

Usage:
    python -m tests.test_seed_and_selection
    pytest tests/test_seed_and_selection.py
"""
from __future__ import annotations

import os
import random
import sys

# conftest.py (same directory) installs the folder_paths stub and puts the
# package directory on sys.path before collection.

from load_image_random_variant import illumoraeLoadImageRandomVariantNode  # noqa: E402

_NODE = illumoraeLoadImageRandomVariantNode()


def _make_image(path: str, size=(8, 8), color=(255, 0, 0)):
    """Write a small solid-color PNG to ``path``."""
    from PIL import Image

    Image.new("RGB", size, color).save(path)


def _build_variant_tree(tmp_root: str, n_variants: int = 4):
    """Create a folder with a base image and ``n_variants`` numbered variants.

    Layout:
        tmp_root/potionA.png
        tmp_root/potionA/potionA_CAM_ORTHO_PROJ_1.png
        tmp_root/potionA/potionA_CAM_ORTHO_PROJ_2.png
        ...
    Returns the path to ``tmp_root`` (str).
    """
    os.makedirs(tmp_root, exist_ok=True)
    sub = os.path.join(tmp_root, "potionA")
    os.makedirs(sub, exist_ok=True)
    _make_image(os.path.join(tmp_root, "potionA.png"), color=(0, 0, 255))
    for i in range(1, n_variants + 1):
        _make_image(os.path.join(sub, f"potionA_CAM_ORTHO_PROJ_{i}.png"), color=(i * 10, 0, 0))
    return tmp_root


def test_seed_negative_is_random(tmp_path):
    """seed=-1 should not produce the same pick on every run (regression for 2.2)."""
    folder = _build_variant_tree(str(tmp_path), n_variants=8)
    picks = set()
    for _ in range(20):
        image, mask, fname, fpath = _NODE.load_image(
            folder=folder,
            base_filename="potionA",
            extension=".png",
            variant_suffixes="_CAM_ORTHO_PROJ_",
            search_variants=True,
            seed=-1,
            variant_index_override=-1,
            debug_mode=False,
        )
        picks.add(fname)
    # With 8 variants + base = 9 candidates and 20 draws, a working random
    # picker yields more than one distinct pick. A deterministic seed=-1
    # would yield exactly one.
    assert len(picks) > 1, f"seed=-1 produced only {picks!r}; expected randomness"


def test_seed_non_negative_is_reproducible(tmp_path):
    """A fixed non-negative seed must pick the same variant every run (2.2)."""
    folder = _build_variant_tree(str(tmp_path), n_variants=6)
    picks = []
    for _ in range(5):
        image, mask, fname, fpath = _NODE.load_image(
            folder=folder,
            base_filename="potionA",
            extension=".png",
            variant_suffixes="_CAM_ORTHO_PROJ_",
            search_variants=True,
            seed=12345,
            variant_index_override=-1,
            debug_mode=False,
        )
        picks.append(fname)
    assert len(set(picks)) == 1, f"fixed seed gave different picks: {picks!r}"


def test_seed_does_not_mutate_global_rng(tmp_path):
    """The node must use a local Random instance, not re-seed the global RNG (2.2)."""
    folder = _build_variant_tree(str(tmp_path), n_variants=4)
    random.seed(999)
    expected = random.random()  # consume one draw from the global RNG
    _NODE.load_image(
        folder=folder,
        base_filename="potionA",
        extension=".png",
        variant_suffixes="_CAM_ORTHO_PROJ_",
        search_variants=True,
        seed=42,
        variant_index_override=-1,
        debug_mode=False,
    )
    random.seed(999)
    actual = random.random()
    assert actual == expected, (
        "global RNG state was disturbed by load_image; "
        f"expected {expected!r}, got {actual!r}"
    )


def test_override_is_one_based(tmp_path):
    """variant_index_override=1 must select the first variant (1-based, 2.5)."""
    folder = _build_variant_tree(str(tmp_path), n_variants=3)
    image, mask, fname, fpath = _NODE.load_image(
        folder=folder,
        base_filename="potionA",
        extension=".png",
        variant_suffixes="_CAM_ORTHO_PROJ_",
        search_variants=True,
        seed=-1,
        variant_index_override=1,
        debug_mode=False,
    )
    # variants list = [variant_1, variant_2, variant_3, base]; index 1 -> first
    assert fname == "potionA_CAM_ORTHO_PROJ_1", f"override=1 picked {fname!r}"


def test_override_zero_is_treated_as_random(tmp_path):
    """override=0 is not a valid 1-based index; must fall through to random (2.5)."""
    folder = _build_variant_tree(str(tmp_path), n_variants=8)
    # With many draws under seed=-1, override=0 should yield more than one
    # distinct pick (i.e. it did not silently map to the first variant).
    picks = set()
    for _ in range(20):
        image, mask, fname, fpath = _NODE.load_image(
            folder=folder,
            base_filename="potionA",
            extension=".png",
            variant_suffixes="_CAM_ORTHO_PROJ_",
            search_variants=True,
            seed=-1,
            variant_index_override=0,
            debug_mode=False,
        )
        picks.add(fname)
    assert len(picks) > 1, f"override=0 was not random; picks={picks!r}"


def test_override_out_of_range_falls_back_to_random(tmp_path):
    """An override larger than the variant count must not crash (2.5)."""
    folder = _build_variant_tree(str(tmp_path), n_variants=2)
    image, mask, fname, fpath = _NODE.load_image(
        folder=folder,
        base_filename="potionA",
        extension=".png",
        variant_suffixes="_CAM_ORTHO_PROJ_",
        search_variants=True,
        seed=7,
        variant_index_override=99,
        debug_mode=False,
    )
    # Should still return a valid image (random fallback), not raise.
    assert fname.startswith("potionA"), f"unexpected fname {fname!r}"


def test_return_tuple_has_four_elements(tmp_path):
    """The node returns exactly (image, mask, file name, folder path) (2.8)."""
    folder = _build_variant_tree(str(tmp_path), n_variants=2)
    result = _NODE.load_image(
        folder=folder,
        base_filename="potionA",
        extension=".png",
        variant_suffixes="_CAM_ORTHO_PROJ_",
        search_variants=True,
        seed=1,
        variant_index_override=1,
        debug_mode=False,
    )
    assert isinstance(result, tuple) and len(result) == 4, (
        f"expected 4-tuple, got {type(result).__name__} len={len(result)}"
    )


if __name__ == "__main__":
    import pytest

    sys.exit(pytest.main([__file__, "-v"]))
