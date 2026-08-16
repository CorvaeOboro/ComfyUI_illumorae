"""Regression tests for variant discovery (review items 2.6, 2.7).

Covers:
- 2.6: Numbering gaps in the variant folder must not drop later variants.
  Previously the scan incremented an index from 1 and broke on the first
  missing file, so ``_1, _2, _5`` only discovered ``_1`` and ``_2``. The
  glob + natural-sort implementation must find all three.
- 2.7: The base image is always appended to the variants list when it
  exists; the redundant ``base not in variants`` check was removed.

Usage:
    python -m tests.test_variant_discovery
    pytest tests/test_variant_discovery.py
"""
from __future__ import annotations

import os
import sys

# conftest.py (same directory) installs the folder_paths stub and puts the
# package directory on sys.path before collection.

from load_image_random_variant import illumoraeLoadImageRandomVariantNode  # noqa: E402

_NODE = illumoraeLoadImageRandomVariantNode()


def _make_image(path: str, color=(255, 0, 0)):
    from PIL import Image

    Image.new("RGB", (8, 8), color).save(path)


def test_variant_scan_survives_numbering_gap(tmp_path):
    """A gap in variant numbering must not drop later variants (2.6)."""
    folder = str(tmp_path)
    sub = os.path.join(folder, "potionA")
    os.makedirs(sub, exist_ok=True)
    # Create _1, _2, _5 (gap at 3 and 4). Old scan would stop at _2.
    _make_image(os.path.join(sub, "potionA_CAM_ORTHO_PROJ_1.png"), color=(10, 0, 0))
    _make_image(os.path.join(sub, "potionA_CAM_ORTHO_PROJ_2.png"), color=(20, 0, 0))
    _make_image(os.path.join(sub, "potionA_CAM_ORTHO_PROJ_5.png"), color=(50, 0, 0))
    _make_image(os.path.join(folder, "potionA.png"), color=(0, 0, 255))

    # Use override 1..4 to walk every variant the node discovered. The node
    # returns 4 variants (3 numbered + base); an out-of-range override falls
    # back to random rather than raising, so we stop at the known count.
    discovered = []
    for idx in range(1, 5):
        image, mask, fname, fpath = _NODE.load_image(
            folder=folder,
            base_filename="potionA",
            extension=".png",
            variant_suffixes="_CAM_ORTHO_PROJ_",
            search_variants=True,
            seed=-1,
            variant_index_override=idx,
            debug_mode=False,
        )
        discovered.append(fname)

    assert "potionA_CAM_ORTHO_PROJ_5" in discovered, (
        f"variant _5 was dropped by the scan; discovered={discovered!r}"
    )
    # All three numbered variants plus the base image = 4 entries.
    assert len(discovered) == 4, f"expected 4 variants, got {discovered!r}"
    assert set(discovered) == {
        "potionA_CAM_ORTHO_PROJ_1",
        "potionA_CAM_ORTHO_PROJ_2",
        "potionA_CAM_ORTHO_PROJ_5",
        "potionA",
    }, f"unexpected variant set: {set(discovered)!r}"


def test_natural_sort_orders_numbers_numerically(tmp_path):
    """_10 must sort after _2, not before it (lexicographic vs natural, 2.6)."""
    from load_image_random_variant import _natural_sort_key
    from pathlib import Path

    names = [
        "potionA_CAM_ORTHO_PROJ_10.png",
        "potionA_CAM_ORTHO_PROJ_2.png",
        "potionA_CAM_ORTHO_PROJ_1.png",
    ]
    ordered = sorted([Path(n) for n in names], key=_natural_sort_key)
    ordered_names = [p.name for p in ordered]
    assert ordered_names == [
        "potionA_CAM_ORTHO_PROJ_1.png",
        "potionA_CAM_ORTHO_PROJ_2.png",
        "potionA_CAM_ORTHO_PROJ_10.png",
    ], f"natural sort failed: {ordered_names!r}"


def test_base_image_always_included(tmp_path):
    """The base image must be selectable even with no variants (2.7)."""
    folder = str(tmp_path)
    _make_image(os.path.join(folder, "potionA.png"), color=(0, 0, 255))
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
    assert fname == "potionA", f"expected base image, got {fname!r}"


def test_no_variants_no_base_raises(tmp_path):
    """With no base image and no variants, the node must raise FileNotFoundError."""
    folder = str(tmp_path)
    raised = False
    try:
        _NODE.load_image(
            folder=folder,
            base_filename="does_not_exist",
            extension=".png",
            variant_suffixes="_CAM_ORTHO_PROJ_",
            search_variants=True,
            seed=1,
            variant_index_override=-1,
            debug_mode=False,
        )
    except FileNotFoundError:
        raised = True
    assert raised, "expected FileNotFoundError when no image is found"


def test_search_variants_false_loads_base(tmp_path):
    """search_variants=False skips the scan and loads only the base image."""
    folder = str(tmp_path)
    sub = os.path.join(folder, "potionA")
    os.makedirs(sub, exist_ok=True)
    _make_image(os.path.join(sub, "potionA_CAM_ORTHO_PROJ_1.png"), color=(10, 0, 0))
    _make_image(os.path.join(folder, "potionA.png"), color=(0, 0, 255))

    image, mask, fname, fpath = _NODE.load_image(
        folder=folder,
        base_filename="potionA",
        extension=".png",
        variant_suffixes="_CAM_ORTHO_PROJ_",
        search_variants=False,
        seed=1,
        variant_index_override=-1,
        debug_mode=False,
    )
    assert fname == "potionA", f"search_variants=False should load base, got {fname!r}"


if __name__ == "__main__":
    import pytest

    sys.exit(pytest.main([__file__, "-v"]))
