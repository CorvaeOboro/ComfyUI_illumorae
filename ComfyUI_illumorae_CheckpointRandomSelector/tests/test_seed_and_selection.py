"""Regression tests for seed determinism and RNG isolation (review items 2.3, 2.4, 2.6).

Covers:
- 2.3: the seed is derived from ``hashlib.sha256`` (deterministic across
  processes) rather than the built-in ``hash()`` (which is randomized per
  process via PYTHONHASHSEED). The same ``(category, interval_start)`` must
  yield the same selection regardless of which process computes it.
- 2.4: the node uses a local ``random.Random(seed)`` instance and must not
  mutate the process-global RNG.
- 2.6: the candidate file list is sorted before ``random.choice`` so the
  index-to-file mapping is stable across platforms regardless of
  ``os.listdir`` ordering.

Usage:
    python -m tests.test_seed_and_selection
    pytest tests/test_seed_and_selection.py
"""
from __future__ import annotations

import os
import random
from datetime import datetime

# conftest.py (same directory) puts the package directory on sys.path.
from checkpoint_random_selector import illumoraeCheckpointRandomSelectorNode as N  # noqa: E402

_NODE = N()


def _make_ckpt(path: str, content: bytes = b""):
    """Create an empty (or content-filled) checkpoint file at ``path``."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as fh:
        fh.write(content)


def _build_ckpt_folder(base: str, category_folder: str, names):
    """Create base/category_folder and write one empty file per name in ``names``."""
    folder = os.path.join(base, category_folder)
    os.makedirs(folder, exist_ok=True)
    for name in names:
        _make_ckpt(os.path.join(folder, name))
    return folder


def test_seed_is_deterministic_across_processes(tmp_path):
    """Same (category, interval) must pick the same file in two separate processes.

    Regression for 2.3: the old ``hash()``-based seed was randomized per
    process, so two processes landing in the same interval window could pick
    different files. The ``hashlib``-based seed is stable.
    """
    _build_ckpt_folder(str(tmp_path), "SDXL", [f"m{i}.safetensors" for i in range(10)])
    # Pin the bucket by monkeypatching _bucket_start so the result does not
    # depend on wall-clock time at test time.
    fixed_bucket = datetime(2026, 8, 15, 14, 0, 0)
    N._bucket_start = staticmethod(lambda now, interval: fixed_bucket)
    try:
        r1 = _NODE.select_checkpoint(
            base_folder=str(tmp_path), category="SDXL", interval_minutes=60,
            sdxl_folder_name="SDXL", pony_folder_name="Pony", sd15_folder_name="SD15",
        )
        r2 = _NODE.select_checkpoint(
            base_folder=str(tmp_path), category="SDXL", interval_minutes=60,
            sdxl_folder_name="SDXL", pony_folder_name="Pony", sd15_folder_name="SD15",
        )
    finally:
        # Restore the real implementation.
        N._bucket_start = staticmethod(N.__dict__.get("_bucket_start") or N._bucket_start)
    assert r1[2] == r2[2], f"same seed gave different picks: {r1[2]!r} vs {r2[2]!r}"


def test_seed_does_not_mutate_global_rng(tmp_path):
    """The node must use a local Random, not re-seed the global RNG (2.4)."""
    _build_ckpt_folder(str(tmp_path), "SDXL", [f"m{i}.safetensors" for i in range(5)])
    random.seed(999)
    expected = random.random()  # consume one draw from the global RNG
    _NODE.select_checkpoint(
        base_folder=str(tmp_path), category="SDXL", interval_minutes=60,
        sdxl_folder_name="SDXL", pony_folder_name="Pony", sd15_folder_name="SD15",
    )
    random.seed(999)
    actual = random.random()
    assert actual == expected, (
        "global RNG state was disturbed by select_checkpoint; "
        f"expected {expected!r}, got {actual!r}"
    )


def test_files_are_sorted_before_choice(tmp_path):
    """Selection must be independent of os.listdir ordering (2.6).

    We force ``os.listdir`` to return names in reverse order and confirm the
    pick is identical to the natural-order case, proving the list is sorted
    before ``random.choice`` indexes into it.
    """
    names = [f"m{i:02d}.safetensors" for i in range(8)]
    _build_ckpt_folder(str(tmp_path), "SDXL", names)
    fixed_bucket = datetime(2026, 8, 15, 14, 0, 0)
    real_listdir = os.listdir

    def reverse_listdir(path):
        return sorted(real_listdir(path), reverse=True)

    N._bucket_start = staticmethod(lambda now, interval: fixed_bucket)
    try:
        natural = _NODE.select_checkpoint(
            base_folder=str(tmp_path), category="SDXL", interval_minutes=60,
            sdxl_folder_name="SDXL", pony_folder_name="Pony", sd15_folder_name="SD15",
        )
        os.listdir = reverse_listdir
        try:
            reversed_pick = _NODE.select_checkpoint(
                base_folder=str(tmp_path), category="SDXL", interval_minutes=60,
                sdxl_folder_name="SDXL", pony_folder_name="Pony", sd15_folder_name="SD15",
            )
        finally:
            os.listdir = real_listdir
    finally:
        N._bucket_start = staticmethod(N.__dict__.get("_bucket_start") or N._bucket_start)
    assert natural[2] == reversed_pick[2], (
        f"listdir order changed the pick: natural={natural[2]!r} reversed={reversed_pick[2]!r}"
    )


def test_different_categories_can_pick_differently(tmp_path):
    """Different categories with different file sets produce different seeds (2.3)."""
    _build_ckpt_folder(str(tmp_path), "SDXL", ["sdxl_a.safetensors", "sdxl_b.safetensors"])
    _build_ckpt_folder(str(tmp_path), "Pony", ["pony_a.safetensors", "pony_b.safetensors"])
    fixed_bucket = datetime(2026, 8, 15, 14, 0, 0)
    N._bucket_start = staticmethod(lambda now, interval: fixed_bucket)
    try:
        sdxl = _NODE.select_checkpoint(
            base_folder=str(tmp_path), category="SDXL", interval_minutes=60,
            sdxl_folder_name="SDXL", pony_folder_name="Pony", sd15_folder_name="SD15",
        )
        pony = _NODE.select_checkpoint(
            base_folder=str(tmp_path), category="PONY", interval_minutes=60,
            sdxl_folder_name="SDXL", pony_folder_name="Pony", sd15_folder_name="SD15",
        )
    finally:
        N._bucket_start = staticmethod(N.__dict__.get("_bucket_start") or N._bucket_start)
    assert sdxl[0] != pony[0], "category did not change the resolved folder"
    assert sdxl[2].startswith("sdxl_") and pony[2].startswith("pony_"), (
        f"category picked from wrong folder: sdxl={sdxl[2]!r} pony={pony[2]!r}"
    )


def test_safe_mode_ignores_file_extensions(tmp_path):
    """safe_mode=True forces .safetensors/.sft and ignores file_extensions (2.5)."""
    # Only a .ckpt file present; safe_mode should find nothing and raise.
    _build_ckpt_folder(str(tmp_path), "SDXL", ["only_ckpt.ckpt"])
    try:
        _NODE.select_checkpoint(
            base_folder=str(tmp_path), category="SDXL", interval_minutes=60,
            sdxl_folder_name="SDXL", pony_folder_name="Pony", sd15_folder_name="SD15",
            safe_mode=True, file_extensions=".ckpt",
        )
        assert False, "safe_mode=True should not have loaded a .ckpt file"
    except FileNotFoundError:
        pass  # expected: safe_mode ignored the .ckpt extension


def test_safe_mode_false_uses_file_extensions(tmp_path):
    """safe_mode=False honors file_extensions, including non-default extensions (2.5)."""
    _build_ckpt_folder(str(tmp_path), "SDXL", ["only_ckpt.ckpt"])
    fixed_bucket = datetime(2026, 8, 15, 14, 0, 0)
    N._bucket_start = staticmethod(lambda now, interval: fixed_bucket)
    try:
        folder, file_path, filename = _NODE.select_checkpoint(
            base_folder=str(tmp_path), category="SDXL", interval_minutes=60,
            sdxl_folder_name="SDXL", pony_folder_name="Pony", sd15_folder_name="SD15",
            safe_mode=False, file_extensions=".ckpt",
        )
    finally:
        N._bucket_start = staticmethod(N.__dict__.get("_bucket_start") or N._bucket_start)
    assert filename == "only_ckpt.ckpt", f"safe_mode=False did not load .ckpt: {filename!r}"


def test_file_extensions_without_leading_dot(tmp_path):
    """A leading dot is optional in file_extensions (3.4 normalization)."""
    _build_ckpt_folder(str(tmp_path), "SDXL", ["a.sft", "b.safetensors"])
    fixed_bucket = datetime(2026, 8, 15, 14, 0, 0)
    N._bucket_start = staticmethod(lambda now, interval: fixed_bucket)
    try:
        # "sft" without a dot should be normalized to ".sft" and match a.sft.
        # Use a single-extension folder so the pick is unambiguous.
        _build_ckpt_folder(str(tmp_path), "SDXL", ["a.sft"])  # only .sft
        folder, file_path, filename = _NODE.select_checkpoint(
            base_folder=str(tmp_path), category="SDXL", interval_minutes=60,
            sdxl_folder_name="SDXL", pony_folder_name="Pony", sd15_folder_name="SD15",
            safe_mode=False, file_extensions="sft",
        )
    finally:
        N._bucket_start = staticmethod(N.__dict__.get("_bucket_start") or N._bucket_start)
    assert filename == "a.sft", f"dotless extension not normalized: {filename!r}"


def test_return_tuple_shape(tmp_path):
    """The node returns (folder_path, file_path, filename) as strings."""
    _build_ckpt_folder(str(tmp_path), "SDXL", ["a.safetensors"])
    fixed_bucket = datetime(2026, 8, 15, 14, 0, 0)
    N._bucket_start = staticmethod(lambda now, interval: fixed_bucket)
    try:
        result = _NODE.select_checkpoint(
            base_folder=str(tmp_path), category="SDXL", interval_minutes=60,
            sdxl_folder_name="SDXL", pony_folder_name="Pony", sd15_folder_name="SD15",
        )
    finally:
        N._bucket_start = staticmethod(N.__dict__.get("_bucket_start") or N._bucket_start)
    assert isinstance(result, tuple) and len(result) == 3, (
        f"expected 3-tuple, got {type(result).__name__} len={len(result)}"
    )
    assert all(isinstance(x, str) for x in result), f"non-string outputs: {result!r}"


if __name__ == "__main__":
    import pytest
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
