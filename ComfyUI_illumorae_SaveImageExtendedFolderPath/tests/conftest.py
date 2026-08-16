"""Shared test fixtures for the SaveImageExtendedFolderPath package tests.

The node module imports ``folder_paths`` at module load time. That module is
provided by the ComfyUI runtime and is not installed in the test environment,
so a lightweight stub is injected into ``sys.modules`` before the node is
imported. The stub exposes ``get_output_directory`` and
``get_save_image_path`` — the two functions the node actually calls.

This file also puts the package directory on ``sys.path`` so the node module
can be imported by its bare name (matching the convention used by the other
illumorae per-package test suites).

Usage:
    python -m pytest tests/
    python -m tests.test_save_image_extended_folderpath
"""
from __future__ import annotations

import os
import sys
import types

import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PACKAGE_DIR = os.path.dirname(_THIS_DIR)
if _PACKAGE_DIR not in sys.path:
    sys.path.insert(0, _PACKAGE_DIR)


# ---------------------------------------------------------------------------
# Mutable state shared with the stubs
# ---------------------------------------------------------------------------

_STATE = {"output_dir": os.path.join(_THIS_DIR, "_test_output_tmp")}


def _get_output_directory() -> str:
    return _STATE["output_dir"]


def _get_save_image_path(filename_prefix, output_dir, width, height):
    """Minimal stub for folder_paths.get_save_image_path.

    The real function creates the output folder, resolves subfolders, and
    appends a counter to the prefix if needed. For tests we simply return
    the output_dir as the full folder and the prefix unchanged. The node
    only uses the first two return values (full_output_folder, filename);
    the remaining three are discarded.
    """
    os.makedirs(output_dir, exist_ok=True)
    return (output_dir, filename_prefix, 0, "", filename_prefix)


# --- folder_paths stub ---

_folder_paths_stub = types.ModuleType("folder_paths")
_folder_paths_stub.get_output_directory = _get_output_directory
_folder_paths_stub.get_save_image_path = _get_save_image_path
sys.modules["folder_paths"] = _folder_paths_stub


# ---------------------------------------------------------------------------
# Helpers exposed to test modules
# ---------------------------------------------------------------------------


class FakeTensor:
    """Minimal stand-in for a torch tensor with ``.cpu().numpy()``.

    The node only calls ``image.cpu().numpy()`` and then multiplies by 255.
    We wrap a numpy array so no torch dependency is needed.
    """

    def __init__(self, array: np.ndarray):
        self._array = array

    @property
    def shape(self):
        return self._array.shape

    def cpu(self):
        return self

    def numpy(self):
        return self._array

    def __len__(self):
        return 1


class FakeImageBatch:
    """Wraps a list of FakeTensor frames so ``len()`` and indexing work
    like a torch tensor batch of shape ``(N, H, W, C)``.
    """

    def __init__(self, tensors):
        self._tensors = tensors

    def __len__(self):
        return len(self._tensors)

    def __getitem__(self, idx):
        return self._tensors[idx]


def make_test_batch(frames: int = 2, height: int = 8, width: int = 8) -> FakeImageBatch:
    """Return a FakeImageBatch with deterministic pixel values.

    Each frame is a solid color so the saved PNG is small and deterministic.
    Frame ``i`` gets the color ``(i*40, 100, 200)`` clipped to [0, 255].
    """
    tensors = []
    for i in range(frames):
        r = min(i * 40, 255)
        arr = np.full((height, width, 3), [r, 100, 200], dtype=np.float32) / 255.0
        tensors.append(FakeTensor(arr))
    return FakeImageBatch(tensors)
