"""Shared test bootstrap for the SaveAnimatedWebPExtendedFolderPath tests.

ComfyUI's ``folder_paths`` and ``comfy.cli_args`` modules are not importable
outside of a ComfyUI install. We inject lightweight stubs into ``sys.modules``
*before* the node module is imported, pointing the output directory at a
per-test temp folder and exposing ``args.disable_metadata``.

Each test module imports this file first so the stubs are in place:

    from _test_setup import get_output_dir, make_test_tensor, reset_output_dir

The temp output directory is recreated fresh by ``reset_output_dir()`` for
tests that want a clean slate.
"""
from __future__ import annotations

import os
import shutil
import sys
import tempfile
import types

import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PARENT = os.path.dirname(_THIS_DIR)
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)

# ---------------------------------------------------------------------------
# Mutable state shared with the stubs
# ---------------------------------------------------------------------------

_STATE = {"output_dir": tempfile.mkdtemp(prefix="illumorae_test_output_")}


def get_output_dir() -> str:
    """Return the current temp output directory path."""
    return _STATE["output_dir"]


def _get_output_directory() -> str:
    return _STATE["output_dir"]


# --- folder_paths stub ---

_folder_paths_stub = types.ModuleType("folder_paths")
_folder_paths_stub.get_output_directory = _get_output_directory
sys.modules["folder_paths"] = _folder_paths_stub

# --- comfy.cli_args stub ---

_comfy_stub = types.ModuleType("comfy")
_cli_args_stub = types.ModuleType("comfy.cli_args")


class _Args:
    disable_metadata = False


_cli_args_stub.args = _Args()
_comfy_stub.cli_args = _cli_args_stub
sys.modules["comfy"] = _comfy_stub
sys.modules["comfy.cli_args"] = _cli_args_stub


# ---------------------------------------------------------------------------
# Helpers exposed to test modules
# ---------------------------------------------------------------------------


def reset_output_dir() -> str:
    """Wipe and recreate the temp output directory, returning its path."""
    old = _STATE["output_dir"]
    if os.path.isdir(old):
        shutil.rmtree(old, ignore_errors=True)
    _STATE["output_dir"] = tempfile.mkdtemp(prefix="illumorae_test_output_")
    return _STATE["output_dir"]


def set_disable_metadata(value: bool) -> None:
    """Toggle the ``args.disable_metadata`` flag for metadata tests."""
    _cli_args_stub.args.disable_metadata = value


class FakeTensor:
    """Minimal stand-in for a torch tensor with ``.cpu().numpy()``.

    The node only calls ``image.cpu().numpy()`` and then multiplies by 255.
    We wrap a numpy array so no torch dependency is needed.
    """

    def __init__(self, array: np.ndarray):
        self._array = array

    def cpu(self):
        return self

    def numpy(self):
        return self._array


def make_test_tensor(frames: int = 2, height: int = 8, width: int = 8) -> list:
    """Return a list of ``FakeTensor`` frames with deterministic pixel values.

    Each frame is a solid color so the saved WebP is small and deterministic.
    Frame ``i`` gets the color ``(i*40, 100, 200)`` clipped to [0, 255].
    """
    tensors = []
    for i in range(frames):
        r = min(i * 40, 255)
        arr = np.full((height, width, 3), [r, 100, 200], dtype=np.float32) / 255.0
        tensors.append(FakeTensor(arr))
    return tensors
