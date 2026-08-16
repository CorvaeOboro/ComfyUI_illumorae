"""Shared test fixtures for the CheckpointLoaderByStringDirty package tests.

The node module imports ``folder_paths`` and ``nodes`` at module load time.
Those modules are provided by the ComfyUI runtime and are not installed in the
test environment, so lightweight stubs are injected into ``sys.modules``
before the node is imported.

This file also puts the package directory on ``sys.path`` so the node module
can be imported by its bare name (matching the convention used by the other
illumorae per-package test suites).

Usage:
    python -m pytest tests/
    python -m tests.test_matching
"""
from __future__ import annotations

import os
import sys
import types

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PACKAGE_DIR = os.path.dirname(_THIS_DIR)
if _PACKAGE_DIR not in sys.path:
    sys.path.insert(0, _PACKAGE_DIR)


def _install_stubs() -> None:
    """Install minimal stubs for ComfyUI-only modules."""
    if "folder_paths" not in sys.modules:
        fp = types.ModuleType("folder_paths")
        # Default: no registered checkpoint directories. Individual tests
        # override this via monkeypatching folder_paths.get_folder_paths /
        # get_full_path.
        fp.get_folder_paths = lambda key: []
        fp.get_full_path = lambda key, name: None
        sys.modules["folder_paths"] = fp
    if "nodes" not in sys.modules:
        nd = types.ModuleType("nodes")

        class _CheckpointLoaderSimple:
            """Stub loader that records the name it was called with and
            returns sentinel objects. Tests can inspect ``loaded_names``
            to assert which relative path was actually loaded."""

            loaded_names: list = []

            def load_checkpoint(self, ckpt_name, output_vae=True, output_clip=True):
                type(self).loaded_names.append(ckpt_name)
                return ("MODEL_SENTINEL", "CLIP_SENTINEL", "VAE_SENTINEL")

        nd.CheckpointLoaderSimple = _CheckpointLoaderSimple
        sys.modules["nodes"] = nd


_install_stubs()
