"""Shared test fixtures for the LoadImageRandomVariants package tests.

The node module imports ``folder_paths`` at module load time. That module is
provided by the ComfyUI runtime and is not installed in the test environment,
so a lightweight stub is injected into ``sys.modules`` before the node is
imported. The stub exposes the single attribute the node references.

This file also puts the package directory on ``sys.path`` so the node module
can be imported by its bare name (matching the convention used by the other
illumorae per-package test suites).
"""
from __future__ import annotations

import os
import sys
import types

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PACKAGE_DIR = os.path.dirname(_THIS_DIR)
if _PACKAGE_DIR not in sys.path:
    sys.path.insert(0, _PACKAGE_DIR)

# Inject a stub folder_paths module so importing the node does not fail.
if "folder_paths" not in sys.modules:
    _stub = types.ModuleType("folder_paths")

    def _get_annotated_filepath(path: str) -> str:  # noqa: ANN001
        return path

    _stub.get_annotated_filepath = _get_annotated_filepath
    sys.modules["folder_paths"] = _stub
