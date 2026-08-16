"""Shared test fixtures for the CheckpointRandomSelector package tests.

The node module imports only the Python standard library (``os``, ``random``,
``hashlib``, ``datetime``), so no ComfyUI-runtime stub is required. This file
puts the package directory on ``sys.path`` so the node module can be imported
by its bare name, matching the convention used by the other illumorae
per-package test suites.
"""
from __future__ import annotations

import os
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PACKAGE_DIR = os.path.dirname(_THIS_DIR)
if _PACKAGE_DIR not in sys.path:
    sys.path.insert(0, _PACKAGE_DIR)
