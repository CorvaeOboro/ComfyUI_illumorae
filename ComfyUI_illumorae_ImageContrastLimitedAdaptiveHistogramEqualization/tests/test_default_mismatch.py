"""Regression test for the default-value mismatch between INPUT_TYPES
and the apply_clahe method signature.

The ComfyUI framework calls ``apply_clahe`` with the values declared in
``INPUT_TYPES``. The method's own Python-level defaults are only used
when the function is called directly (e.g. from other tests or scripts).
If the two sets of defaults disagree, the node behaves differently
depending on how it is invoked, which is a correctness bug without indication.

This test also verifies that the active module (the file whose name
matches the package directory) is the one registered in
``NODE_CLASS_MAPPINGS``, not a stale duplicate.

Usage:
    python -m tests.test_default_mismatch
    pytest tests/test_default_mismatch.py
"""
from __future__ import annotations

import os
import sys
import inspect

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PARENT = os.path.dirname(_THIS_DIR)
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)

from image_contrast_limited_adaptive_histogram_equalization import (  # noqa: E402
    illumoraeImageContrastLimitedAdaptiveHistogramEqualizationNode,
    NODE_CLASS_MAPPINGS,
)


def _input_types_defaults(cls) -> dict:
    """Extract the ``default`` value of every optional input declared
    in ``INPUT_TYPES``."""
    defaults: dict = {}
    spec = cls.INPUT_TYPES()
    for section in ("required", "optional"):
        for name, definition in spec.get(section, {}).items():
            if isinstance(definition, tuple) and len(definition) >= 2:
                params = definition[1]
                if isinstance(params, dict) and "default" in params:
                    defaults[name] = params["default"]
    return defaults


def _method_defaults(cls, method_name: str) -> dict:
    """Extract the Python-level default arguments of a method."""
    method = getattr(cls, method_name)
    sig = inspect.signature(method)
    return {
        name: param.default
        for name, param in sig.parameters.items()
        if param.default is not inspect.Parameter.empty
    }


def test_input_types_vs_method_defaults_match():
    """INPUT_TYPES defaults must match apply_clahe signature defaults."""
    cls = illumoraeImageContrastLimitedAdaptiveHistogramEqualizationNode
    ui_defaults = _input_types_defaults(cls)
    method_defaults = _method_defaults(cls, "apply_clahe")

    mismatches = []
    for key in ("clip_limit", "tile_grid_width", "tile_grid_height"):
        ui_val = ui_defaults.get(key)
        meth_val = method_defaults.get(key)
        if ui_val != meth_val:
            mismatches.append(
                f"  {key}: INPUT_TYPES={ui_val!r}  "
                f"apply_clahe()={meth_val!r}"
            )
    assert not mismatches, (
        "Default mismatch between INPUT_TYPES and apply_clahe:\n"
        + "\n".join(mismatches)
    )


def test_registered_class_matches_active_module():
    """NODE_CLASS_MAPPINGS must register the class from the active
    module (the file matching the package directory name), not from a
    stale duplicate like ``image_CLAHE.py``."""
    cls = illumoraeImageContrastLimitedAdaptiveHistogramEqualizationNode
    registered = NODE_CLASS_MAPPINGS[
        "illumoraeImageContrastLimitedAdaptiveHistogramEqualizationNode"
    ]
    assert registered is cls, (
        "NODE_CLASS_MAPPINGS registers a different class than the one "
        "imported from the active module. Check __init__.py imports."
    )


def test_docstring_defaults_match_input_types():
    """The module and class docstrings list default settings; these
    must agree with INPUT_TYPES so users are not misled."""
    cls = illumoraeImageContrastLimitedAdaptiveHistogramEqualizationNode
    ui_defaults = _input_types_defaults(cls)
    # The docstrings claim clip_limit=2.0, grid=8x8.
    docstring_claims = {
        "clip_limit": 2.0,
        "tile_grid_width": 8,
        "tile_grid_height": 8,
    }
    mismatches = []
    for key, claimed in docstring_claims.items():
        actual = ui_defaults.get(key)
        if actual != claimed:
            mismatches.append(
                f"  {key}: docstring says {claimed!r}  "
                f"INPUT_TYPES says {actual!r}"
            )
    assert not mismatches, (
        "Docstring default settings disagree with INPUT_TYPES:\n"
        + "\n".join(mismatches)
    )


if __name__ == "__main__":
    tests = [
        test_input_types_vs_method_defaults_match,
        test_registered_class_matches_active_module,
        test_docstring_defaults_match_input_types,
    ]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL  {t.__name__}\n      {e}")
        except Exception as e:
            failed += 1
            print(f"ERROR {t.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
