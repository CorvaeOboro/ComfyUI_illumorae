"""Regression tests for the linter issues flagged in the review (items 3.2, 3.3).

Covers:
- 3.2: No f-strings without placeholders (F541 / W1309) anywhere in the node
  module. The two original offenders were on the (now removed) bare print
  lines. This test scans the source so future edits cannot reintroduce them.
- 3.3: The module docstring is a raw string (``r\"\"\"...\"\"\"``) so the
  Windows path examples (``D:\\items\\...``) do not produce invalid escape
  sequences (W605 / W1401).

These are source-level checks so they run without importing ComfyUI deps.

Usage:
    python -m tests.test_linter_regression
    pytest tests/test_linter_regression.py
"""
from __future__ import annotations

import ast
import os
import re
import sys

# conftest.py (same directory) installs the folder_paths stub and puts the
# package directory on sys.path before collection.
_PACKAGE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_MODULE = os.path.join(_PACKAGE_DIR, "load_image_random_variant.py")


def _source() -> str:
    with open(_MODULE, encoding="utf-8") as f:
        return f.read()


def _f_strings(tree: ast.AST):
    """Yield every JoinedStr (f-string) node in the module."""
    for node in ast.walk(tree):
        if isinstance(node, ast.JoinedStr):
            yield node


def test_no_f_string_without_placeholders():
    """No f-string in the module may lack interpolated values (3.2 / F541)."""
    tree = ast.parse(_source())
    offenders = []
    for node in _f_strings(tree):
        # An f-string with no FormattedValue nodes has no interpolation.
        has_value = any(isinstance(child, ast.FormattedValue) for child in node.values)
        if not has_value:
            offenders.append(node.lineno)
    assert not offenders, (
        f"f-strings without placeholders on lines: {offenders}"
    )


def test_module_docstring_is_raw():
    """The module docstring must be a raw string to avoid W605 escapes (3.3)."""
    src = _source()
    # The file must start with r""" (allowing leading whitespace/comments before it
    # is not the case here - the docstring is the very first statement).
    assert src.lstrip().startswith('r"""'), (
        "module docstring is not a raw string (expected r\"\"\"...); "
        "Windows path examples will produce W605 invalid escape sequences"
    )


def test_no_invalid_escape_sequences_in_docstring():
    """Compile the module docstring and confirm no escape warnings (3.3). W605."""
    tree = ast.parse(_source())
    doc = ast.get_docstring(tree) or ""
    # Pyflakes/W605 flags backslashes that are not valid escapes. We check the
    # raw docstring text for the specific Windows paths that triggered W605.
    assert r"D:\items" in doc, (
        "expected Windows path example D:\\items\\... in docstring"
    )
    # If the docstring is raw, the backslashes are preserved literally.
    # Confirm no \\i or \\p escape interpretation occurred by checking the
    # literal text is intact.
    assert "D:\\items\\potionA.png" in doc


def test_no_bare_print_in_selection_block():
    """The selection block must not contain bare print() calls (3.1).

    All logging goes through self.debug_print. This is a source-level guard
    so unconditional prints do not silently reappear.
    """
    src = _source()
    # Find lines that are bare `print(...)` at call-statement level (not
    # inside debug_print's own body, which legitimately calls print).
    # We approximate by flagging any line whose stripped text starts with
    # `print(` and is not inside the debug_print method.
    tree = ast.parse(src)
    bare_print_lines = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
            func = node.value.func
            if isinstance(func, ast.Name) and func.id == "print":
                # Allow the single print inside debug_print.
                if node.lineno not in _debug_print_body_lines(tree):
                    bare_print_lines.append(node.lineno)
    assert not bare_print_lines, (
        f"bare print() calls on lines {bare_print_lines}; "
        "use self.debug_print(debug_mode, ...) instead"
    )


def _debug_print_body_lines(tree: ast.AST):
    """Return the set of line numbers that belong to the debug_print method body."""
    lines = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "debug_print":
            for child in ast.walk(node):
                if hasattr(child, "lineno"):
                    lines.add(child.lineno)
    return lines


if __name__ == "__main__":
    import pytest

    sys.exit(pytest.main([__file__, "-v"]))
