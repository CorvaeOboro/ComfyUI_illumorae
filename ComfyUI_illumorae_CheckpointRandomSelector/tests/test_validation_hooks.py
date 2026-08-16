"""Regression tests for ComfyUI hooks and input validation (review items 2.1, 2.7).

Covers:
- 2.1: ``IS_CHANGED`` returns a value that changes at interval boundaries and
  is stable in between, so ComfyUI's cache invalidates exactly when the
  selection should rotate. An unknown category returns ``float("nan")`` so
  ComfyUI re-executes rather than caching an error.
- 2.7: ``VALIDATE_INPUTS`` rejects unknown category values on the API path
  (the UI ``choices`` constraint does not apply to API calls). The execution
  path also raises ``ValueError`` for an unknown category instead of silently
  falling back to the SDXL folder.

Usage:
    python -m tests.test_validation_hooks
    pytest tests/test_validation_hooks.py
"""
from __future__ import annotations

import os
from datetime import datetime

# conftest.py (same directory) puts the package directory on sys.path.
from checkpoint_random_selector import illumoraeCheckpointRandomSelectorNode as N  # noqa: E402

_NODE = N()


def test_validate_inputs_accepts_known_categories():
    for cat in N._CATEGORIES:
        assert N.VALIDATE_INPUTS(category=cat) is True, f"rejected valid category {cat!r}"


def test_validate_inputs_rejects_unknown_category():
    result = N.VALIDATE_INPUTS(category="FOO")
    assert result is not True, f"accepted unknown category; got {result!r}"
    assert "FOO" in str(result), f"error message missing the bad value: {result!r}"


def test_validate_inputs_rejects_empty_category():
    result = N.VALIDATE_INPUTS(category="")
    assert result is not True, f"accepted empty category; got {result!r}"


def test_is_changed_changes_across_interval_boundary(monkeypatch):
    """IS_CHANGED must return different values before/after a bucket boundary (2.1)."""
    before = datetime(2026, 8, 15, 14, 59, 59)
    after = datetime(2026, 8, 15, 15, 0, 1)
    monkeypatch.setattr(
        "checkpoint_random_selector.datetime",
        type("DT", (), {"now": staticmethod(lambda: before)}),
    )
    v_before = N.IS_CHANGED(category="SDXL", interval_minutes=60)
    monkeypatch.setattr(
        "checkpoint_random_selector.datetime",
        type("DT", (), {"now": staticmethod(lambda: after)}),
    )
    v_after = N.IS_CHANGED(category="SDXL", interval_minutes=60)
    assert v_before != v_after, (
        f"IS_CHANGED did not change across hour boundary: {v_before!r} == {v_after!r}"
    )


def test_is_changed_stable_within_interval(monkeypatch):
    """IS_CHANGED must return the same value within one bucket (2.1)."""
    t1 = datetime(2026, 8, 15, 14, 5, 0)
    t2 = datetime(2026, 8, 15, 14, 55, 0)
    monkeypatch.setattr(
        "checkpoint_random_selector.datetime",
        type("DT", (), {"now": staticmethod(lambda: t1)}),
    )
    v1 = N.IS_CHANGED(category="SDXL", interval_minutes=60)
    monkeypatch.setattr(
        "checkpoint_random_selector.datetime",
        type("DT", (), {"now": staticmethod(lambda: t2)}),
    )
    v2 = N.IS_CHANGED(category="SDXL", interval_minutes=60)
    assert v1 == v2, f"IS_CHANGED was not stable within the hour: {v1!r} != {v2!r}"


def test_is_changed_reflects_category(monkeypatch):
    """Different categories must produce different IS_CHANGED values (2.1)."""
    t = datetime(2026, 8, 15, 14, 5, 0)
    monkeypatch.setattr(
        "checkpoint_random_selector.datetime",
        type("DT", (), {"now": staticmethod(lambda: t)}),
    )
    v_sdxl = N.IS_CHANGED(category="SDXL", interval_minutes=60)
    v_pony = N.IS_CHANGED(category="PONY", interval_minutes=60)
    assert v_sdxl != v_pony, f"category did not affect IS_CHANGED: {v_sdxl!r} == {v_pony!r}"


def test_is_changed_reflects_interval(monkeypatch):
    """Different interval lengths must produce different IS_CHANGED values (2.1)."""
    t = datetime(2026, 8, 15, 14, 5, 0)
    monkeypatch.setattr(
        "checkpoint_random_selector.datetime",
        type("DT", (), {"now": staticmethod(lambda: t)}),
    )
    v_60 = N.IS_CHANGED(category="SDXL", interval_minutes=60)
    v_30 = N.IS_CHANGED(category="SDXL", interval_minutes=30)
    assert v_60 != v_30, f"interval did not affect IS_CHANGED: {v_60!r} == {v_30!r}"


def test_is_changed_unknown_category_returns_nan(monkeypatch):
    """An unknown category returns float('nan') so ComfyUI re-executes (2.1)."""
    t = datetime(2026, 8, 15, 14, 5, 0)
    monkeypatch.setattr(
        "checkpoint_random_selector.datetime",
        type("DT", (), {"now": staticmethod(lambda: t)}),
    )
    v = N.IS_CHANGED(category="FOO", interval_minutes=60)
    assert v != v, f"expected NaN for unknown category, got {v!r}"  # NaN != NaN


def test_select_checkpoint_raises_on_unknown_category(tmp_path):
    """Execution raises ValueError for an unknown category instead of SDXL fallback (2.7)."""
    os.makedirs(os.path.join(str(tmp_path), "SDXL"), exist_ok=True)
    try:
        _NODE.select_checkpoint(
            base_folder=str(tmp_path), category="FOO", interval_minutes=60,
            sdxl_folder_name="SDXL", pony_folder_name="Pony", sd15_folder_name="SD15",
        )
        assert False, "unknown category did not raise"
    except ValueError as e:
        assert "FOO" in str(e), f"error message missing bad value: {e!r}"


def test_select_checkpoint_raises_on_missing_folder(tmp_path):
    """A non-existent category folder raises FileNotFoundError."""
    try:
        _NODE.select_checkpoint(
            base_folder=str(tmp_path), category="SDXL", interval_minutes=60,
            sdxl_folder_name="does_not_exist", pony_folder_name="Pony", sd15_folder_name="SD15",
        )
        assert False, "missing folder did not raise"
    except FileNotFoundError:
        pass


def test_select_checkpoint_raises_on_empty_folder(tmp_path):
    """A folder with no matching checkpoint files raises FileNotFoundError."""
    os.makedirs(os.path.join(str(tmp_path), "SDXL"), exist_ok=True)
    try:
        _NODE.select_checkpoint(
            base_folder=str(tmp_path), category="SDXL", interval_minutes=60,
            sdxl_folder_name="SDXL", pony_folder_name="Pony", sd15_folder_name="SD15",
        )
        assert False, "empty folder did not raise"
    except FileNotFoundError:
        pass


if __name__ == "__main__":
    import pytest
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
