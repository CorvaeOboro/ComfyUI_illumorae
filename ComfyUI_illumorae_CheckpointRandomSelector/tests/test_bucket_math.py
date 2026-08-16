"""Regression tests for the interval-bucket math (review item 2.2).

The original implementation computed the bucket with
``now - timedelta(minutes=now.minute % interval_minutes, ...)`` which is only
correct for intervals that divide 60 evenly. For any other value (e.g. 90,
120, 1440) the node silently degraded to hourly rotation or produced
inconsistent buckets. The fix floors the epoch timestamp to the interval
length, which is correct for every integer interval in [1, 1440].

Buckets are aligned to the Unix epoch (UTC), not to local midnight, so these
tests compute expected bucket starts from the epoch directly to remain
timezone-independent.

Usage:
    python -m tests.test_bucket_math
    pytest tests/test_bucket_math.py
"""
from __future__ import annotations

from datetime import datetime

# conftest.py (same directory) puts the package directory on sys.path.
from checkpoint_random_selector import illumoraeCheckpointRandomSelectorNode as N  # noqa: E402


def _bucket(now, interval_minutes):
    return N._bucket_start(now, interval_minutes)


def _expected_bucket_epoch(now, interval_minutes):
    """Compute the expected bucket start epoch directly from the timestamp."""
    epoch = int(now.timestamp())
    return epoch - (epoch % (interval_minutes * 60))


def test_divisor_of_60_same_hour_bucket():
    """60-minute interval: all minutes within the same hour share a bucket."""
    b_early = _bucket(datetime(2026, 8, 15, 14, 35, 12), 60)
    b_late = _bucket(datetime(2026, 8, 15, 14, 59, 59), 60)
    assert b_early == b_late, f"same-hour buckets differ: {b_early!r} vs {b_late!r}"


def test_divisor_of_60_crosses_hour_boundary():
    """60-minute interval: the bucket changes at the top of the next hour."""
    b_before = _bucket(datetime(2026, 8, 15, 14, 59, 59), 60)
    b_after = _bucket(datetime(2026, 8, 15, 15, 0, 1), 60)
    assert b_before != b_after, f"hour boundary not crossed: {b_before!r} == {b_after!r}"


def test_30min_buckets_at_half_hour():
    """30-minute interval: buckets align to :00 and :30 past the hour."""
    b_top = _bucket(datetime(2026, 8, 15, 14, 5), 30)
    b_half = _bucket(datetime(2026, 8, 15, 14, 35), 30)
    assert b_top != b_half, f"30-min boundary not crossed: {b_top!r} == {b_half!r}"
    # The two buckets are 30 minutes apart.
    delta = (b_half - b_top).total_seconds()
    assert delta == 1800, f"30-min bucket spacing wrong: {delta}s"


def test_1440_does_not_collapse_to_hourly():
    """1440-minute (24h) interval must NOT rotate hourly (the original bug).

    Under the old math, 1440 produced the same buckets as 60 because
    ``minute`` is always < 1440. With epoch flooring, two timestamps one hour
    apart share the same daily bucket (they do not rotate).
    """
    b_early = _bucket(datetime(2026, 8, 15, 10, 0), 1440)
    b_later = _bucket(datetime(2026, 8, 15, 11, 0), 1440)
    assert b_early == b_later, (
        f"1440-min interval rotated hourly: {b_early!r} != {b_later!r}"
    )


def test_1440_changes_at_next_day():
    """1440-minute interval: the bucket changes after 24 hours."""
    b1 = _bucket(datetime(2026, 8, 15, 10, 0), 1440)
    b2 = _bucket(datetime(2026, 8, 16, 10, 0), 1440)
    assert b1 != b2, f"1440-min did not change after 24h: {b1!r} == {b2!r}"
    delta = (b2 - b1).total_seconds()
    assert delta == 86400, f"1440-min bucket spacing wrong: {delta}s"


def test_90min_not_silently_hourly():
    """90-minute interval must not collapse to hourly rotation (the original bug).

    Under the old ``minute % interval`` math, 90 produced the same buckets as
    60. With epoch flooring the buckets are 90 minutes apart. Two timestamps
    90 minutes apart at a boundary must have different buckets.
    """
    # Pick a timestamp on a 90-minute boundary (epoch-aligned).
    # 2026-08-15 00:00:00 local may not be on a 90-min boundary, so compute
    # from the epoch: find a time where epoch % (90*60) == 0.
    t0 = datetime(2026, 8, 15, 0, 0, 0)
    e0 = int(t0.timestamp())
    offset = e0 % (90 * 60)
    boundary_epoch = e0 - offset  # first 90-min boundary at or before t0
    from datetime import timedelta
    b_start = datetime.fromtimestamp(boundary_epoch)
    b_in_bucket = b_start + timedelta(minutes=89)
    b_next_bucket = b_start + timedelta(minutes=90)
    assert _bucket(b_start, 90) == _bucket(b_in_bucket, 90), (
        f"90-min: 89-min span crossed boundary: {b_start!r} vs {b_in_bucket!r}"
    )
    assert _bucket(b_start, 90) != _bucket(b_next_bucket, 90), (
        f"90-min: boundary not crossed after 90 min: {b_start!r} == {b_next_bucket!r}"
    )


def test_120min_not_silently_hourly():
    """120-minute interval must not collapse to hourly rotation (the original bug)."""
    t0 = datetime(2026, 8, 15, 0, 0, 0)
    e0 = int(t0.timestamp())
    offset = e0 % (120 * 60)
    from datetime import timedelta
    b_start = datetime.fromtimestamp(e0 - offset)
    b_in_bucket = b_start + timedelta(minutes=119)
    b_next_bucket = b_start + timedelta(minutes=120)
    assert _bucket(b_start, 120) == _bucket(b_in_bucket, 120), (
        "120-min: 119-min span crossed boundary"
    )
    assert _bucket(b_start, 120) != _bucket(b_next_bucket, 120), (
        "120-min: boundary not crossed after 120 min"
    )


def test_45min_buckets_are_consistent():
    """45-minute interval produces a consistent grid (was irregular under old math)."""
    t0 = datetime(2026, 8, 15, 0, 0, 0)
    e0 = int(t0.timestamp())
    offset = e0 % (45 * 60)
    from datetime import timedelta
    b_start = datetime.fromtimestamp(e0 - offset)
    b_in_bucket = b_start + timedelta(minutes=44)
    b_next_bucket = b_start + timedelta(minutes=45)
    assert _bucket(b_start, 45) == _bucket(b_in_bucket, 45), (
        "45-min: 44-min span crossed boundary"
    )
    assert _bucket(b_start, 45) != _bucket(b_next_bucket, 45), (
        "45-min: boundary not crossed after 45 min"
    )


def test_min_interval_one_minute():
    """1-minute interval must produce minute-aligned buckets."""
    b_a = _bucket(datetime(2026, 8, 15, 14, 5, 30), 1)
    b_b = _bucket(datetime(2026, 8, 15, 14, 5, 59), 1)
    b_c = _bucket(datetime(2026, 8, 15, 14, 6, 0), 1)
    assert b_a == b_b, f"1-min same-bucket points differ: {b_a!r} != {b_b!r}"
    assert b_b != b_c, f"1-min boundary not crossed: {b_b!r} == {b_c!r}"


def test_bucket_matches_epoch_floor():
    """The bucket start equals the epoch-floored timestamp for any interval."""
    for interval in (1, 7, 30, 45, 60, 90, 120, 1440):
        now = datetime(2026, 8, 15, 14, 35, 12)
        got = _bucket(now, interval)
        expected_epoch = _expected_bucket_epoch(now, interval)
        assert int(got.timestamp()) == expected_epoch, (
            f"interval={interval}: bucket epoch {int(got.timestamp())} != expected {expected_epoch}"
        )


if __name__ == "__main__":
    import pytest
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
