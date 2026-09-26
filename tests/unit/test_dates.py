"""Unit tests for explicit date extraction helpers."""

from __future__ import annotations

from datetime import date

from app.utils.dates import (
    extract_explicit_date,
    is_within_months,
    parse_within_months_window,
)


def test_extract_explicit_iso_and_us_dates() -> None:
    assert extract_explicit_date("Encounter 2026-08-10 labs") == "2026-08-10"
    assert extract_explicit_date("Seen on 08/10/2026 for follow-up") == "2026-08-10"
    assert extract_explicit_date("August 10, 2026 TSH normal") == "2026-08-10"
    assert extract_explicit_date("No calendar date here") is None


def test_within_months_window_is_conservative() -> None:
    assert parse_within_months_window("within the previous 6 months") == 6
    assert parse_within_months_window("current") is None
    assert (
        is_within_months(
            "2026-07-01",
            months=6,
            reference=date(2026, 9, 1),
        )
        is True
    )
    assert (
        is_within_months(
            "2021-01-01",
            months=6,
            reference=date(2026, 9, 1),
        )
        is False
    )
    assert is_within_months(None, months=6, reference=date(2026, 9, 1)) is None
    assert is_within_months("2026-07-01", months=6, reference=None) is None
