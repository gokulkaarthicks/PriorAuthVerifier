"""Lightweight extraction of explicit document/encounter dates from text."""

from __future__ import annotations

import re
from datetime import date, datetime

# Explicit calendar dates only; no relative phrases like "yesterday".
_ISO_DATE_RE = re.compile(r"\b((?:19|20)\d{2})-(\d{2})-(\d{2})\b")
_US_DATE_RE = re.compile(r"\b(\d{1,2})/(\d{1,2})/((?:19|20)\d{2})\b")
_MONTH_NAME_RE = re.compile(
    r"\b("
    r"Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|"
    r"Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|"
    r"Dec(?:ember)?"
    r")\s+(\d{1,2})(?:\s*[–—-]\s*(\d{1,2}))?,?\s+((?:19|20)\d{2})\b",
    re.IGNORECASE,
)
# PDF line breaks often split "November 14," from "2025" onto the next line.
_MONTH_NAME_BROKEN_YEAR_RE = re.compile(
    r"\b("
    r"Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|"
    r"Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|"
    r"Dec(?:ember)?"
    r")\s+(\d{1,2})(?:\s*[–—-]\s*(\d{1,2}))?,?\s*((?:19|20)\d{2})\b",
    re.IGNORECASE,
)

_MONTH_MAP = {
    "jan": 1,
    "january": 1,
    "feb": 2,
    "february": 2,
    "mar": 3,
    "march": 3,
    "apr": 4,
    "april": 4,
    "may": 5,
    "jun": 6,
    "june": 6,
    "jul": 7,
    "july": 7,
    "aug": 8,
    "august": 8,
    "sep": 9,
    "september": 9,
    "oct": 10,
    "october": 10,
    "nov": 11,
    "november": 11,
    "dec": 12,
    "december": 12,
}

_WITHIN_MONTHS_RE = re.compile(
    r"(?:within|in)\s+(?:the\s+)?(?:previous|past|last)\s+(\d+)\s+months?",
    re.IGNORECASE,
)

_SIGNATURE_CONTEXT_RE = re.compile(
    r"(signed|signature|electronically signed|authenticated)\b",
    re.IGNORECASE,
)
_CLINICAL_CONTEXT_RE = re.compile(
    r"("
    r"encounter|service|performed|study|holter|monitoring|test|exam|"
    r"procedure|admission|discharge|clinic|visit|collected|drawn|"
    r"recording|monitored"
    r")\b",
    re.IGNORECASE,
)
_PA_SUBMISSION_CONTEXT_RE = re.compile(
    r"("
    r"prior\s+auth(?:orization)?|prior-authorization|"
    r"pa\s+submission|pa\s+status|authorization\s+submission|"
    r"submission\s+date|\bsubmitted\b"
    r")\b",
    re.IGNORECASE,
)
_PLANNED_PROCEDURE_CONTEXT_RE = re.compile(
    r"("
    r"planned\s+procedure|scheduled\s+for|date\s+of\s+service|"
    r"procedure\s+date|dos\b"
    r")\b",
    re.IGNORECASE,
)
_SUBMITTED_MONTH_DAY_RE = re.compile(
    r"\bsubmitted\s+("
    r"Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|"
    r"Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|"
    r"Dec(?:ember)?"
    r")\s+(\d{1,2})\b",
    re.IGNORECASE,
)
_NEARBY_YEAR_RE = re.compile(r"\b((?:19|20)\d{2})\b")


def extract_explicit_date(*texts: str | None) -> str | None:
    """Return the preferred explicit clinical date as YYYY-MM-DD, else None."""
    primary, _warnings = extract_document_date_with_warnings(*texts)
    return primary


def extract_document_date_with_warnings(
    *texts: str | None,
) -> tuple[str | None, list[str]]:
    """Prefer clinical/service dates; signature is fallback metadata only.

    Never invent or rewrite dates. When multiple distinct source dates appear,
    keep the preferred extracted date and return a temporal warning.
    """
    clinical: list[str] = []
    heading_like: list[str] = []
    signature: list[str] = []
    other: list[str] = []

    for text_index, text in enumerate(texts):
        if not text:
            continue
        sample = _normalize_date_sample(text)
        for start, end, iso in _iter_dates(sample):
            window = sample[max(0, start - 40) : min(len(sample), end + 40)]
            if _SIGNATURE_CONTEXT_RE.search(window):
                signature.append(iso)
            elif text_index == 0 or _CLINICAL_CONTEXT_RE.search(window):
                if text_index == 0 and not _CLINICAL_CONTEXT_RE.search(window):
                    heading_like.append(iso)
                else:
                    clinical.append(iso)
            else:
                other.append(iso)

    preferred_list = clinical or heading_like or other or signature
    if not preferred_list:
        return None, []

    primary = preferred_list[0]
    unique_dates = list(
        dict.fromkeys([*clinical, *heading_like, *other, *signature])
    )
    warnings: list[str] = []
    if len(unique_dates) > 1:
        warnings.append(
            "Conflicting source dates were found "
            f"({', '.join(unique_dates)}); preserved extracted date {primary} "
            "without rewriting chronology."
        )
    return primary, warnings


def extract_analysis_reference_date(*texts: str | None) -> str | None:
    """Resolve the authorization/procedure episode reference date.

    Precedence (deterministic, no LLM, never ``date.today()``):
    1. Explicit PA / authorization submission date
    2. Explicit planned procedure / DOS date
    3. Latest relevant encounter/document date present in the supplied texts
    4. ``None`` when no defensible date exists
    """
    pa_dates: list[str] = []
    planned_dates: list[str] = []
    clinical_dates: list[str] = []

    for text in texts:
        if not text:
            continue
        sample = _normalize_date_sample(text)
        submitted = _extract_submitted_month_day_year(sample)
        if submitted is not None:
            pa_dates.append(submitted)
        for start, end, iso in _iter_dates(sample):
            window = sample[max(0, start - 64) : min(len(sample), end + 64)]
            if _PA_SUBMISSION_CONTEXT_RE.search(window):
                pa_dates.append(iso)
            elif _PLANNED_PROCEDURE_CONTEXT_RE.search(window):
                planned_dates.append(iso)
            elif not _SIGNATURE_CONTEXT_RE.search(window):
                clinical_dates.append(iso)

    if pa_dates:
        return pa_dates[0]
    if planned_dates:
        return planned_dates[0]
    if clinical_dates:
        # Latest relevant chart date, not the earliest/first encounter.
        return max(clinical_dates)
    return None


def _extract_submitted_month_day_year(sample: str) -> str | None:
    """Parse 'Submitted November 14 ... 2025' even when tokens intervene."""
    match = _SUBMITTED_MONTH_DAY_RE.search(sample)
    if match is None:
        return None
    month = _MONTH_MAP.get(match.group(1).lower())
    if month is None:
        return None
    day = int(match.group(2))
    # Prefer a year shortly after the day (handles Auth # / line-break noise).
    tail = sample[match.end() : match.end() + 80]
    year_match = _NEARBY_YEAR_RE.search(tail)
    if year_match is None:
        # Also allow year immediately before in rare layouts.
        head = sample[max(0, match.start() - 40) : match.start()]
        year_match = _NEARBY_YEAR_RE.search(head)
    if year_match is None:
        return None
    parsed = _safe_date(int(year_match.group(1)), month, day)
    return parsed.isoformat() if parsed is not None else None


def parse_within_months_window(temporal_requirement: str | None) -> int | None:
    """Parse an explicit 'within the previous N months' window, else None."""
    if not temporal_requirement:
        return None
    match = _WITHIN_MONTHS_RE.search(temporal_requirement.strip())
    if not match:
        return None
    months = int(match.group(1))
    return months if months > 0 else None


def is_within_months(
    source_date: str | None,
    *,
    months: int,
    reference: date | None = None,
) -> bool | None:
    """Return whether source_date is within ``months`` of reference.

    Returns None when the source date or reference is missing/unparseable so
    callers can keep unknown-date evidence instead of inventing a clock date.
    """
    if not source_date or months <= 0 or reference is None:
        return None
    parsed = _parse_iso(source_date)
    if parsed is None:
        return None
    # Approximate month window as 30-day months; conservative and explicit.
    delta_days = (reference - parsed).days
    if delta_days < 0:
        return True
    return delta_days <= months * 30


def _normalize_date_sample(text: str) -> str:
    """Collapse PDF line breaks so month-name dates stay parseable."""
    # Keep a bounded prefix; collapse runs of whitespace/newlines to spaces.
    return re.sub(r"\s+", " ", text[:2000]).strip()


def _iter_dates(sample: str) -> list[tuple[int, int, str]]:
    found: list[tuple[int, int, str]] = []
    for match in _ISO_DATE_RE.finditer(sample):
        parsed = _safe_date(
            int(match.group(1)),
            int(match.group(2)),
            int(match.group(3)),
        )
        if parsed is not None:
            found.append((match.start(), match.end(), parsed.isoformat()))
    for match in _US_DATE_RE.finditer(sample):
        parsed = _safe_date(
            int(match.group(3)),
            int(match.group(1)),
            int(match.group(2)),
        )
        if parsed is not None:
            found.append((match.start(), match.end(), parsed.isoformat()))
    for pattern in (_MONTH_NAME_RE, _MONTH_NAME_BROKEN_YEAR_RE):
        for match in pattern.finditer(sample):
            month = _MONTH_MAP.get(match.group(1).lower())
            if month is None:
                continue
            # For ranges like "Nov 1–29, 2025", keep the start as the clinical date.
            day = int(match.group(2))
            parsed = _safe_date(int(match.group(4)), month, day)
            if parsed is not None:
                found.append((match.start(), match.end(), parsed.isoformat()))
    found.sort(key=lambda item: item[0])
    # De-dupe identical spans created by overlapping month patterns.
    deduped: list[tuple[int, int, str]] = []
    seen: set[tuple[int, int, str]] = set()
    for item in found:
        if item in seen:
            continue
        seen.add(item)
        deduped.append(item)
    return deduped


def _safe_date(year: int, month: int, day: int) -> date | None:
    try:
        return date(year, month, day)
    except ValueError:
        return None


def _parse_iso(value: str) -> date | None:
    try:
        return datetime.strptime(value, "%Y-%m-%d").date()
    except ValueError:
        return None
