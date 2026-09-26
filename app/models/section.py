"""Section domain model."""

from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class TextSegment:
    """A contiguous text fragment tied to a source page."""

    page_number: int
    text: str


@dataclass(frozen=True, slots=True)
class Section:
    """A lightweight logical document section used for cheap narrowing."""

    id: str
    document_id: str
    heading: str | None
    parent_heading: str | None
    text: str
    start_page: int
    end_page: int
    token_count: int = 0
    requires_ocr: bool = False
    segments: tuple[TextSegment, ...] = field(default_factory=tuple)
    # Explicit encounter/document date from source text when available (YYYY-MM-DD).
    document_date: str | None = None
    # Non-fatal chronology notes when multiple source dates conflict.
    temporal_warnings: tuple[str, ...] = field(default_factory=tuple)
