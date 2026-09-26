"""Chunk domain model."""

from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class Chunk:
    """A token-bounded segment derived from a document section."""

    id: str
    document_id: str
    text: str
    page_number: int
    end_page_number: int
    heading: str | None
    parent_heading: str | None
    token_count: int
    section_id: str
    chunk_index: int
    # Explicit encounter/document date propagated from the source section.
    document_date: str | None = None
    temporal_warnings: tuple[str, ...] = field(default_factory=tuple)
