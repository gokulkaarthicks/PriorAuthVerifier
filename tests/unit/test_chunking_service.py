"""Unit tests for ChunkingService."""

from __future__ import annotations

from app.core.config import Settings
from app.models.section import Section, TextSegment
from app.services.chunking_service import ChunkingService
from app.utils.tokens import count_tokens, decode_tokens, encode_text


def _section(
    *,
    text: str,
    heading: str | None = "Section A",
    parent_heading: str | None = "Parent",
    start_page: int = 1,
    end_page: int = 1,
    section_id: str = "doc:section:0000",
    segments: tuple[TextSegment, ...] | None = None,
) -> Section:
    if segments is None:
        segments = (TextSegment(page_number=start_page, text=text),)
    return Section(
        id=section_id,
        document_id="doc",
        heading=heading,
        parent_heading=parent_heading,
        text=text,
        start_page=start_page,
        end_page=end_page,
        requires_ocr=False,
        segments=segments,
    )


def _words(token_target: int) -> str:
    # cl100k_base: simple words are ~1 token each; pad if needed.
    words = [f"word{i}" for i in range(token_target)]
    text = " ".join(words)
    while count_tokens(text) < token_target:
        text += " extra"
    return text


def test_small_section_becomes_one_chunk() -> None:
    service = ChunkingService(settings=Settings(MAX_SECTION_TOKENS=900))
    section = _section(text="Short clinical note.")
    chunks = service.chunk_sections([section])

    assert len(chunks) == 1
    assert chunks[0].text == "Short clinical note."
    assert chunks[0].chunk_index == 0


def test_max_tokens_respected_for_small_sections() -> None:
    service = ChunkingService(settings=Settings(MAX_SECTION_TOKENS=900))
    text = _words(200)
    chunks = service.chunk_sections([_section(text=text)])

    assert len(chunks) == 1
    assert chunks[0].token_count <= 900


def test_oversized_sections_split() -> None:
    service = ChunkingService(
        settings=Settings(MAX_SECTION_TOKENS=50, SECTION_OVERLAP_TOKENS=10)
    )
    text = _words(180)
    chunks = service.chunk_sections([_section(text=text)])

    assert len(chunks) > 1
    assert all(chunk.token_count <= 50 for chunk in chunks)


def test_overlap_applied_within_oversized_section() -> None:
    service = ChunkingService(
        settings=Settings(MAX_SECTION_TOKENS=40, SECTION_OVERLAP_TOKENS=10)
    )
    tokens = encode_text(" ".join(f"tok{i}" for i in range(120)))
    text = decode_tokens(tokens)
    chunks = service.chunk_sections([_section(text=text)])

    assert len(chunks) >= 2
    first_tail = set(encode_text(chunks[0].text)[-10:])
    second_head = set(encode_text(chunks[1].text)[:20])
    assert len(first_tail & second_head) >= 5


def test_separate_sections_do_not_overlap() -> None:
    service = ChunkingService(
        settings=Settings(MAX_SECTION_TOKENS=900, SECTION_OVERLAP_TOKENS=100)
    )
    first = _section(
        text="Alpha unique section content.",
        heading="Alpha",
        section_id="doc:section:0000",
    )
    second = _section(
        text="Beta unique section content.",
        heading="Beta",
        section_id="doc:section:0001",
    )
    chunks = service.chunk_sections([first, second])

    assert len(chunks) == 2
    assert "Alpha unique" in chunks[0].text
    assert "Beta unique" in chunks[1].text
    assert "Alpha unique" not in chunks[1].text
    assert "Beta unique" not in chunks[0].text


def test_page_metadata_survives_splitting() -> None:
    service = ChunkingService(
        settings=Settings(MAX_SECTION_TOKENS=40, SECTION_OVERLAP_TOKENS=5)
    )
    page1 = _words(60)
    page2 = _words(60)
    text = f"{page1}\n\n{page2}"
    section = _section(
        text=text,
        start_page=4,
        end_page=5,
        segments=(
            TextSegment(page_number=4, text=page1),
            TextSegment(page_number=5, text=page2),
        ),
    )
    chunks = service.chunk_sections([section])

    assert len(chunks) > 1
    assert chunks[0].page_number >= 4
    assert chunks[-1].end_page_number <= 5
    assert any(chunk.end_page_number >= chunk.page_number for chunk in chunks)


def test_heading_and_parent_survive_splitting() -> None:
    service = ChunkingService(
        settings=Settings(MAX_SECTION_TOKENS=40, SECTION_OVERLAP_TOKENS=5)
    )
    section = _section(
        text=_words(120),
        heading="Clinical Indication Summary",
        parent_heading="8. PLANNED PROCEDURE",
    )
    chunks = service.chunk_sections([section])

    assert len(chunks) > 1
    assert all(chunk.heading == "Clinical Indication Summary" for chunk in chunks)
    assert all(chunk.parent_heading == "8. PLANNED PROCEDURE" for chunk in chunks)
    assert all(chunk.section_id == section.id for chunk in chunks)


def test_section_order_is_deterministic() -> None:
    service = ChunkingService(settings=Settings(MAX_SECTION_TOKENS=900))
    sections = [
        _section(text="First", heading="One", section_id="doc:section:0000"),
        _section(text="Second", heading="Two", section_id="doc:section:0001"),
        _section(text="Third", heading="Three", section_id="doc:section:0002"),
    ]
    first = service.chunk_sections(sections)
    second = service.chunk_sections(sections)

    assert [chunk.id for chunk in first] == [chunk.id for chunk in second]
    assert [chunk.heading for chunk in first] == ["One", "Two", "Three"]
