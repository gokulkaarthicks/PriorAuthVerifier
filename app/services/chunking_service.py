"""Section-aware token chunking."""

from __future__ import annotations

import re
from collections.abc import Sequence

from app.core.config import Settings, get_settings
from app.core.logging import get_logger
from app.models.chunk import Chunk
from app.models.section import Section, TextSegment
from app.utils.tokens import count_tokens, decode_tokens, encode_text

logger = get_logger(__name__)

_PARAGRAPH_SPLIT_RE = re.compile(r"\n\s*\n+")
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


class ChunkingService:
    """Convert sections into token-bounded chunks with provenance."""

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()

    def chunk_sections(self, sections: Sequence[Section]) -> list[Chunk]:
        """Chunk each section independently without cross-section merging."""
        chunks: list[Chunk] = []
        oversized_splits = 0

        for section in sections:
            if section.requires_ocr:
                logger.info(
                    "chunking_skipped_ocr_section",
                    section_id=section.id,
                    start_page=section.start_page,
                )
                continue

            section_chunks = self._chunk_section(section)
            if len(section_chunks) > 1:
                oversized_splits += 1
            chunks.extend(section_chunks)

        logger.info(
            "chunks_built",
            section_count=len(sections),
            chunk_count=len(chunks),
            oversized_sections_split=oversized_splits,
        )
        return chunks

    def _chunk_section(self, section: Section) -> list[Chunk]:
        text = section.text.strip()
        if not text:
            return []

        max_tokens = self._settings.MAX_SECTION_TOKENS
        overlap_tokens = self._settings.SECTION_OVERLAP_TOKENS
        token_count = count_tokens(text)

        if token_count <= max_tokens:
            return [
                self._make_chunk(
                    section=section,
                    text=text,
                    chunk_index=0,
                    start_page=section.start_page,
                    end_page=section.end_page,
                )
            ]

        windows = self._split_oversized_text(text, max_tokens, overlap_tokens)
        chunks: list[Chunk] = []
        for index, window in enumerate(windows):
            start_page, end_page = self._page_range_for_text(section, window)
            chunks.append(
                self._make_chunk(
                    section=section,
                    text=window,
                    chunk_index=index,
                    start_page=start_page,
                    end_page=end_page,
                )
            )
        return chunks

    def _make_chunk(
        self,
        *,
        section: Section,
        text: str,
        chunk_index: int,
        start_page: int,
        end_page: int,
    ) -> Chunk:
        return Chunk(
            id=f"{section.id}:chunk:{chunk_index:04d}",
            document_id=section.document_id,
            text=text,
            page_number=start_page,
            end_page_number=end_page,
            heading=section.heading,
            parent_heading=section.parent_heading,
            token_count=count_tokens(text),
            section_id=section.id,
            chunk_index=chunk_index,
            document_date=section.document_date,
            temporal_warnings=section.temporal_warnings,
        )

    def _split_oversized_text(
        self,
        text: str,
        max_tokens: int,
        overlap_tokens: int,
    ) -> list[str]:
        units = self._split_units(text)
        windows: list[str] = []
        current = ""

        for unit in units:
            candidate = f"{current}\n\n{unit}".strip() if current else unit
            if current and count_tokens(candidate) > max_tokens:
                windows.append(current.strip())
                overlap = self._overlap_prefix(windows[-1], overlap_tokens)
                current = f"{overlap} {unit}".strip() if overlap else unit
                if count_tokens(current) > max_tokens:
                    hard_parts = self._hard_split_tokens(
                        current,
                        max_tokens,
                        overlap_tokens,
                    )
                    windows.extend(hard_parts[:-1])
                    current = hard_parts[-1]
            else:
                current = candidate

        if current.strip():
            if count_tokens(current) > max_tokens:
                windows.extend(
                    self._hard_split_tokens(current, max_tokens, overlap_tokens)
                )
            else:
                windows.append(current.strip())

        return [window.strip() for window in windows if window.strip()]

    def _split_units(self, text: str) -> list[str]:
        paragraphs = [
            part.strip()
            for part in _PARAGRAPH_SPLIT_RE.split(text)
            if part.strip()
        ]
        if len(paragraphs) > 1:
            return paragraphs

        sentences = [
            part.strip() for part in _SENTENCE_SPLIT_RE.split(text) if part.strip()
        ]
        if len(sentences) > 1:
            return sentences

        return [text.strip()]

    def _overlap_prefix(self, text: str, overlap_tokens: int) -> str:
        if overlap_tokens <= 0 or not text:
            return ""
        tokens = encode_text(text)
        if len(tokens) <= overlap_tokens:
            return text
        return decode_tokens(tokens[-overlap_tokens:]).strip()

    def _hard_split_tokens(
        self,
        text: str,
        max_tokens: int,
        overlap_tokens: int,
    ) -> list[str]:
        tokens = encode_text(text)
        if len(tokens) <= max_tokens:
            return [text]

        windows: list[str] = []
        start = 0
        while start < len(tokens):
            end = min(start + max_tokens, len(tokens))
            windows.append(decode_tokens(tokens[start:end]).strip())
            if end >= len(tokens):
                break
            start = max(end - overlap_tokens, start + 1)
        return [window for window in windows if window]

    def _page_range_for_text(
        self, section: Section, chunk_text: str
    ) -> tuple[int, int]:
        if not section.segments:
            return section.start_page, section.end_page

        needle = chunk_text.strip()
        if not needle:
            return section.start_page, section.end_page

        # Prefer exact/near membership via sliding segment windows.
        pages = self._pages_covering_substring(section.segments, needle)
        if pages:
            return min(pages), max(pages)

        # Fallback: approximate by token overlap with each segment.
        scored: list[tuple[int, int]] = []
        chunk_tokens = set(encode_text(needle))
        for segment in section.segments:
            segment_tokens = set(encode_text(segment.text))
            overlap = len(chunk_tokens & segment_tokens)
            if overlap:
                scored.append((overlap, segment.page_number))
        if scored:
            scored.sort(reverse=True)
            top_pages = {page for _, page in scored[:3]}
            return min(top_pages), max(top_pages)

        return section.start_page, section.end_page

    def _pages_covering_substring(
        self,
        segments: Sequence[TextSegment],
        needle: str,
    ) -> list[int]:
        full_parts: list[str] = []
        page_map: list[int] = []
        for index, segment in enumerate(segments):
            if index > 0:
                full_parts.append("\n")
                page_map.append(segment.page_number)
            full_parts.append(segment.text)
            page_map.extend([segment.page_number] * len(segment.text))

        full_text = "".join(full_parts)
        if len(page_map) != len(full_text):
            return []

        start = full_text.find(needle)
        if start < 0:
            prefix = needle[:120]
            start = full_text.find(prefix)
            if start < 0:
                return []
            end = min(start + len(needle), len(page_map))
        else:
            end = start + len(needle)

        end = min(end, len(page_map))
        if start >= len(page_map) or start < 0:
            return []
        return page_map[start:end]
