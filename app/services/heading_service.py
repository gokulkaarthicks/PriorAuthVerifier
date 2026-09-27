"""Heading detection and section construction from parsed PDF pages."""

from __future__ import annotations

import re
import statistics
import time
from collections.abc import Sequence

from app.core.logging import get_logger
from app.core.metrics import get_preprocess_doc_kind, get_request_metrics
from app.models.layout import DetectedHeading, DocumentLine
from app.models.pdf import ParsedDocument, PDFPage, TextSpan
from app.models.section import Section, TextSegment
from app.utils.dates import extract_document_date_with_warnings
from app.utils.tokens import count_tokens

logger = get_logger(__name__)

_LINE_Y_TOLERANCE = 3.0
_MIN_HEADING_SCORE_LARGE = 4.0
_MIN_HEADING_SCORE_BODY = 6.0
_NUMBERED_HEADING_RE = re.compile(
    r"^(?:"
    r"SECTION\s+\d+\s*[:.\-–—]?\s*\S|"
    r"\d+(?:\.\d+)+\s+\S|"
    r"\d+[.)]\s+\S|"
    r"[A-Z]\.\s+\S"
    r")",
    re.IGNORECASE,
)
_MAJOR_NUMBERED_RE = re.compile(
    r"^(?:SECTION\s+\d+|\d+[.)])\s+",
    re.IGNORECASE,
)
_SUB_NUMBERED_RE = re.compile(r"^\d+\.\d+", re.IGNORECASE)
_LABEL_VALUE_RE = re.compile(r"^[^:\n]{1,60}:\s+\S.{8,}")
_SEPARATOR_RE = re.compile(r"^[=_\-–—*]{6,}\s*$")
_COLON_ONLY_HEADING_RE = re.compile(r"^.{2,80}:\s*$")


class HeadingService:
    """Detect headings from layout signals and build document sections."""

    def build_sections(self, document: ParsedDocument) -> list[Section]:
        """Convert a parsed document into ordered sections."""
        lines = self.reconstruct_lines(document.pages)
        body_font_size = self._estimate_body_font_size(lines)
        headings = self.detect_headings(lines, body_font_size=body_font_size)
        sections = self._build_sections_from_lines(
            document_id=document.id,
            lines=lines,
            headings=headings,
            pages=document.pages,
        )

        heading_count = sum(1 for section in sections if section.heading)
        ocr_pages = [
            page.page_number for page in document.pages if page.requires_ocr
        ]
        logger.info(
            "document_sections_created",
            document_type=document.document_type.value,
            document_id=document.id,
            filename=document.filename,
            section_count=len(sections),
            heading_count=heading_count,
            body_font_size=body_font_size,
            ocr_candidate_pages=ocr_pages,
        )
        return sections

    def reconstruct_lines(self, pages: Sequence[PDFPage]) -> list[DocumentLine]:
        """Rebuild visual lines from page spans in reading order."""
        lines: list[DocumentLine] = []
        for page in pages:
            if page.requires_ocr:
                continue
            lines.extend(self._lines_from_page(page))
        return lines

    def detect_headings(
        self,
        lines: Sequence[DocumentLine],
        *,
        body_font_size: float | None = None,
    ) -> list[DetectedHeading]:
        """Score lines and return heading candidates."""
        if not lines:
            return []

        body_size = (
            body_font_size
            if body_font_size is not None
            else self._estimate_body_font_size(lines)
        )
        detected: list[DetectedHeading] = []

        for index, line in enumerate(lines):
            previous = lines[index - 1] if index > 0 else None
            score = self._score_heading_candidate(line, previous, body_size)
            threshold = (
                _MIN_HEADING_SCORE_LARGE
                if line.font_size >= body_size + 1.5
                else _MIN_HEADING_SCORE_BODY
            )
            if score < threshold:
                continue

            detected.append(
                DetectedHeading(
                    text=self._normalize_heading_text(line.text),
                    page_number=line.page_number,
                    level=self._heading_level(line, body_size),
                    score=score,
                    line_index=index,
                    bbox=line.bbox,
                )
            )
        return detected

    def _lines_from_page(self, page: PDFPage) -> list[DocumentLine]:
        if not page.spans:
            page_text = page.text.strip()
            if not page_text:
                return []
            return [
                DocumentLine(
                    page_number=page.page_number,
                    text=page_text,
                    font_size=10.0,
                    font_name="",
                    is_bold=False,
                    bbox=(0.0, 0.0, 0.0, 0.0),
                    x0=0.0,
                    y0=0.0,
                    y1=0.0,
                )
            ]

        sorted_spans = sorted(
            page.spans,
            key=lambda span: (round(span.bbox[1], 1), span.bbox[0]),
        )
        groups: list[list[TextSpan]] = []
        current: list[TextSpan] = []

        for span in sorted_spans:
            if not current:
                current = [span]
                continue
            anchor_y = statistics.median([item.bbox[1] for item in current])
            if abs(span.bbox[1] - anchor_y) <= _LINE_Y_TOLERANCE:
                current.append(span)
            else:
                groups.append(current)
                current = [span]
        if current:
            groups.append(current)

        return [self._spans_to_line(page.page_number, group) for group in groups]

    def _spans_to_line(
        self, page_number: int, spans: Sequence[TextSpan]
    ) -> DocumentLine:
        ordered = sorted(spans, key=lambda span: span.bbox[0])
        text = self._join_span_texts(ordered)
        font_size = max(span.font_size for span in ordered)
        is_bold = any(span.is_bold for span in ordered)
        dominant = max(ordered, key=lambda span: len(span.text.strip()) or 0)
        x0 = min(span.bbox[0] for span in ordered)
        y0 = min(span.bbox[1] for span in ordered)
        x1 = max(span.bbox[2] for span in ordered)
        y1 = max(span.bbox[3] for span in ordered)
        return DocumentLine(
            page_number=page_number,
            text=text,
            font_size=font_size,
            font_name=dominant.font_name,
            is_bold=is_bold,
            bbox=(x0, y0, x1, y1),
            x0=x0,
            y0=y0,
            y1=y1,
        )

    def _join_span_texts(self, spans: Sequence[TextSpan]) -> str:
        parts: list[str] = []
        previous: TextSpan | None = None
        for span in spans:
            piece = span.text
            if previous is None:
                parts.append(piece)
                previous = span
                continue
            gap = span.bbox[0] - previous.bbox[2]
            if (
                gap > 2.0
                and not previous.text.endswith((" ", "-"))
                and not piece.startswith(" ")
            ):
                parts.append(" ")
            parts.append(piece)
            previous = span
        return "".join(parts).strip()

    def _estimate_body_font_size(self, lines: Sequence[DocumentLine]) -> float:
        if not lines:
            return 10.0
        sizes = [line.font_size for line in lines if line.text.strip()]
        if not sizes:
            return 10.0
        return float(statistics.median(sizes))

    def _score_heading_candidate(
        self,
        line: DocumentLine,
        previous: DocumentLine | None,
        body_font_size: float,
    ) -> float:
        text = line.text.strip()
        if not text or _SEPARATOR_RE.match(text):
            return 0.0
        if _LABEL_VALUE_RE.match(text) and not _NUMBERED_HEADING_RE.match(text):
            return 0.0

        score = 0.0
        size_delta = line.font_size - body_font_size

        if size_delta >= 2.5:
            score += 3.0
        elif size_delta >= 1.5:
            score += 2.5
        elif size_delta >= 0.8:
            score += 1.0

        if line.is_bold:
            score += 2.0 if size_delta >= 0 else 1.0

        if previous is not None and previous.page_number == line.page_number:
            gap = line.y0 - previous.y1
            line_height = max(line.y1 - line.y0, 1.0)
            if gap >= line_height * 1.6:
                score += 2.0
            elif gap >= line_height * 1.1:
                score += 1.0

        length = len(text)
        if length <= 40:
            score += 2.0
        elif length <= 60:
            score += 1.5
        elif length <= 90:
            score += 0.5
        else:
            score -= 1.5

        if _NUMBERED_HEADING_RE.match(text):
            score += 3.0

        letters = [ch for ch in text if ch.isalpha()]
        if letters:
            upper_ratio = sum(1 for ch in letters if ch.isupper()) / len(letters)
            if upper_ratio >= 0.85 and length <= 80:
                score += 2.0
            elif text.istitle() and length <= 70:
                score += 1.0

        if _COLON_ONLY_HEADING_RE.match(text) and line.is_bold:
            score += 2.0

        return score

    def _heading_level(self, line: DocumentLine, body_font_size: float) -> int:
        text = line.text.strip()
        if _SUB_NUMBERED_RE.match(text):
            return 2
        if _MAJOR_NUMBERED_RE.match(text) or line.font_size >= body_font_size + 1.5:
            return 1
        if _COLON_ONLY_HEADING_RE.match(text):
            return 2
        if line.is_bold and line.font_size <= body_font_size + 1.0:
            return 2
        return 1

    def _normalize_heading_text(self, text: str) -> str:
        cleaned = " ".join(text.strip().split())
        if cleaned.endswith(":"):
            cleaned = cleaned[:-1].rstrip()
        return cleaned

    def _build_sections_from_lines(
        self,
        *,
        document_id: str,
        lines: Sequence[DocumentLine],
        headings: Sequence[DetectedHeading],
        pages: Sequence[PDFPage],
    ) -> list[Section]:
        heading_by_index = {heading.line_index: heading for heading in headings}
        sections: list[Section] = []
        current_segments: list[TextSegment] = []
        current_heading: str | None = None
        current_parent: str | None = None
        last_level_one: str | None = None
        section_index = 0

        def flush() -> None:
            nonlocal section_index, current_segments
            if not current_segments and current_heading is None:
                return

            text = "\n".join(
                segment.text for segment in current_segments if segment.text.strip()
            ).strip()
            if not text and current_heading is None:
                current_segments = []
                return

            pages_used = [segment.page_number for segment in current_segments] or [1]
            sections.append(
                self._make_section(
                    document_id=document_id,
                    section_index=section_index,
                    heading=current_heading,
                    parent_heading=current_parent,
                    text=text,
                    start_page=min(pages_used),
                    end_page=max(pages_used),
                    segments=tuple(current_segments),
                )
            )
            section_index += 1
            current_segments = []

        # No reliable headings: fall back to page-level sections so cheap
        # CPT / BM25 narrowing can still operate without embedding the full body.
        if not headings:
            return self._page_level_sections(document_id=document_id, pages=pages)

        for index, line in enumerate(lines):
            heading = heading_by_index.get(index)
            if heading is not None:
                flush()
                current_heading = heading.text
                if heading.level == 1:
                    last_level_one = heading.text
                    current_parent = None
                else:
                    current_parent = last_level_one
                current_segments = [
                    TextSegment(page_number=line.page_number, text=heading.text)
                ]
                continue

            if _SEPARATOR_RE.match(line.text.strip()):
                continue

            current_segments.append(
                TextSegment(page_number=line.page_number, text=line.text)
            )

        flush()

        for page in pages:
            if not page.requires_ocr:
                continue
            logger.info(
                "ocr_required_page",
                document_id=document_id,
                page_number=page.page_number,
            )
            sections.append(
                self._make_section(
                    document_id=document_id,
                    section_index=section_index,
                    heading=None,
                    parent_heading=None,
                    text="",
                    start_page=page.page_number,
                    end_page=page.page_number,
                    requires_ocr=True,
                    segments=(),
                )
            )
            section_index += 1

        if not sections:
            return self._page_level_sections(document_id=document_id, pages=pages)

        return sections

    def _page_level_sections(
        self,
        *,
        document_id: str,
        pages: Sequence[PDFPage],
    ) -> list[Section]:
        """One section per page when layout yields no reliable heading structure."""
        sections: list[Section] = []
        section_index = 0
        for page in pages:
            if page.requires_ocr:
                sections.append(
                    self._make_section(
                        document_id=document_id,
                        section_index=section_index,
                        heading=None,
                        parent_heading=None,
                        text="",
                        start_page=page.page_number,
                        end_page=page.page_number,
                        requires_ocr=True,
                        segments=(),
                    )
                )
                section_index += 1
                continue

            text = page.text.strip()
            if not text:
                continue
            segments = (TextSegment(page_number=page.page_number, text=text),)
            sections.append(
                self._make_section(
                    document_id=document_id,
                    section_index=section_index,
                    heading=None,
                    parent_heading=None,
                    text=text,
                    start_page=page.page_number,
                    end_page=page.page_number,
                    segments=segments,
                )
            )
            section_index += 1

        if sections:
            logger.info(
                "page_level_sections_built",
                document_id=document_id,
                section_count=len(sections),
                reason="no_reliable_headings",
            )
        return sections

    def _make_section(
        self,
        *,
        document_id: str,
        section_index: int,
        heading: str | None,
        parent_heading: str | None,
        text: str,
        start_page: int,
        end_page: int,
        segments: tuple[TextSegment, ...] = (),
        requires_ocr: bool = False,
    ) -> Section:
        date_started = time.perf_counter()
        document_date, warnings = extract_document_date_with_warnings(heading, text)
        date_ms = int((time.perf_counter() - date_started) * 1000)
        metrics = get_request_metrics()
        if metrics is not None and get_preprocess_doc_kind() == "patient":
            metrics.add_substage_ms("patient_date_metadata_ms", date_ms)
        return Section(
            id=f"{document_id}:section:{section_index:04d}",
            document_id=document_id,
            heading=heading,
            parent_heading=parent_heading,
            text=text,
            start_page=start_page,
            end_page=end_page,
            token_count=count_tokens(text),
            requires_ocr=requires_ocr,
            segments=segments,
            document_date=document_date,
            temporal_warnings=tuple(warnings),
        )
