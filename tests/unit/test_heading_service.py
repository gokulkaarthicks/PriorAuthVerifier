"""Unit tests for HeadingService."""

from __future__ import annotations

from app.models.document import DocumentType
from app.models.layout import DocumentLine
from app.models.pdf import ParsedDocument, PDFPage, TextSpan
from app.services.heading_service import HeadingService


def _span(
    text: str,
    *,
    size: float,
    bold: bool,
    y: float,
    x: float = 50.0,
    font: str | None = None,
) -> TextSpan:
    width = max(len(text) * size * 0.5, 10.0)
    font_name = font or ("Helvetica-Bold" if bold else "Helvetica")
    return TextSpan(
        text=text,
        font_size=size,
        font_name=font_name,
        is_bold=bold,
        bbox=(x, y, x + width, y + size),
    )


def _page(page_number: int, spans: list[TextSpan], text: str = "") -> PDFPage:
    joined = text or "\n".join(span.text for span in spans)
    return PDFPage(
        page_number=page_number,
        text=joined,
        spans=spans,
        requires_ocr=False,
    )


def _document(pages: list[PDFPage]) -> ParsedDocument:
    return ParsedDocument(
        id="doc-1",
        filename="sample.pdf",
        document_type=DocumentType.PATIENT_CHART,
        page_count=len(pages),
        pages=pages,
    )


def test_larger_bold_text_becomes_heading() -> None:
    service = HeadingService()
    page = _page(
        1,
        [
            _span("1. MAJOR SECTION", size=14.0, bold=True, y=40.0),
            _span("Body paragraph explaining details.", size=10.0, bold=False, y=70.0),
        ],
    )
    sections = service.build_sections(_document([page]))
    headings = [section.heading for section in sections if section.heading]

    assert "1. MAJOR SECTION" in headings
    assert all(heading != "Body paragraph explaining details." for heading in headings)


def test_normal_body_text_is_not_heading() -> None:
    service = HeadingService()
    lines = [
        DocumentLine(
            page_number=1,
            text="The patient continues current therapy without changes.",
            font_size=10.0,
            font_name="Helvetica",
            is_bold=False,
            bbox=(50.0, 100.0, 400.0, 112.0),
            x0=50.0,
            y0=100.0,
            y1=112.0,
        )
    ]
    headings = service.detect_headings(lines, body_font_size=10.0)
    assert headings == []


def test_multiple_headings_on_one_page() -> None:
    service = HeadingService()
    page = _page(
        1,
        [
            _span("6. VISIT NOTES", size=13.0, bold=True, y=40.0),
            _span("Chief Complaint:", size=10.0, bold=True, y=70.0),
            _span("Referral for evaluation.", size=10.0, bold=False, y=90.0),
            _span("History of Present Illness:", size=10.0, bold=True, y=120.0),
            _span("Symptoms began last month.", size=10.0, bold=False, y=140.0),
            _span("Physical Examination:", size=10.0, bold=True, y=170.0),
            _span("Exam findings are listed here.", size=10.0, bold=False, y=190.0),
            _span("Assessment & Plan:", size=10.0, bold=True, y=220.0),
            _span("Continue monitoring.", size=10.0, bold=False, y=240.0),
        ],
    )
    sections = service.build_sections(_document([page]))
    headings = [section.heading for section in sections if section.heading]

    assert "6. VISIT NOTES" in headings
    assert "Chief Complaint" in headings
    assert "History of Present Illness" in headings
    assert "Physical Examination" in headings
    assert "Assessment & Plan" in headings


def test_numbered_headings_work() -> None:
    service = HeadingService()
    page = _page(
        1,
        [
            _span("SECTION 1: DEFINITIONS", size=12.0, bold=False, y=40.0),
            _span(
                "Definitions body content appears here.",
                size=12.0,
                bold=False,
                y=70.0,
            ),
            _span("2.1 EMERGENCY TRANSPORT", size=12.0, bold=False, y=110.0),
            _span(
                "Transport rules continue in this paragraph.",
                size=12.0,
                bold=False,
                y=140.0,
            ),
        ],
    )
    sections = service.build_sections(_document([page]))
    headings = [section.heading for section in sections if section.heading]

    assert any(heading and heading.startswith("SECTION 1") for heading in headings)
    assert any(heading and heading.startswith("2.1") for heading in headings)


def test_bold_subheadings_work_with_parent() -> None:
    service = HeadingService()
    page = _page(
        1,
        [
            _span("8. PLANNED PROCEDURE", size=13.0, bold=True, y=40.0),
            _span("Procedure overview content.", size=10.0, bold=False, y=70.0),
            _span("Clinical Indication Summary:", size=11.0, bold=True, y=110.0),
            _span("Indication details are listed.", size=10.0, bold=False, y=140.0),
        ],
    )
    sections = service.build_sections(_document([page]))
    by_heading = {section.heading: section for section in sections if section.heading}

    assert "8. PLANNED PROCEDURE" in by_heading
    assert "Clinical Indication Summary" in by_heading
    assert by_heading["Clinical Indication Summary"].parent_heading == (
        "8. PLANNED PROCEDURE"
    )


def test_text_without_headings_is_preserved() -> None:
    service = HeadingService()
    page = _page(
        1,
        [
            _span(
                "Plain paragraph one with enough text.",
                size=10.0,
                bold=False,
                y=40.0,
            ),
            _span(
                "Plain paragraph two continues the story.",
                size=10.0,
                bold=False,
                y=70.0,
            ),
        ],
    )
    sections = service.build_sections(_document([page]))

    assert len(sections) == 1
    assert sections[0].heading is None
    assert "Plain paragraph one" in sections[0].text
    assert "Plain paragraph two" in sections[0].text


def test_no_specific_medical_heading_is_hardcoded() -> None:
    with open("app/services/heading_service.py", encoding="utf-8") as handle:
        source = handle.read()
    forbidden = [
        "PLANNED PROCEDURE",
        "VISIT NOTES",
        "Chief Complaint",
        "Assessment & Plan",
        "Position Statement",
    ]
    for term in forbidden:
        assert term not in source
