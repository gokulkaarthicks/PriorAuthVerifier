"""Layout analysis models used during heading detection."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class DocumentLine:
    """A reconstructed visual line from PDF spans."""

    page_number: int
    text: str
    font_size: float
    font_name: str
    is_bold: bool
    bbox: tuple[float, float, float, float]
    x0: float
    y0: float
    y1: float


@dataclass(frozen=True, slots=True)
class DetectedHeading:
    """A heading candidate with score and hierarchy level."""

    text: str
    page_number: int
    level: int
    score: float
    line_index: int
    bbox: tuple[float, float, float, float]
