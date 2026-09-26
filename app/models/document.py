"""Document domain model."""

from enum import StrEnum


class DocumentType(StrEnum):
    """Supported document categories."""

    PATIENT_CHART = "PATIENT_CHART"
    POLICY = "POLICY"
