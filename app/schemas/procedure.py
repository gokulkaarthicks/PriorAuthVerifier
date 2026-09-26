"""Procedure-related API schemas."""

from pydantic import BaseModel, ConfigDict, Field


class ProcedureExtraction(BaseModel):
    """Structured planned-procedure extraction result."""

    model_config = ConfigDict(extra="forbid")

    procedure_name: str | None = None
    cpt_code: str | None = None
    status: str | None = None
    source_chunk_id: str | None = None
    source_text: str | None = None
    page_number: int | None = Field(default=None, ge=1)
    is_future_procedure: bool = False
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
