"""Request and response models for the HTTP API."""

from __future__ import annotations

from pydantic import BaseModel, Field

from ledgerlens.schemas import ExtractionResult


class DocumentResponse(BaseModel):
    """A processed document as stored."""

    id: str
    created_at: str
    filename: str | None
    status: str
    result: ExtractionResult


class QueueItem(BaseModel):
    """What a reviewer sees in the queue list."""

    id: str
    created_at: str
    filename: str | None
    doc_type: str
    confidence: float
    failed_rules: list[str]
    fields: dict[str, str | None]


class QueuePage(BaseModel):
    items: list[QueueItem]
    pending: int
    limit: int
    offset: int


class ReviewRequest(BaseModel):
    """Corrections submitted by a human. Only changed fields need to be listed."""

    reviewer: str | None = Field(default=None, max_length=120)
    fields: dict[str, str | None] = Field(default_factory=dict)
    line_items_corrected: int = Field(default=0, ge=0)
    note: str | None = Field(default=None, max_length=2000)


class ReviewResponse(BaseModel):
    document_id: str
    fields_total: int
    fields_corrected: int
    fields_confirmed: int
    effort_saved_fraction: float


class StatsResponse(BaseModel):
    """Service-wide counters and the measured human-effort saving."""

    documents: int
    accepted: int
    pending: int
    reviewed: int
    reviews: int
    fields_reviewed: int
    fields_corrected: int
    fields_auto_accepted: int
    effort_saved_fraction: float
    effort_model: dict[str, float]
    threshold: float
    ocr_engine: str


class HealthResponse(BaseModel):
    status: str
    version: str
    ocr_engine: str
    threshold: float
