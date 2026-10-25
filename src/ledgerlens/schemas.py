"""Typed records shared by the generator, the pipeline, the benchmark and the API.

Every stage speaks the same vocabulary so that ground truth and predictions can be
compared field by field without ad hoc conversions. Money is carried as Decimal and
compared at cent precision; dates are real dates, never free text. Field names are
shared across document types (a bank name is a ``vendor_name``) so that the metric
code has one definition of a field instead of three.
"""

from __future__ import annotations

from datetime import date
from decimal import Decimal
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class DocType(StrEnum):
    """The three document families the generator produces and the pipeline reads."""

    INVOICE = "invoice"
    RECEIPT = "receipt"
    BANK_STATEMENT = "bank_statement"


class Severity(StrEnum):
    """Degradation level applied by the generator; reported per level in the benchmark."""

    CLEAN = "clean"
    LIGHT = "light"
    HEAVY = "heavy"


class LineItem(BaseModel):
    """One billed line: quantity times unit price must equal amount at cent precision."""

    model_config = ConfigDict(frozen=True)

    description: str
    quantity: Decimal
    unit_price: Decimal
    amount: Decimal


class Transaction(BaseModel):
    """One statement row. Exactly one of debit or credit is set; balance is the running total."""

    model_config = ConfigDict(frozen=True)

    date: date
    description: str
    debit: Decimal | None = None
    credit: Decimal | None = None
    balance: Decimal


class DocumentFields(BaseModel):
    """Header fields. ``None`` means the field does not exist on this document."""

    vendor_name: str | None = None
    customer_name: str | None = None
    document_number: str | None = None
    document_date: date | None = None
    due_date: date | None = None
    period_start: date | None = None
    period_end: date | None = None
    currency: str | None = None
    subtotal: Decimal | None = None
    tax: Decimal | None = None
    total: Decimal | None = None
    opening_balance: Decimal | None = None
    closing_balance: Decimal | None = None
    gstin: str | None = None
    iban: str | None = None


FIELD_NAMES: tuple[str, ...] = tuple(DocumentFields.model_fields)

MONEY_FIELDS: frozenset[str] = frozenset({"subtotal", "tax", "total", "opening_balance", "closing_balance"})
DATE_FIELDS: frozenset[str] = frozenset({"document_date", "due_date", "period_start", "period_end"})
IDENTIFIER_FIELDS: frozenset[str] = frozenset({"gstin", "iban"})


class WordBox(BaseModel):
    """A rendered word and its ink bounding box in the undistorted page frame."""

    model_config = ConfigDict(frozen=True)

    text: str
    x0: int
    y0: int
    x1: int
    y1: int
    line_id: int


class Degradation(BaseModel):
    """The exact distortion parameters applied to one page, kept for error analysis."""

    rotation_deg: float = 0.0
    blur_sigma: float = 0.0
    motion_px: int = 0
    noise_sigma: float = 0.0
    salt_pepper: float = 0.0
    illumination: float = 0.0
    stamp: bool = False
    jpeg_quality: int | None = None
    paper_tint: int = 255


class GroundTruth(BaseModel):
    """Everything the generator knows about one page, written next to its PNG."""

    doc_id: str
    doc_type: DocType
    template: str
    font: str
    font_seen: bool
    severity: Severity
    seed: int
    width: int
    height: int
    fields: DocumentFields
    line_items: list[LineItem] = Field(default_factory=list)
    transactions: list[Transaction] = Field(default_factory=list)
    words: list[WordBox] = Field(default_factory=list)
    degradation: Degradation = Field(default_factory=Degradation)


class ExtractedField(BaseModel):
    """A predicted header field in canonical string form with its confidence and evidence."""

    name: str
    value: str | None
    confidence: float = Field(ge=0.0, le=1.0)
    evidence: str = ""


class RuleResult(BaseModel):
    """Outcome of one validation rule. ``passed`` is None when the rule did not apply."""

    name: str
    passed: bool | None
    detail: str = ""
    fields: list[str] = Field(default_factory=list)


class Decision(StrEnum):
    """Routing outcome: straight through, or into the human review queue."""

    ACCEPTED = "accepted"
    REVIEW = "review"


class ExtractionResult(BaseModel):
    """The pipeline output for one page."""

    doc_type: DocType
    doc_type_confidence: float
    fields: list[ExtractedField]
    line_items: list[LineItem] = Field(default_factory=list)
    transactions: list[Transaction] = Field(default_factory=list)
    rules: list[RuleResult] = Field(default_factory=list)
    document_confidence: float
    missing_fields: list[str] = Field(default_factory=list)
    decision: Decision
    ocr_engine: str
    skew_deg: float = 0.0
    latency_ms: float = 0.0
    page_text: str = ""

    def field(self, name: str) -> ExtractedField | None:
        """Return the prediction for ``name`` or None when the field was not emitted."""
        for item in self.fields:
            if item.name == name:
                return item
        return None

    def field_map(self) -> dict[str, str | None]:
        """Canonical values keyed by field name, for metric code and the API."""
        return {item.name: item.value for item in self.fields}
