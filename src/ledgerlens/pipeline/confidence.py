"""Field and document confidence, and the accept-or-review decision.

Field confidence starts as the OCR probability that every character of the value is
right, times a label-match factor. Rules then move it: a passed rule removes part of
the remaining doubt, a failed rule halves it. Document confidence is the product over
the fields a reviewer would have to check, with a penalty for every field the document
type should carry but the pipeline did not find, so a threshold on it trades
automation against error rate directly.
"""

from __future__ import annotations

from ledgerlens.pipeline.validate import HARD_RULES
from ledgerlens.schemas import Decision, DocType, ExtractedField, RuleResult

PASS_RELIEF = 0.5
FAIL_FACTOR = 0.5
MISSING_FACTOR = 0.2
EXPECTED_FIELDS: dict[DocType, tuple[str, ...]] = {
    DocType.INVOICE: (
        "vendor_name",
        "customer_name",
        "document_number",
        "document_date",
        "due_date",
        "currency",
        "subtotal",
        "tax",
        "total",
    ),
    DocType.RECEIPT: ("vendor_name", "document_date", "currency", "subtotal", "tax", "total"),
    DocType.BANK_STATEMENT: (
        "vendor_name",
        "customer_name",
        "document_number",
        "document_date",
        "period_start",
        "period_end",
        "currency",
        "opening_balance",
        "closing_balance",
    ),
}


def apply_rules(fields: dict[str, ExtractedField], rules: list[RuleResult]) -> dict[str, ExtractedField]:
    """Return fields with confidences adjusted by the rules that touched them."""
    adjusted = {name: field.model_copy() for name, field in fields.items()}
    for rule in rules:
        if rule.passed is None:
            continue
        for name in rule.fields:
            field = adjusted.get(name)
            if field is None:
                continue
            if rule.passed:
                field.confidence = round(1.0 - (1.0 - field.confidence) * PASS_RELIEF, 4)
            else:
                field.confidence = round(field.confidence * FAIL_FACTOR, 4)
    return adjusted


def document_confidence(
    doc_type: DocType, fields: dict[str, ExtractedField], doc_type_confidence: float, flagged: list[str] | None = None
) -> float:
    """Product of field confidences; every expected or labelled field that is missing costs MISSING_FACTOR."""
    expected = EXPECTED_FIELDS[doc_type]
    score = min(1.0, doc_type_confidence + 0.3)
    for name in expected:
        field = fields.get(name)
        score *= field.confidence if field is not None else MISSING_FACTOR
    for name, field in fields.items():
        if name not in expected:
            score *= field.confidence
    score *= MISSING_FACTOR ** len(flagged or [])
    return round(max(0.0, min(1.0, score)), 4)


def missing_fields(doc_type: DocType, fields: dict[str, ExtractedField], flagged: list[str] | None = None) -> list[str]:
    """Expected fields that were not extracted plus labelled ones that were not read, for the reviewer."""
    return [name for name in EXPECTED_FIELDS[doc_type] if name not in fields] + list(flagged or [])


def decide(confidence: float, rules: list[RuleResult], threshold: float) -> Decision:
    """Accept only above the threshold and with no failed hard rule."""
    if any(rule.passed is False and rule.name in HARD_RULES for rule in rules):
        return Decision.REVIEW
    return Decision.ACCEPTED if confidence >= threshold else Decision.REVIEW
