"""Field and document confidence, and the accept-or-review decision.

Field confidence starts as the OCR probability that every character of the value is
right, times a label-match factor. Rules then move it: a passed rule removes part of
the remaining doubt, a failed rule halves it. Document confidence is the product over
the fields a reviewer would have to check, so it reads as "probability the record is
fully correct" and a threshold on it trades automation against error rate directly.
"""

from __future__ import annotations

from ledgerlens.pipeline.validate import HARD_RULES
from ledgerlens.schemas import Decision, DocType, ExtractedField, RuleResult

PASS_RELIEF = 0.5
FAIL_FACTOR = 0.5
MISSING_FACTOR = 0.35
REQUIRED_FIELDS: dict[DocType, tuple[str, ...]] = {
    DocType.INVOICE: ("vendor_name", "document_number", "document_date", "subtotal", "tax", "total"),
    DocType.RECEIPT: ("vendor_name", "document_date", "subtotal", "tax", "total"),
    DocType.BANK_STATEMENT: ("vendor_name", "period_start", "period_end", "opening_balance", "closing_balance"),
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


def document_confidence(doc_type: DocType, fields: dict[str, ExtractedField], doc_type_confidence: float) -> float:
    """Product of field confidences with a penalty per missing required field."""
    score = min(1.0, doc_type_confidence + 0.3)
    for name in REQUIRED_FIELDS[doc_type]:
        field = fields.get(name)
        score *= field.confidence if field is not None else MISSING_FACTOR
    for name, field in fields.items():
        if name not in REQUIRED_FIELDS[doc_type]:
            score *= field.confidence
    return round(max(0.0, min(1.0, score)), 4)


def decide(confidence: float, rules: list[RuleResult], threshold: float) -> Decision:
    """Accept only above the threshold and with no failed hard rule."""
    if any(rule.passed is False and rule.name in HARD_RULES for rule in rules):
        return Decision.REVIEW
    return Decision.ACCEPTED if confidence >= threshold else Decision.REVIEW
