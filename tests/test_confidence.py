from ledgerlens.pipeline.confidence import apply_rules, decide, document_confidence, missing_fields
from ledgerlens.schemas import Decision, DocType, ExtractedField, RuleResult


def _fields(**confs: float) -> dict[str, ExtractedField]:
    return {name: ExtractedField(name=name, value="x", confidence=conf) for name, conf in confs.items()}


def test_passed_rules_relieve_doubt_and_failed_rules_halve() -> None:
    fields = _fields(total=0.8, tax=0.8)
    rules = [
        RuleResult(name="totals_reconcile", passed=True, fields=["total"]),
        RuleResult(name="tax_rate_plausible", passed=False, fields=["tax"]),
        RuleResult(name="iban_checksum", passed=None, fields=["total"]),
    ]
    adjusted = apply_rules(fields, rules)
    assert adjusted["total"].confidence == 0.9 and adjusted["tax"].confidence == 0.4
    assert fields["total"].confidence == 0.8


def test_document_confidence_penalises_missing_and_flagged_fields() -> None:
    full = _fields(
        vendor_name=1.0,
        customer_name=1.0,
        document_number=1.0,
        document_date=1.0,
        due_date=1.0,
        currency=1.0,
        subtotal=1.0,
        tax=1.0,
        total=1.0,
    )
    assert document_confidence(DocType.INVOICE, full, 0.9) == 1.0
    missing = dict(full)
    del missing["tax"]
    assert document_confidence(DocType.INVOICE, missing, 0.9) == 0.2
    extra = dict(full, iban=ExtractedField(name="iban", value="x", confidence=0.5))
    assert document_confidence(DocType.INVOICE, extra, 0.9) == 0.5
    assert document_confidence(DocType.INVOICE, full, 0.9, flagged=["gstin"]) == 0.2
    assert missing_fields(DocType.INVOICE, missing, ["gstin"]) == ["tax", "gstin"]


def test_decision_requires_threshold_and_no_hard_failure() -> None:
    assert decide(0.9, [], 0.5) is Decision.ACCEPTED
    assert decide(0.4, [], 0.5) is Decision.REVIEW
    hard = [RuleResult(name="iban_checksum", passed=False)]
    assert decide(0.99, hard, 0.5) is Decision.REVIEW
    soft = [RuleResult(name="tax_rate_plausible", passed=False)]
    assert decide(0.99, soft, 0.5) is Decision.ACCEPTED
