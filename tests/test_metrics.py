from datetime import date
from decimal import Decimal

from ledgerlens.bench.metrics import (
    Counts,
    DocScore,
    EffortModel,
    acceptance_summary,
    canonical_quantity,
    choose_threshold,
    effort_summary,
    latency_summary,
    line_cer,
    match_line_items,
    match_transactions,
    score_fields,
)
from ledgerlens.pipeline.ocr.base import OCRWord
from ledgerlens.schemas import LineItem, Transaction, WordBox


def test_field_scoring_follows_exact_match_convention() -> None:
    counts = score_fields(
        {"total": "1.00", "tax": "2.00", "iban": None, "gstin": None},
        {"total": "1.00", "tax": "3.00", "iban": "GB1", "gstin": None},
    )
    assert counts["total"].tp == 1
    assert counts["tax"].fp == 1 and counts["tax"].fn == 1
    assert counts["iban"].fp == 1 and counts["iban"].fn == 0
    assert "gstin" not in counts


def test_counts_arithmetic() -> None:
    counts = Counts(tp=3, fp=1, fn=2)
    assert counts.precision == 0.75 and counts.recall == 0.6 and round(counts.f1, 4) == 0.6667
    assert Counts().f1 == 0.0


def test_row_matching_is_multiset_exact() -> None:
    gold = [
        LineItem(description="Steel bracket", quantity=Decimal(40), unit_price=Decimal("1.00"), amount=Decimal("40.00"))
    ] * 2
    pred = [
        LineItem(
            description="steel  bracket", quantity=Decimal("40.0"), unit_price=Decimal("1.00"), amount=Decimal("40.00")
        ),
        LineItem(description="other", quantity=Decimal(1), unit_price=Decimal("1.00"), amount=Decimal("1.00")),
    ]
    counts = match_line_items(gold, pred)
    assert (counts.tp, counts.fp, counts.fn) == (1, 1, 1)
    assert canonical_quantity(Decimal("40.0")) == "40" and canonical_quantity(Decimal("2.50")) == "2.5"
    rows = [Transaction(date=date(2026, 3, 1), description="x", debit=Decimal("1.00"), balance=Decimal("9.00"))]
    assert match_transactions(rows, rows).tp == 1


def test_line_cer_uses_words_inside_each_line() -> None:
    truth = [
        WordBox(text="Total", x0=10, y0=10, x1=60, y1=30, line_id=0),
        WordBox(text="12.50", x0=70, y0=10, x1=120, y1=30, line_id=0),
    ]
    words = [OCRWord("Total", 0.9, 10, 10, 60, 30), OCRWord("12.5O", 0.9, 70, 10, 120, 30)]
    assert line_cer(truth, words) == (1, 11)
    assert line_cer(truth, []) == (11, 11)


def _score(confidence: float, correct: bool, accepted: bool = True) -> DocScore:
    fields = {"total": Counts(tp=1) if correct else Counts(fp=1, fn=1), "tax": Counts(tp=1)}
    return DocScore(
        "d", "invoice", "t", "clean", True, True, fields, Counts(), Counts(), confidence, accepted, 500.0, 0, 10
    )


def test_acceptance_effort_and_threshold_selection() -> None:
    scores = [_score(0.95, True), _score(0.9, True), _score(0.6, False), _score(0.3, False, accepted=False)]
    summary = acceptance_summary(scores, 0.8)
    assert summary["accepted"] == 2 and summary["accepted_record_accuracy"] == 1.0
    assert acceptance_summary(scores, 0.5)["wrong_accepted"] == 1
    effort = effort_summary(scores, 0.8, EffortModel())
    assert effort["manual_cost"] == 8.0 and effort["assisted_cost"] == 2 * (1.0 + 0.2)
    assert choose_threshold(scores, 0.98) == 0.61
    assert choose_threshold([_score(0.9, False)], 0.98) == 1.0
    assert latency_summary([100.0, 200.0, 300.0])["p50"] == 200.0
