"""Metric definitions used by the benchmark and documented in docs/EVALUATION.md.

Field scoring follows the SROIE convention: a predicted value counts only on exact
match of its canonical form; a wrong value is both a false positive and a false
negative. Line items and transactions are matched as multisets of exact tuples.
Acceptance metrics treat a document as correct only when every header field and
every row is right, because that is what "no human review" has to mean.
"""

from __future__ import annotations

import difflib
from collections import Counter
from dataclasses import dataclass, field
from decimal import Decimal

import numpy as np

from ledgerlens.canonical import canonical_description
from ledgerlens.pipeline.ocr.base import OCRWord
from ledgerlens.schemas import LineItem, Transaction, WordBox


@dataclass
class Counts:
    """True positives, false positives, false negatives."""

    tp: int = 0
    fp: int = 0
    fn: int = 0

    def add(self, other: Counts) -> None:
        self.tp += other.tp
        self.fp += other.fp
        self.fn += other.fn

    @property
    def precision(self) -> float:
        return self.tp / (self.tp + self.fp) if self.tp + self.fp else 0.0

    @property
    def recall(self) -> float:
        return self.tp / (self.tp + self.fn) if self.tp + self.fn else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if p + r else 0.0

    def as_dict(self) -> dict[str, float | int]:
        return {
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1": round(self.f1, 4),
            "tp": self.tp,
            "fp": self.fp,
            "fn": self.fn,
        }


def score_fields(gold: dict[str, str | None], pred: dict[str, str | None]) -> dict[str, Counts]:
    """Per-field counts; fields absent on both sides are ignored."""
    counts: dict[str, Counts] = {}
    for name, expected in gold.items():
        predicted = pred.get(name)
        if expected is None and predicted is None:
            continue
        counts[name] = Counts()
        if expected is not None and predicted == expected:
            counts[name].tp = 1
        else:
            if predicted is not None:
                counts[name].fp = 1
            if expected is not None:
                counts[name].fn = 1
    return counts


def canonical_quantity(value: Decimal) -> str:
    """Plain decimal text without exponent or trailing zeros: 40, 2.5."""
    return format(value.normalize(), "f")


def _item_key(item: LineItem) -> tuple[str, str, str]:
    return (
        canonical_description(item.description),
        canonical_quantity(item.quantity),
        str(item.amount.quantize(Decimal("0.01"))),
    )


def _transaction_key(row: Transaction) -> tuple[str, str, str, str, str]:
    return (
        row.date.isoformat(),
        canonical_description(row.description),
        str(row.debit or ""),
        str(row.credit or ""),
        str(row.balance),
    )


def match_rows(gold_keys: list[tuple[str, ...]], pred_keys: list[tuple[str, ...]]) -> Counts:
    """Multiset intersection: each gold tuple can be claimed by one prediction."""
    matched = sum((Counter(gold_keys) & Counter(pred_keys)).values())
    return Counts(tp=matched, fp=len(pred_keys) - matched, fn=len(gold_keys) - matched)


def match_line_items(gold: list[LineItem], pred: list[LineItem]) -> Counts:
    return match_rows([_item_key(i) for i in gold], [_item_key(i) for i in pred])


def match_transactions(gold: list[Transaction], pred: list[Transaction]) -> Counts:
    return match_rows([_transaction_key(r) for r in gold], [_transaction_key(r) for r in pred])


def edit_distance(a: str, b: str) -> int:
    """Levenshtein distance via difflib opcodes (adequate for line-length strings)."""
    matcher = difflib.SequenceMatcher(a=a, b=b, autojunk=False)
    return sum(max(i2 - i1, j2 - j1) for tag, i1, i2, j1, j2 in matcher.get_opcodes() if tag != "equal")


def line_cer(truth_words: list[WordBox], ocr_words: list[OCRWord]) -> tuple[int, int]:
    """(errors, characters) over ground-truth lines; predicted words are those inside each line's box."""
    lines: dict[int, list[WordBox]] = {}
    for word in truth_words:
        lines.setdefault(word.line_id, []).append(word)
    errors = chars = 0
    for words in lines.values():
        x0, y0 = min(w.x0 for w in words), min(w.y0 for w in words)
        x1, y1 = max(w.x1 for w in words), max(w.y1 for w in words)
        expected = " ".join(w.text for w in words)
        inside = sorted(
            (d for d in ocr_words if x0 - 4 <= d.cx <= x1 + 4 and y0 - 3 <= d.cy <= y1 + 3), key=lambda d: d.x0
        )
        errors += edit_distance(" ".join(d.text for d in inside), expected)
        chars += len(expected)
    return errors, chars


@dataclass
class EffortModel:
    """Keystroke-equivalent cost of handling one document by hand or after extraction.

    Manual entry costs one unit per field. A queued document costs ``verify`` per
    correct prefilled field and ``correct`` per wrong or missing one. An accepted
    document costs nothing, which is why wrong acceptances are reported separately.
    """

    verify: float = 0.2
    correct: float = 1.0
    manual: float = 1.0

    def as_dict(self) -> dict[str, float]:
        return {
            "verify_cost_per_field": self.verify,
            "correct_cost_per_field": self.correct,
            "manual_cost_per_field": self.manual,
        }


@dataclass
class DocScore:
    """Everything the summary needs about one scored document."""

    doc_id: str
    doc_type: str
    template: str
    severity: str
    font_seen: bool
    type_correct: bool
    fields: dict[str, Counts]
    items: Counts
    transactions: Counts
    confidence: float
    accepted: bool
    latency_ms: float
    cer_errors: int
    cer_chars: int
    predicted_fields: dict[str, str | None] = field(default_factory=dict)

    @property
    def header_correct(self) -> bool:
        return all(c.fp == 0 and c.fn == 0 for c in self.fields.values())

    @property
    def record_correct(self) -> bool:
        return (
            self.header_correct
            and self.items.fp == 0
            and self.items.fn == 0
            and self.transactions.fp == 0
            and self.transactions.fn == 0
        )

    @property
    def gold_field_count(self) -> int:
        return sum(1 for c in self.fields.values() if c.tp or c.fn)

    @property
    def wrong_field_count(self) -> int:
        return sum(1 for c in self.fields.values() if c.fp or c.fn)


def latency_summary(values: list[float]) -> dict[str, float]:
    array = np.array(values, dtype=float)
    if array.size == 0:
        return {"mean": 0.0, "p50": 0.0, "p95": 0.0, "max": 0.0}
    return {
        "mean": round(float(array.mean()), 1),
        "p50": round(float(np.percentile(array, 50)), 1),
        "p95": round(float(np.percentile(array, 95)), 1),
        "max": round(float(array.max()), 1),
    }


def acceptance_summary(scores: list[DocScore], threshold: float) -> dict[str, float | int]:
    """Automation rate and the accuracy of what was automated at a threshold."""
    accepted = [s for s in scores if s.accepted and s.confidence >= threshold]
    total = len(scores)
    correct = sum(1 for s in accepted if s.record_correct)
    return {
        "threshold": round(threshold, 4),
        "accepted": len(accepted),
        "acceptance_rate": round(len(accepted) / total, 4) if total else 0.0,
        "accepted_record_accuracy": round(correct / len(accepted), 4) if accepted else 0.0,
        "accepted_header_accuracy": round(sum(1 for s in accepted if s.header_correct) / len(accepted), 4)
        if accepted
        else 0.0,
        "wrong_accepted": len(accepted) - correct,
        "review_rate": round(1 - len(accepted) / total, 4) if total else 0.0,
    }


def effort_summary(scores: list[DocScore], threshold: float, model: EffortModel) -> dict[str, float | int]:
    """Fraction of manual field-entry effort removed under the cost model."""
    manual = sum(s.gold_field_count for s in scores) * model.manual
    cost = 0.0
    for score in scores:
        if score.accepted and score.confidence >= threshold:
            continue
        wrong = score.wrong_field_count
        cost += wrong * model.correct + (score.gold_field_count - min(wrong, score.gold_field_count)) * model.verify
    return {
        "manual_cost": round(manual, 1),
        "assisted_cost": round(cost, 1),
        "effort_saved_fraction": round(1 - cost / manual, 4) if manual else 0.0,
        **model.as_dict(),
    }


def choose_threshold(scores: list[DocScore], target_accuracy: float, grid_step: float = 0.01) -> float:
    """Lowest threshold whose accepted-record accuracy on the calibration split meets the target.

    Returns 1.0 (accept nothing) when no threshold reaches the target.
    """
    best = 1.0
    for step in range(int(1 / grid_step), -1, -1):
        threshold = round(step * grid_step, 2)
        summary = acceptance_summary(scores, threshold)
        if summary["accepted"] and summary["accepted_record_accuracy"] >= target_accuracy:
            best = threshold
        elif summary["accepted"]:
            break
    return best
