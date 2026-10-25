"""Quality gate: compare results.json with explicit thresholds and fail loudly."""

from __future__ import annotations

import json
from pathlib import Path

DEFAULT_THRESHOLDS: dict[str, float] = {
    "fields_micro_f1_min": 0.0,
    "line_items_f1_min": 0.0,
    "transactions_f1_min": 0.0,
    "accepted_record_accuracy_min": 0.0,
    "cer_seen_max": 1.0,
    "latency_p95_ms_max": 1e9,
    "doc_type_accuracy_min": 0.0,
}


def load_thresholds(path: Path) -> dict[str, float]:
    if not path.exists():
        return dict(DEFAULT_THRESHOLDS)
    return {**DEFAULT_THRESHOLDS, **json.loads(path.read_text())}


def evaluate_gate(results: dict[str, object], thresholds: dict[str, float]) -> list[str]:
    """Return a list of human-readable failures; empty means the gate passed."""
    test = results["test"]  # type: ignore[index]
    checks = [
        ("fields_micro_f1_min", test["fields"]["micro"]["f1"], True),  # type: ignore[index]
        ("line_items_f1_min", test["line_items"]["f1"], True),  # type: ignore[index]
        ("transactions_f1_min", test["transactions"]["f1"], True),  # type: ignore[index]
        ("accepted_record_accuracy_min", test["acceptance"]["accepted_record_accuracy"], True),  # type: ignore[index]
        ("doc_type_accuracy_min", test["doc_type_accuracy"], True),  # type: ignore[index]
        ("cer_seen_max", test["ocr_cer"].get("seen", 0.0), False),  # type: ignore[index]
        ("latency_p95_ms_max", test["latency_ms"]["p95"], False),  # type: ignore[index]
    ]
    failures = []
    for name, value, is_minimum in checks:
        limit = thresholds[name]
        if (is_minimum and value < limit) or (not is_minimum and value > limit):
            failures.append(f"{name}: measured {value} violates limit {limit}")
    return failures
