"""Quality gate: compare results.json with explicit thresholds and fail loudly."""

from __future__ import annotations

import json
from pathlib import Path

DEFAULT_THRESHOLDS: dict[str, float] = {
    "fields_micro_f1_min": 0.0,
    "line_items_f1_min": 0.0,
    "transactions_f1_min": 0.0,
    "acceptance_rate_min": 0.0,
    "wrong_accepted_rate_max": 1.0,
    "cer_seen_max": 1.0,
    "latency_p95_ms_max": 1e9,
    "doc_type_accuracy_min": 0.0,
}


def load_thresholds(path: Path) -> dict[str, float]:
    if not path.exists():
        return dict(DEFAULT_THRESHOLDS)
    return {**DEFAULT_THRESHOLDS, **json.loads(path.read_text())}


def _wrong_rate(test: dict[str, object]) -> float:
    """Share of all test documents that were accepted with at least one wrong header field."""
    acceptance = test["acceptance"]  # type: ignore[index]
    return round(acceptance["wrong_header_accepted"] / max(1, int(test["documents"])), 4)  # type: ignore[index, call-overload]


def evaluate_gate(results: dict[str, object], thresholds: dict[str, float]) -> list[str]:
    """Return a list of human-readable failures; empty means the gate passed."""
    test = results["test"]  # type: ignore[index]
    checks = [
        ("fields_micro_f1_min", test["fields"]["micro"]["f1"], True),  # type: ignore[index]
        ("line_items_f1_min", test["line_items"]["f1"], True),  # type: ignore[index]
        ("transactions_f1_min", test["transactions"]["f1"], True),  # type: ignore[index]
        ("acceptance_rate_min", test["acceptance"]["acceptance_rate"], True),  # type: ignore[index]
        ("wrong_accepted_rate_max", _wrong_rate(test), False),
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
