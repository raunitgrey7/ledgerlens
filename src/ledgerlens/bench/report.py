"""Aggregate document scores into the results dictionary and its Markdown rendering."""

from __future__ import annotations

from collections import defaultdict

from ledgerlens.bench.metrics import (
    Counts,
    DocScore,
    EffortModel,
    acceptance_summary,
    effort_summary,
    latency_summary,
)

CURVE_THRESHOLDS: tuple[float, ...] = (0.3, 0.5, 0.7, 0.8, 0.9, 0.95)


def _micro(scores: list[DocScore]) -> Counts:
    total = Counts()
    for score in scores:
        for counts in score.fields.values():
            total.add(counts)
    return total


def _rows_micro(scores: list[DocScore], attribute: str) -> Counts:
    total = Counts()
    for score in scores:
        total.add(getattr(score, attribute))
    return total


def _group(scores: list[DocScore], key: str) -> dict[str, list[DocScore]]:
    groups: dict[str, list[DocScore]] = defaultdict(list)
    for score in scores:
        groups[str(getattr(score, key))].append(score)
    return dict(sorted(groups.items()))


def _cer(scores: list[DocScore]) -> float:
    chars = sum(s.cer_chars for s in scores)
    return round(sum(s.cer_errors for s in scores) / chars, 4) if chars else 0.0


def summarise(scores: list[DocScore], threshold: float, effort: EffortModel) -> dict[str, object]:
    """The full metric block for one split at one threshold."""
    per_field: dict[str, Counts] = defaultdict(Counts)
    for score in scores:
        for name, counts in score.fields.items():
            per_field[name].add(counts)
    by_template = {
        name: {
            **_micro(group).as_dict(),
            "documents": len(group),
            "line_items_f1": round(_rows_micro(group, "items").f1, 4),
            "transactions_f1": round(_rows_micro(group, "transactions").f1, 4),
        }
        for name, group in _group(scores, "template").items()
    }
    by_severity = {
        name: {
            **_micro(group).as_dict(),
            "documents": len(group),
            "cer": _cer(group),
            "record_accuracy": round(sum(s.record_correct for s in group) / len(group), 4),
        }
        for name, group in _group(scores, "severity").items()
    }
    by_font = {
        ("seen" if name == "True" else "unseen"): {
            **_micro(group).as_dict(),
            "documents": len(group),
            "cer": _cer(group),
        }
        for name, group in _group(scores, "font_seen").items()
    }
    return {
        "documents": len(scores),
        "doc_type_accuracy": round(sum(s.type_correct for s in scores) / len(scores), 4),
        "fields": {
            "micro": _micro(scores).as_dict(),
            "per_field": {name: counts.as_dict() for name, counts in sorted(per_field.items())},
            "per_template": by_template,
            "per_severity": by_severity,
            "per_font": by_font,
        },
        "line_items": _rows_micro(scores, "items").as_dict(),
        "transactions": _rows_micro(scores, "transactions").as_dict(),
        "ocr_cer": {"all": _cer(scores), **{k: v["cer"] for k, v in by_font.items()}},
        "record_accuracy": round(sum(s.record_correct for s in scores) / len(scores), 4),
        "header_accuracy": round(sum(s.header_correct for s in scores) / len(scores), 4),
        "acceptance": acceptance_summary(scores, threshold),
        "acceptance_curve": [acceptance_summary(scores, t) for t in CURVE_THRESHOLDS],
        "effort": effort_summary(scores, threshold, effort),
        "latency_ms": latency_summary([s.latency_ms for s in scores]),
    }


def _pct(value: object) -> str:
    return f"{float(value) * 100:.1f}%"


def markdown_table(results: dict[str, object]) -> str:
    """The headline table used in README.md and RESULTS.md."""
    test = results["test"]  # type: ignore[index]
    fields = test["fields"]  # type: ignore[index]
    lines = [
        "| Metric | Value |",
        "|---|---|",
        f"| Documents scored (test split) | {test['documents']} |",
        f"| Document type accuracy | {_pct(test['doc_type_accuracy'])} |",
        f"| Field precision / recall / F1 (micro, exact match) | {_pct(fields['micro']['precision'])} / "
        f"{_pct(fields['micro']['recall'])} / {_pct(fields['micro']['f1'])} |",
        f"| Line-item F1 (exact description, quantity, amount) | {_pct(test['line_items']['f1'])} |",
        f"| Transaction F1 (exact row) | {_pct(test['transactions']['f1'])} |",
        f"| OCR character error rate, seen fonts / unseen fonts | {_pct(test['ocr_cer'].get('seen', 0.0))} / "
        f"{_pct(test['ocr_cer'].get('unseen', 0.0))} |",
        f"| Fully correct records (all fields and rows) | {_pct(test['record_accuracy'])} |",
        f"| Acceptance threshold (chosen on calibration split) | {test['acceptance']['threshold']} |",
        f"| Accepted without review | {_pct(test['acceptance']['acceptance_rate'])} |",
        f"| Accepted records fully correct | {_pct(test['acceptance']['accepted_record_accuracy'])} |",
        f"| Human effort saved (cost model in docs/EVALUATION.md) | {_pct(test['effort']['effort_saved_fraction'])} |",
        f"| Latency per page, mean / p50 / p95 (ms) | {test['latency_ms']['mean']} / {test['latency_ms']['p50']} / "
        f"{test['latency_ms']['p95']} |",
    ]
    return "\n".join(lines)


def markdown_breakdowns(results: dict[str, object]) -> str:
    """Per-field, per-template and per-severity tables."""
    test = results["test"]  # type: ignore[index]
    fields = test["fields"]  # type: ignore[index]
    out = ["### Per field", "", "| Field | Precision | Recall | F1 | Support |", "|---|---|---|---|---|"]
    for name, counts in fields["per_field"].items():
        out.append(
            f"| {name} | {_pct(counts['precision'])} | {_pct(counts['recall'])} | {_pct(counts['f1'])} | "
            f"{counts['tp'] + counts['fn']} |"
        )
    out += [
        "",
        "### Per template",
        "",
        "| Template | Docs | Field F1 | Line-item F1 | Transaction F1 |",
        "|---|---|---|---|---|",
    ]
    for name, block in fields["per_template"].items():
        out.append(
            f"| {name} | {block['documents']} | {_pct(block['f1'])} | {_pct(block['line_items_f1'])} | "
            f"{_pct(block['transactions_f1'])} |"
        )
    out += [
        "",
        "### Per severity",
        "",
        "| Severity | Docs | Field F1 | OCR CER | Fully correct records |",
        "|---|---|---|---|---|",
    ]
    for name, block in fields["per_severity"].items():
        out.append(
            f"| {name} | {block['documents']} | {_pct(block['f1'])} | {_pct(block['cer'])} | "
            f"{_pct(block['record_accuracy'])} |"
        )
    out += ["", "### Seen versus unseen fonts", "", "| Fonts | Docs | Field F1 | OCR CER |", "|---|---|---|---|"]
    for name, block in fields["per_font"].items():
        out.append(f"| {name} | {block['documents']} | {_pct(block['f1'])} | {_pct(block['cer'])} |")
    out += [
        "",
        "### Acceptance at other thresholds (test split)",
        "",
        "| Threshold | Accepted | Accepted records fully correct | Wrong acceptances |",
        "|---|---|---|---|",
    ]
    for point in test.get("acceptance_curve", []):
        out.append(
            f"| {point['threshold']} | {_pct(point['acceptance_rate'])} | "
            f"{_pct(point['accepted_record_accuracy'])} | {point['wrong_accepted']} |"
        )
    return "\n".join(out)
