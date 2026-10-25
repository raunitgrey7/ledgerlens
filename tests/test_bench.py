import json
from pathlib import Path

from ledgerlens.bench.gate import evaluate_gate, load_thresholds
from ledgerlens.bench.metrics import EffortModel
from ledgerlens.bench.report import markdown_breakdowns, markdown_table, summarise
from ledgerlens.bench.runner import run_split
from ledgerlens.generator.dataset import SplitSpec, build_split
from tests.conftest import ATLAS_DIR


def test_runner_and_report_on_a_tiny_split(tmp_path: Path) -> None:
    manifest = build_split(tmp_path, SplitSpec("tiny", 3, 17, severity_weights=(1.0, 0.0, 0.0), unseen_fraction=0.0))
    scores = run_split(tmp_path / "tiny", manifest.doc_ids, "atlas", ATLAS_DIR, 0.5, jobs=1)
    assert len(scores) == 3 and all(s.latency_ms > 0 for s in scores)
    summary = summarise(scores, 0.5, EffortModel())
    assert summary["documents"] == 3 and 0.0 <= summary["fields"]["micro"]["f1"] <= 1.0
    results = {"test": summary, "calibration": summary, "threshold": {"value": 0.5}}
    table = markdown_table(results)
    assert "| Documents scored (test split) | 3 |" in table
    assert "### Per template" in markdown_breakdowns(results)


def test_gate_reports_each_violation(tmp_path: Path) -> None:
    results = {
        "test": {
            "fields": {"micro": {"f1": 0.5}},
            "line_items": {"f1": 0.2},
            "transactions": {"f1": 0.3},
            "acceptance": {"accepted_record_accuracy": 0.9},
            "doc_type_accuracy": 0.99,
            "ocr_cer": {"seen": 0.2},
            "latency_ms": {"p95": 5000.0},
        }
    }
    assert evaluate_gate(results, load_thresholds(tmp_path / "missing.json")) == []
    strict = tmp_path / "thresholds.json"
    strict.write_text(json.dumps({"fields_micro_f1_min": 0.9, "latency_p95_ms_max": 1000.0}))
    failures = evaluate_gate(results, load_thresholds(strict))
    assert len(failures) == 2 and failures[0].startswith("fields_micro_f1_min")
