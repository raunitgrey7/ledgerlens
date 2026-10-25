"""Command line entry: build splits, run, score, choose the threshold, write results, gate.

uv run python -m ledgerlens.bench --test-size 400 --cal-size 200 --jobs 4
uv run python -m ledgerlens.bench --gate
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

from ledgerlens import __version__
from ledgerlens.bench.gate import evaluate_gate, load_thresholds
from ledgerlens.bench.metrics import DocScore, EffortModel, choose_threshold
from ledgerlens.bench.report import markdown_breakdowns, markdown_table, summarise
from ledgerlens.bench.runner import calibration_samples, rescore, run_split
from ledgerlens.config import get_settings
from ledgerlens.generator.dataset import Manifest, SplitSpec, build_split, split_dir
from ledgerlens.pipeline.calibration import FieldCalibrator, fit_calibrator
from ledgerlens.pipeline.ocr.factory import resolve_engine_name

CAL_SEED = 101
TEST_SEED = 303
TARGET_ACCEPTED_HEADER_ACCURACY = 0.95


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="ledgerlens.bench")
    parser.add_argument("--test-size", type=int, default=400)
    parser.add_argument("--cal-size", type=int, default=200)
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--out", type=Path, default=Path("benchmarks"))
    parser.add_argument("--data", type=Path, default=None)
    parser.add_argument("--engine", default=None)
    parser.add_argument("--gate", action="store_true", help="fail when results violate the thresholds file")
    parser.add_argument("--thresholds", type=Path, default=Path("benchmarks/thresholds.json"))
    parser.add_argument("--rebuild", action="store_true", help="regenerate the splits even if cached")
    return parser.parse_args(argv)


def git_commit() -> str:
    try:
        return (
            subprocess.run(
                ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=False, timeout=5
            ).stdout.strip()
            or "unknown"
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def write_documents(path: Path, scores: list[DocScore]) -> None:
    """One line per test document: what was decided and which fields were wrong, for error analysis."""
    with path.open("w") as handle:
        for score in scores:
            wrong = sorted(name for name, counts in score.fields.items() if counts.fp or counts.fn)
            handle.write(
                json.dumps(
                    {
                        "doc_id": score.doc_id,
                        "template": score.template,
                        "severity": score.severity,
                        "font_seen": score.font_seen,
                        "predicted_type": score.predicted_type,
                        "confidence": score.confidence,
                        "accepted": score.accepted,
                        "header_correct": score.header_correct,
                        "record_correct": score.record_correct,
                        "wrong_fields": wrong,
                        "missing_fields": score.missing_fields,
                        "latency_ms": score.latency_ms,
                        "cer": round(score.cer_errors / score.cer_chars, 4) if score.cer_chars else 0.0,
                    }
                )
                + "\n"
            )


def calibrate_and_choose(raw_cal_scores: list[DocScore], out: Path) -> tuple[FieldCalibrator, float, list[DocScore]]:
    """Fit the field calibration on the calibration split, save it, and choose the threshold on it."""
    calibrator = fit_calibrator(calibration_samples(raw_cal_scores))
    calibrator.save(out / "calibration.json")
    threshold = choose_threshold(rescore(raw_cal_scores, calibrator, 0.0), TARGET_ACCEPTED_HEADER_ACCURACY)
    return calibrator, threshold, rescore(raw_cal_scores, calibrator, threshold)


def run_meta(
    args: argparse.Namespace, engine_name: str, manifests: dict[str, Manifest], seconds: dict[str, float]
) -> dict[str, object]:
    """Provenance recorded with every result: code, machine, engine, split inputs and wall time."""
    seeds = {"calibration": (args.cal_size, CAL_SEED), "test": (args.test_size, TEST_SEED)}
    return {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "ledgerlens_version": __version__,
        "git_commit": git_commit(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "ocr_engine": engine_name,
        "jobs": args.jobs,
        "splits": {
            name: {"size": size, "seed": seed, "fingerprint": manifests[name].fingerprint}
            for name, (size, seed) in seeds.items()
        },
        "seconds": seconds,
    }


def write_outputs(out: Path, results: dict[str, object], test_scores: list[DocScore], threshold: float) -> str:
    """Write results.json, documents.jsonl, threshold.json and RESULTS.md; returns the headline table."""
    (out / "results.json").write_text(json.dumps(results, indent=2))
    write_documents(out / "documents.jsonl", test_scores)
    (out / "threshold.json").write_text(json.dumps({"threshold": threshold}, indent=2))
    table = markdown_table(results)
    meta = results["meta"]
    (out / "RESULTS.md").write_text(
        f"# Benchmark results\n\nGenerated {meta['generated_at']} with engine `{meta['ocr_engine']}` "  # type: ignore[index]
        f"on {meta['platform']}.\n\n{table}\n\n{markdown_breakdowns(results)}\n"  # type: ignore[index]
    )
    return table


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    settings = get_settings()
    data_root = (args.data or settings.data_dir) / "synthetic"
    engine_name = resolve_engine_name(args.engine or settings.ocr_engine)
    args.out.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    manifests = {
        "calibration": build_split(
            data_root, SplitSpec("calibration", args.cal_size, CAL_SEED), args.rebuild, args.jobs
        ),
        "test": build_split(data_root, SplitSpec("test", args.test_size, TEST_SEED), args.rebuild, args.jobs),
    }
    built = time.perf_counter()
    raw_cal_scores = run_split(
        split_dir(data_root, "calibration"),
        manifests["calibration"].doc_ids,
        engine_name,
        settings.atlas_dir,
        0.0,
        args.jobs,
    )
    calibrator, threshold, cal_scores = calibrate_and_choose(raw_cal_scores, args.out)
    test_scores = run_split(
        split_dir(data_root, "test"),
        manifests["test"].doc_ids,
        engine_name,
        settings.atlas_dir,
        threshold,
        args.jobs,
        calibration=args.out / "calibration.json",
    )
    finished = time.perf_counter()
    effort = EffortModel()
    results: dict[str, object] = {
        "meta": run_meta(
            args,
            engine_name,
            manifests,
            {"build_data": round(built - started, 1), "evaluate": round(finished - built, 1)},
        ),
        "field_calibration": {name: list(params) for name, params in sorted(calibrator.params.items())},
        "threshold": {
            "value": threshold,
            "target_accepted_header_accuracy": TARGET_ACCEPTED_HEADER_ACCURACY,
            "chosen_on": "calibration",
        },
        "calibration": summarise(cal_scores, threshold, effort),
        "test": summarise(test_scores, threshold, effort),
    }
    print(write_outputs(args.out, results, test_scores, threshold))
    failures = evaluate_gate(results, load_thresholds(args.thresholds)) if args.gate else []
    for failure in failures:
        print(f"GATE FAIL {failure}")
    print(f"wrote {args.out / 'results.json'} ({finished - started:.0f}s)")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
