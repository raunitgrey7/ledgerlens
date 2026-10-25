"""Run the pipeline over a split and score every document.

Documents are processed in worker processes because a page costs well under a second
but a split holds hundreds; each worker loads the cached atlas once. Results are
plain dataclasses so the summary code never touches images again.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, replace
from pathlib import Path

from ledgerlens.bench.metrics import (
    DocScore,
    line_cer,
    match_line_items,
    match_line_items_relaxed,
    match_transactions,
    match_transactions_relaxed,
    score_fields,
)
from ledgerlens.canonical import canonical_fields
from ledgerlens.generator.dataset import load_page, load_truth
from ledgerlens.pipeline.calibration import FieldCalibrator
from ledgerlens.pipeline.confidence import EXPECTED_FIELDS, document_confidence
from ledgerlens.pipeline.ocr.factory import get_engine
from ledgerlens.pipeline.run import Pipeline
from ledgerlens.pipeline.validate import HARD_RULES
from ledgerlens.schemas import DocType, ExtractedField, GroundTruth


def score_document(truth: GroundTruth, directory: Path, pipeline: Pipeline) -> DocScore:
    """Run one page end to end and compare with its ground truth."""
    page = load_page(directory, truth.doc_id)
    result, ocr = pipeline.process_with_page(page)
    errors, chars = line_cer(truth.words, ocr.words)
    predicted = result.field_map()
    return DocScore(
        doc_id=truth.doc_id,
        doc_type=truth.doc_type.value,
        template=truth.template,
        severity=truth.severity.value,
        font_seen=truth.font_seen,
        type_correct=result.doc_type == truth.doc_type,
        fields=score_fields(canonical_fields(truth.fields), predicted),
        items=match_line_items(truth.line_items, result.line_items),
        transactions=match_transactions(truth.transactions, result.transactions),
        items_relaxed=match_line_items_relaxed(truth.line_items, result.line_items),
        transactions_relaxed=match_transactions_relaxed(truth.transactions, result.transactions),
        confidence=result.document_confidence,
        accepted=result.decision.value == "accepted",
        latency_ms=result.latency_ms,
        cer_errors=errors,
        cer_chars=chars,
        predicted_fields=predicted,
        field_confidences={f.name: f.confidence for f in result.fields},
        predicted_type=result.doc_type.value,
        doc_type_confidence=result.doc_type_confidence,
        missing_fields=result.missing_fields,
        hard_rule_failed=any(rule.passed is False and rule.name in HARD_RULES for rule in result.rules),
    )


def calibration_samples(scores: list[DocScore]) -> list[tuple[str, float, bool]]:
    """(field, raw confidence, value correct) for every predicted field of every document."""
    return [
        (name, confidence, score.fields[name].tp == 1)
        for score in scores
        for name, confidence in score.field_confidences.items()
        if name in score.fields
    ]


def rescore(scores: list[DocScore], calibrator: FieldCalibrator, threshold: float) -> list[DocScore]:
    """Recompute document confidence and the decision under a calibrator, without rerunning OCR."""
    rescored = []
    for score in scores:
        doc_type = DocType(score.predicted_type)
        fields = {
            name: ExtractedField(
                name=name, value=score.predicted_fields.get(name), confidence=calibrator.calibrate(name, confidence)
            )
            for name, confidence in score.field_confidences.items()
        }
        flagged = [name for name in score.missing_fields if name not in EXPECTED_FIELDS[doc_type]]
        confidence = document_confidence(doc_type, fields, score.doc_type_confidence, flagged)
        accepted = not score.hard_rule_failed and confidence >= threshold
        rescored.append(replace(score, confidence=confidence, accepted=accepted))
    return rescored


def _worker(args: tuple[str, str, str, str, str, list[str]]) -> list[dict[str, object]]:
    directory, engine_name, atlas_dir, threshold, calibration, doc_ids = args
    calibrator = FieldCalibrator.load(Path(calibration) if calibration else None)
    pipeline = Pipeline(get_engine(engine_name, atlas_dir), float(threshold), calibrator=calibrator)
    scored = [score_document(load_truth(Path(directory), doc_id), Path(directory), pipeline) for doc_id in doc_ids]
    return [_serialise(score) for score in scored]


def _serialise(score: DocScore) -> dict[str, object]:
    payload = asdict(score)
    return payload


ROW_COUNTS = ("items", "transactions", "items_relaxed", "transactions_relaxed")


def _deserialise(payload: dict[str, object]) -> DocScore:
    from ledgerlens.bench.metrics import Counts

    fields = {name: Counts(**counts) for name, counts in payload["fields"].items()}  # type: ignore[union-attr]
    rows = {key: Counts(**payload[key]) for key in ROW_COUNTS}  # type: ignore[arg-type]
    return DocScore(**{**payload, "fields": fields, **rows})  # type: ignore[arg-type]


def run_split(
    directory: Path,
    doc_ids: list[str],
    engine_name: str,
    atlas_dir: Path,
    threshold: float,
    jobs: int,
    calibration: Path | None = None,
) -> list[DocScore]:
    """Score every document of a split, optionally across processes; order is preserved."""
    if jobs <= 1:
        pipeline = Pipeline(
            get_engine(engine_name, str(atlas_dir)), threshold, calibrator=FieldCalibrator.load(calibration)
        )
        return [score_document(load_truth(directory, doc_id), directory, pipeline) for doc_id in doc_ids]
    chunks = [doc_ids[i::jobs] for i in range(jobs)]
    tasks = [
        (str(directory), engine_name, str(atlas_dir), str(threshold), str(calibration or ""), chunk)
        for chunk in chunks
        if chunk
    ]
    with ProcessPoolExecutor(max_workers=jobs) as pool:
        batches = list(pool.map(_worker, tasks))
    by_id = {payload["doc_id"]: _deserialise(payload) for batch in batches for payload in batch}
    return [by_id[doc_id] for doc_id in doc_ids]
