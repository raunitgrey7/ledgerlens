"""Run the pipeline over a split and score every document.

Documents are processed in worker processes because a page costs well under a second
but a split holds hundreds; each worker loads the cached atlas once. Results are
plain dataclasses so the summary code never touches images again.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from pathlib import Path

from ledgerlens.bench.metrics import DocScore, line_cer, match_line_items, match_transactions, score_fields
from ledgerlens.canonical import canonical_fields
from ledgerlens.generator.dataset import load_page, load_truth
from ledgerlens.pipeline.ocr.factory import get_engine
from ledgerlens.pipeline.preprocess import preprocess
from ledgerlens.pipeline.run import Pipeline
from ledgerlens.schemas import GroundTruth


def score_document(truth: GroundTruth, directory: Path, pipeline: Pipeline) -> DocScore:
    """Run one page end to end and compare with its ground truth."""
    page = load_page(directory, truth.doc_id)
    result = pipeline.process(page)
    prepared = preprocess(page)
    ocr = pipeline.engine.recognize(prepared.gray, prepared.ink)
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
        confidence=result.document_confidence,
        accepted=result.decision.value == "accepted",
        latency_ms=result.latency_ms,
        cer_errors=errors,
        cer_chars=chars,
        predicted_fields=predicted,
    )


def _worker(args: tuple[str, str, str, str, list[str]]) -> list[dict[str, object]]:
    directory, engine_name, atlas_dir, threshold, doc_ids = args
    pipeline = Pipeline(get_engine(engine_name, atlas_dir), float(threshold))
    scored = [score_document(load_truth(Path(directory), doc_id), Path(directory), pipeline) for doc_id in doc_ids]
    return [_serialise(score) for score in scored]


def _serialise(score: DocScore) -> dict[str, object]:
    payload = asdict(score)
    return payload


def _deserialise(payload: dict[str, object]) -> DocScore:
    from ledgerlens.bench.metrics import Counts

    fields = {name: Counts(**counts) for name, counts in payload["fields"].items()}  # type: ignore[union-attr]
    return DocScore(
        **{
            **payload,
            "fields": fields,
            "items": Counts(**payload["items"]),  # type: ignore[arg-type]
            "transactions": Counts(**payload["transactions"]),
        }
    )  # type: ignore[arg-type]


def run_split(
    directory: Path, doc_ids: list[str], engine_name: str, atlas_dir: Path, threshold: float, jobs: int
) -> list[DocScore]:
    """Score every document of a split, optionally across processes; order is preserved."""
    if jobs <= 1:
        pipeline = Pipeline(get_engine(engine_name, str(atlas_dir)), threshold)
        return [score_document(load_truth(directory, doc_id), directory, pipeline) for doc_id in doc_ids]
    chunks = [doc_ids[i::jobs] for i in range(jobs)]
    tasks = [(str(directory), engine_name, str(atlas_dir), str(threshold), chunk) for chunk in chunks if chunk]
    with ProcessPoolExecutor(max_workers=jobs) as pool:
        batches = list(pool.map(_worker, tasks))
    by_id = {payload["doc_id"]: _deserialise(payload) for batch in batches for payload in batch}
    return [by_id[doc_id] for doc_id in doc_ids]
