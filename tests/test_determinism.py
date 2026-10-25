import hashlib
from pathlib import Path

from ledgerlens.generator.dataset import SplitSpec, build_document, build_split
from ledgerlens.pipeline.run import Pipeline


def test_same_seed_gives_identical_pages_and_truth() -> None:
    spec = SplitSpec("det", 5, 31)
    for index in range(3):
        page_a, truth_a = build_document(spec, index)
        page_b, truth_b = build_document(spec, index)
        assert truth_a == truth_b
        assert hashlib.sha256(page_a.tobytes()).hexdigest() == hashlib.sha256(page_b.tobytes()).hexdigest()


def test_split_fingerprint_is_stable_across_builds(tmp_path: Path) -> None:
    spec = SplitSpec("det", 4, 31)
    first = build_split(tmp_path / "a", spec)
    second = build_split(tmp_path / "b", spec)
    assert first.fingerprint == second.fingerprint and first.doc_ids == second.doc_ids
    cached = build_split(tmp_path / "a", spec)
    assert cached.fingerprint == first.fingerprint


def test_page_index_independent_of_split_size() -> None:
    small = build_document(SplitSpec("s", 2, 9), 1)[1]
    large = build_document(SplitSpec("s", 50, 9), 1)[1]
    assert small.fields == large.fields


def test_pipeline_output_is_deterministic(pipeline: Pipeline, page_by_type) -> None:
    for page, _ in page_by_type.values():
        first = pipeline.process(page)
        second = pipeline.process(page)
        assert first.field_map() == second.field_map()
        assert first.document_confidence == second.document_confidence
        assert first.line_items == second.line_items and first.transactions == second.transactions
