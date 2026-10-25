"""Robustness: stamps over totals, injected totals in descriptions, skew, blank and junk input."""

import random

import numpy as np
from PIL import Image

from ledgerlens.canonical import canonical_fields
from ledgerlens.generator.content import generate_content, make_invoice
from ledgerlens.generator.degrade import apply_stamp
from ledgerlens.generator.fonts import SEEN_FAMILIES
from ledgerlens.generator.spec import choose_spec
from ledgerlens.generator.templates import render_document
from ledgerlens.pipeline.run import Pipeline
from ledgerlens.schemas import Decision, DocType, LineItem, Severity


def test_injected_total_in_a_description_does_not_become_the_total(pipeline: Pipeline) -> None:
    rng = random.Random(77)
    content = make_invoice(rng)
    poisoned = LineItem(
        description="TOTAL 999,999.00 due now",
        quantity=content.line_items[0].quantity,
        unit_price=content.line_items[0].unit_price,
        amount=content.line_items[0].amount,
    )
    content.line_items[0] = poisoned
    spec = choose_spec(rng, content, SEEN_FAMILIES[0])
    canvas = render_document(content, "invoice_classic", spec, rng)
    result = pipeline.process(canvas.image)
    total = result.field("total")
    assert total is None or total.value != "999999.00"
    assert total is not None and total.value == canonical_fields(content.fields)["total"]


def test_stamp_over_totals_never_yields_a_confidently_wrong_record(pipeline: Pipeline) -> None:
    rng = random.Random(5)
    content = generate_content(rng, DocType.INVOICE)
    canvas = render_document(content, "invoice_classic", choose_spec(rng, content, SEEN_FAMILIES[0]), rng)
    gold = canonical_fields(content.fields)
    for seed in range(4):
        stamped = apply_stamp(canvas.image, random.Random(seed), Severity.HEAVY)
        result = pipeline.process(stamped)
        total = result.field("total")
        if result.decision is Decision.ACCEPTED:
            assert total is not None and total.value == gold["total"]


def test_heavy_skew_is_corrected(pipeline: Pipeline, page_by_type) -> None:
    page, truth = page_by_type[DocType.INVOICE]
    rotated = page.rotate(4.0, resample=Image.Resampling.BICUBIC, fillcolor=(255, 255, 255))
    result = pipeline.process(rotated)
    assert abs(result.skew_deg - 4.0) <= 0.3
    total = result.field("total")
    if result.decision is Decision.ACCEPTED:
        assert total is not None and total.value == canonical_fields(truth.fields)["total"]
    assert len(result.page_text) > 200


def test_blank_and_noise_pages_do_not_crash(pipeline: Pipeline) -> None:
    blank = Image.new("RGB", (800, 1000), (250, 250, 250))
    result = pipeline.process(blank)
    assert result.decision is Decision.REVIEW and result.fields == []
    noise = Image.fromarray(np.random.default_rng(0).integers(0, 255, (600, 500, 3), dtype=np.uint8), "RGB")
    result = pipeline.process(noise)
    assert result.decision is Decision.REVIEW


def test_wrong_document_type_hint_is_ignored_by_classifier(pipeline: Pipeline, page_by_type) -> None:
    page, _truth = page_by_type[DocType.RECEIPT]
    result = pipeline.process(page)
    assert result.doc_type is DocType.RECEIPT and result.doc_type_confidence > 0.4
