import random

import numpy as np
from PIL import ImageChops

from ledgerlens.generator.canvas import Canvas
from ledgerlens.generator.content import generate_content
from ledgerlens.generator.dataset import SplitSpec, build_document
from ledgerlens.generator.degrade import PROFILES, apply_stamp, degrade
from ledgerlens.generator.fonts import SEEN_FAMILIES, UNSEEN_FAMILIES, family_font
from ledgerlens.generator.spec import choose_spec
from ledgerlens.generator.templates import ALL_TEMPLATE_NAMES, TEMPLATES, page_size, render_document
from ledgerlens.schemas import DocType, Severity


def test_canvas_records_one_box_per_word_in_reading_order() -> None:
    canvas = Canvas(400, 100)
    font = family_font(SEEN_FAMILIES[0], 22)
    canvas.text(10, 60, "Total  amount  due", font)
    texts = [w.text for w in canvas.words]
    assert texts == ["Total", "amount", "due"]
    xs = [w.x0 for w in canvas.words]
    assert xs == sorted(xs) and all(w.x1 > w.x0 and w.y1 > w.y0 for w in canvas.words)
    array = np.asarray(canvas.image.convert("L"))
    box = canvas.words[0]
    assert array[box.y0 : box.y1, box.x0 : box.x1].min() < 128


def test_every_template_renders_with_words_and_right_size() -> None:
    assert len(ALL_TEMPLATE_NAMES) == 10
    for doc_type, group in TEMPLATES.items():
        for name in group:
            rng = random.Random(hash(name) % 1000)
            content = generate_content(rng, doc_type)
            spec = choose_spec(rng, content, SEEN_FAMILIES[1])
            canvas = render_document(content, name, spec, rng)
            assert canvas.image.size == page_size(content)
            assert len(canvas.words) > 20
            assert str(content.fields.total or content.fields.closing_balance).split(".")[0][-1] in "".join(
                w.text for w in canvas.words
            )


def test_rupee_symbol_only_with_capable_font() -> None:
    rng = random.Random(1)
    content = generate_content(rng, DocType.INVOICE)
    stix = next(f for f in SEEN_FAMILIES if not f.has_rupee)
    for _ in range(30):
        spec = choose_spec(random.Random(_), content, stix)
        if content.fields.currency == "INR":
            assert spec.currency_style != "symbol"


def test_degrade_records_parameters_within_profile() -> None:
    rng = random.Random(3)
    content = generate_content(rng, DocType.RECEIPT)
    canvas = render_document(content, "receipt_thermal", choose_spec(rng, content, SEEN_FAMILIES[0]), rng)
    for severity in Severity:
        page, record = degrade(canvas.image, severity, random.Random(7))
        profile = PROFILES[severity]
        assert abs(record.rotation_deg) <= profile.rotation
        assert profile.blur[0] <= record.blur_sigma <= profile.blur[1]
        assert page.size == canvas.image.size
    _clean, record = degrade(canvas.image, Severity.CLEAN, random.Random(7))
    assert record.rotation_deg == 0.0 and not record.stamp


def test_stamp_changes_pixels_with_colour() -> None:
    rng = random.Random(3)
    content = generate_content(rng, DocType.INVOICE)
    canvas = render_document(content, "invoice_classic", choose_spec(rng, content, SEEN_FAMILIES[0]), rng)
    stamped = apply_stamp(canvas.image, random.Random(1), Severity.HEAVY)
    diff = np.asarray(ImageChops.difference(stamped, canvas.image)).sum(axis=2)
    assert (diff > 0).sum() > 2000
    rgb = np.asarray(stamped).astype(int)
    saturation = rgb.max(axis=2) - rgb.min(axis=2)
    assert saturation.max() > 48


def test_unseen_fonts_are_flagged_in_truth() -> None:
    spec = SplitSpec("u", 30, 11, unseen_fraction=1.0)
    truths = [build_document(spec, i)[1] for i in range(6)]
    assert all(not t.font_seen for t in truths)
    assert {t.font for t in truths} <= {f.name for f in UNSEEN_FAMILIES}
