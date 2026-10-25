import random

import numpy as np
from PIL import Image, ImageDraw

from ledgerlens.generator.content import generate_content
from ledgerlens.generator.degrade import apply_stamp
from ledgerlens.generator.fonts import SEEN_FAMILIES, family_font
from ledgerlens.generator.spec import choose_spec
from ledgerlens.generator.templates import render_document
from ledgerlens.pipeline.preprocess import (
    clean_impulses,
    estimate_noise,
    estimate_skew,
    preprocess,
    quick_ink,
    sauvola,
    suppress_colour_ink,
)
from ledgerlens.schemas import DocType, Severity


def _invoice_image() -> Image.Image:
    rng = random.Random(21)
    content = generate_content(rng, DocType.INVOICE)
    return render_document(content, "invoice_classic", choose_spec(rng, content, SEEN_FAMILIES[0]), rng).image


def test_skew_is_recovered_within_tolerance() -> None:
    image = _invoice_image()
    for angle in (-3.5, -1.2, 0.0, 2.0, 4.0):
        rotated = image.rotate(angle, resample=Image.Resampling.BICUBIC, fillcolor=(255, 255, 255))
        gray = suppress_colour_ink(np.asarray(rotated))
        assert abs(estimate_skew(quick_ink(gray)) - angle) <= 0.3


def test_colour_suppression_removes_stamp_keeps_text() -> None:
    image = _invoice_image()
    stamped = apply_stamp(image, random.Random(2), Severity.HEAVY)
    gray = suppress_colour_ink(np.asarray(stamped))
    plain = suppress_colour_ink(np.asarray(image))
    rgb = np.asarray(stamped).astype(int)
    coloured = (rgb.max(axis=2) - rgb.min(axis=2)) > 48
    assert gray[coloured].mean() > 150
    black_text = plain < 60
    assert (gray[black_text] < 128).mean() > 0.9


def test_sauvola_marks_ink_and_cleanup_keeps_hairlines() -> None:
    canvas = Image.new("L", (200, 60), 250)
    draw = ImageDraw.Draw(canvas)
    draw.text((10, 45), "Total 12.50", font=family_font(SEEN_FAMILIES[0], 22), fill=0, anchor="ls")
    draw.line([(120, 20), (160, 20)], fill=0, width=1)
    ink = sauvola(np.asarray(canvas))
    assert ink[20, 121:159].all()
    cleaned = clean_impulses(ink)
    assert cleaned[20, 121:159].all()
    noisy = ink.copy()
    noisy[5, 5] = True
    assert not clean_impulses(noisy)[5, 5]


def test_noise_estimate_is_monotone() -> None:
    base = np.full((300, 300), 230, dtype=np.float32)
    rng = np.random.default_rng(0)
    levels = [
        estimate_noise(np.clip(base + rng.normal(0, sigma, base.shape), 0, 255).astype(np.uint8))
        for sigma in (0.0, 5.0, 15.0)
    ]
    assert levels[0] < levels[1] < levels[2]


def test_preprocess_returns_aligned_arrays() -> None:
    prepared = preprocess(_invoice_image())
    assert prepared.gray.shape == prepared.ink.shape
    assert prepared.ink.dtype == bool and prepared.gray.dtype == np.uint8
    assert 0.002 < prepared.ink.mean() < 0.2
