import difflib
import random
import types

import numpy as np
from PIL import Image, ImageDraw

from ledgerlens.generator.fonts import SEEN_FAMILIES, atlas_font_files, family_font
from ledgerlens.pipeline.ocr import factory
from ledgerlens.pipeline.ocr.atlas import AtlasOCR, _class_path, condense, glyph_vector
from ledgerlens.pipeline.ocr.tesseract import TesseractOCR, page_from_table
from ledgerlens.pipeline.preprocess import clean_impulses, sauvola

SAMPLE = "Receipt Date 14/03/2026 Balance 1,234.56 INR Harbour Logistics Ltd"


def _cer(predicted: str, expected: str) -> float:
    matcher = difflib.SequenceMatcher(a=predicted, b=expected, autojunk=False)
    errors = sum(max(i2 - i1, j2 - j1) for tag, i1, i2, j1, j2 in matcher.get_opcodes() if tag != "equal")
    return errors / len(expected)


def _line(text: str, family_index: int, size: int, bold: bool = False) -> tuple[np.ndarray, np.ndarray]:
    font = family_font(SEEN_FAMILIES[family_index], size, bold)
    canvas = Image.new("L", (int(font.getlength(text)) + 80, size * 3), 250)
    ImageDraw.Draw(canvas).text((40, size * 2), text, font=font, fill=30, anchor="ls")
    gray = np.asarray(canvas)
    return gray, clean_impulses(sauvola(gray))


def test_clean_lines_are_read_accurately_in_every_seen_family(engine: AtlasOCR) -> None:
    for index in range(len(SEEN_FAMILIES)):
        for bold in (False, True):
            gray, ink = _line(SAMPLE, index, 22, bold)
            page = engine.recognize(gray, ink)
            assert _cer(page.text, SAMPLE) < 0.08, (SEEN_FAMILIES[index].name, bold, page.text)


def test_recognition_is_deterministic(engine: AtlasOCR) -> None:
    gray, ink = _line(SAMPLE, 1, 20)
    first = engine.recognize(gray, ink)
    second = engine.recognize(gray, ink)
    assert first.text == second.text
    assert [w.confidence for w in first.words] == [w.confidence for w in second.words]


def test_word_confidences_are_probabilities_and_track_quality(engine: AtlasOCR) -> None:
    gray, ink = _line("Total 4,311.04", 0, 22)
    clean = engine.recognize(gray, ink)
    noisy_gray = np.clip(gray.astype(float) + np.random.default_rng(1).normal(0, 25, gray.shape), 0, 255).astype(
        np.uint8
    )
    noisy = engine.recognize(noisy_gray, clean_impulses(sauvola(noisy_gray)))
    assert all(0.0 <= w.confidence <= 1.0 for w in clean.words + noisy.words)
    assert np.mean([w.confidence for w in clean.words]) > np.mean([w.confidence for w in noisy.words])


def test_atlas_covers_fonts_and_condense_keeps_every_label(engine: AtlasOCR) -> None:
    labels = set(engine.atlas.labels.tolist())
    assert "0" in labels and "Z" in labels and "€" in labels
    assert len(atlas_font_files()) == 8
    vectors = np.stack([glyph_vector(np.ones((10, 6), dtype=bool))] * 4)
    geos = np.zeros((4, 3), dtype=np.float32)
    kept_vectors, _, kept_labels = condense(vectors, geos, np.array(["a", "a", "b", "b"]))
    assert kept_labels.tolist() == ["a", "b"] and kept_vectors.shape == (2, vectors.shape[1])


def test_class_path_prefers_digits_inside_numeric_tokens() -> None:
    ambiguous = [("O", 0.90), ("0", 0.88)]
    digit = [("1", 0.95)]
    assert _class_path([digit, ambiguous, ambiguous, digit]) == ["digit"] * 4
    assert _class_path([[("H", 0.95)], [("a", 0.9)], [("o", 0.89), ("0", 0.9)]]) == ["upper", "lower", "lower"]


def test_tesseract_adapter_groups_words_into_lines() -> None:
    table = {
        "text": ["Invoice", "No:", "", "Total"],
        "conf": [96.0, 90.0, -1, 88.0],
        "left": [10, 80, 0, 10],
        "top": [10, 10, 0, 50],
        "width": [60, 30, 0, 50],
        "height": [20, 20, 0, 20],
        "block_num": [1, 1, 1, 1],
        "par_num": [1, 1, 1, 1],
        "line_num": [1, 1, 1, 2],
    }
    page = page_from_table(table, 400, 100, "tesseract")
    assert [line.text for line in page.lines] == ["Invoice No:", "Total"]
    assert page.words[0].confidence == 0.96
    fake = types.SimpleNamespace(image_to_data=lambda *a, **k: table, Output=types.SimpleNamespace(DICT="dict"))
    adapter = TesseractOCR(module=fake)
    page = adapter.recognize(np.full((100, 400), 255, dtype=np.uint8), np.zeros((100, 400), dtype=bool))
    assert page.engine == "tesseract" and len(page.words) == 3


def test_factory_resolution(monkeypatch) -> None:
    monkeypatch.setattr(factory, "tesseract_available", lambda: False)
    assert factory.resolve_engine_name("auto") == "atlas"
    try:
        factory.resolve_engine_name("tesseract")
    except RuntimeError:
        pass
    else:
        raise AssertionError("missing tesseract should raise")
    try:
        factory.resolve_engine_name("bogus")
    except ValueError:
        pass
    else:
        raise AssertionError("unknown engine should raise")
    monkeypatch.setattr(factory, "tesseract_available", lambda: True)
    assert factory.resolve_engine_name("auto") == "tesseract"


def test_random_seeded_lines_keep_low_cer(engine: AtlasOCR) -> None:
    rng = random.Random(5)
    vocabulary = ["Subtotal", "GSTIN", "07ABCDE1234F1Z5", "2,345.00", "Due", "Date:", "Ltd", "Amount", "14-Mar-2026"]
    total_errors = total_chars = 0
    for _ in range(6):
        text = " ".join(rng.choice(vocabulary) for _ in range(4))
        gray, ink = _line(text, rng.randrange(len(SEEN_FAMILIES)), rng.choice((19, 20, 22, 24)))
        page = engine.recognize(gray, ink)
        total_errors += _cer(page.text, text) * len(text)
        total_chars += len(text)
    assert total_errors / total_chars < 0.08
