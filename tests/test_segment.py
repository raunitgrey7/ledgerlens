import numpy as np
from PIL import Image, ImageDraw

from ledgerlens.generator.fonts import SEEN_FAMILIES, family_font
from ledgerlens.pipeline.preprocess import clean_impulses, sauvola
from ledgerlens.pipeline.segment import extract_components, group_lines, monospace_pitch, segment


def _render(lines: list[tuple[int, int, str, int, bool]], size: tuple[int, int] = (900, 200)) -> np.ndarray:
    canvas = Image.new("L", size, 252)
    draw = ImageDraw.Draw(canvas)
    for x, y, text, px, mono in lines:
        draw.text(
            (x, y), text, font=family_font(SEEN_FAMILIES[2] if mono else SEEN_FAMILIES[0], px), fill=0, anchor="ls"
        )
    return clean_impulses(sauvola(np.asarray(canvas)))


def test_words_are_split_correctly_for_proportional_and_mono() -> None:
    ink = _render([(20, 60, "Invoice No: INV-2026-00421", 22, False), (20, 130, "Total due 1,234.00 USD", 22, True)])
    lines, _ = segment(ink)
    words = [["".join("x" for _ in w) for w in line.words] for line in lines]
    assert [len(w) for w in words] == [3, 4]


def test_monospace_pitch_detected_only_for_lattices() -> None:
    regular = np.array([13.0, 13.2, 12.8, 26.1, 13.0, 13.1, 39.0, 13.0])
    assert monospace_pitch(regular, 16.0) is not None
    irregular = np.array([9.0, 14.0, 11.0, 17.5, 8.0, 22.0, 12.5, 15.0])
    assert monospace_pitch(irregular, 16.0) is None


def test_tall_title_does_not_swallow_neighbouring_small_lines() -> None:
    ink = _render(
        [
            (20, 120, "Invoice", 60, False),
            (500, 95, "Harbour Logistics Ltd", 24, False),
            (500, 130, "12 High Street", 22, False),
        ],
        size=(900, 180),
    )
    lines, _ = segment(ink)
    assert len(lines) == 3
    assert sorted(len(line.words) for line in lines) == [1, 3, 3]


def test_rules_are_separated_from_glyphs() -> None:
    canvas = Image.new("L", (600, 120), 255)
    draw = ImageDraw.Draw(canvas)
    draw.text((20, 50), "Description Qty", font=family_font(SEEN_FAMILIES[0], 22), fill=0, anchor="ls")
    draw.line([(20, 62), (580, 62)], fill=0, width=2)
    draw.line([(300, 70), (300, 115)], fill=0, width=2)
    glyphs, rules = extract_components(clean_impulses(sauvola(np.asarray(canvas))))
    assert len(rules) == 2 and {r.horizontal for r in rules} == {True, False}
    assert 12 <= len(glyphs) <= 16


def test_group_lines_merges_dots_of_i_within_line() -> None:
    ink = _render([(20, 60, "ii.ii", 24, False)], size=(200, 90))
    glyphs, _ = extract_components(ink)
    lines = group_lines(glyphs)
    assert len(lines) == 1
    segmented, _ = segment(ink)
    assert len(segmented[0].glyphs) == 5
