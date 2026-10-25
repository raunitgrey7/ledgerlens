from ledgerlens.pipeline.layout import (
    build_rows,
    find_label,
    find_table_header,
    line_below,
    normalise,
    similarity,
    words_below,
)
from ledgerlens.pipeline.ocr.base import OCRLine, OCRPage, OCRWord


def _word(text: str, x0: int, y0: int, conf: float = 0.95) -> OCRWord:
    return OCRWord(text, conf, x0, y0, x0 + 12 * len(text), y0 + 18)


def _line(words: list[OCRWord]) -> OCRLine:
    return OCRLine(
        words,
        min(w.x0 for w in words),
        min(w.y0 for w in words),
        max(w.x1 for w in words),
        max(w.y1 for w in words),
        float(max(w.y1 for w in words)),
        18.0,
    )


def _page() -> OCRPage:
    lines = [
        _line([_word("Invoice", 80, 100), _word("No:", 180, 100), _word("INV-1", 240, 100)]),
        _line([_word("Tota1", 800, 101), _word("12,119.68", 1000, 100)]),
        _line([_word("Bill", 80, 150), _word("To:", 140, 150)]),
        _line([_word("Vistara", 80, 185), _word("Software", 180, 185)]),
        _line([_word("Ship", 700, 150), _word("To", 760, 150)]),
        _line([_word("Granite", 700, 185), _word("Pharma", 800, 185)]),
        _line(
            [_word("Description", 80, 300), _word("Qty", 700, 300), _word("Rate", 900, 300), _word("Amount", 1100, 300)]
        ),
    ]
    return OCRPage(lines, 1240, 1754, "test")


def test_rows_merge_lines_with_overlapping_bands() -> None:
    rows = build_rows(_page())
    assert len(rows) == 4
    assert rows[0].text == "Invoice No: INV-1 Tota1 12,119.68"


def test_fuzzy_labels_with_exclusions() -> None:
    rows = build_rows(_page())
    totals = find_label(rows, ("total",), 0.8, exclude=("sub total",))
    assert totals and totals[0].rest[0].text == "12,119.68"
    assert find_label(rows, ("sub total",), 0.8) == []
    assert similarity("Tota1", "total") >= 0.8 and normalise("Bill To:") == "billto"


def test_values_below_a_label_stay_in_their_block() -> None:
    rows = build_rows(_page())
    match = find_label(rows, ("bill to",), 0.8, exclude=("ship to",))[0]
    below = words_below(rows, match.row_index, match.words[0].x0, match.words[-1].x1)
    assert [w.text for w in below] == ["Vistara", "Software"]
    assert [w.text for w in line_below(rows, match.row_index, match.words[0].x0, match.words[-1].x1)] == [
        "Vistara",
        "Software",
    ]


def test_table_header_defines_columns_by_position() -> None:
    rows = build_rows(_page())
    header = find_table_header(
        rows,
        {"description": ("description",), "quantity": ("qty",), "unit_price": ("rate",), "amount": ("amount",)},
        1240,
        minimum=3,
    )
    assert header is not None and [c.name for c in header.columns] == [
        "description",
        "quantity",
        "unit_price",
        "amount",
    ]
    assert header.column_for(_word("5", 720, 340)).name == "quantity"
    assert header.column_for(_word("1,000.00", 1120, 340)).name == "amount"
    assert header.columns[-1].x1 == 1240.0
