from decimal import Decimal

from ledgerlens.canonical import canonical_fields
from ledgerlens.pipeline.extract import _build_item, _receipt_row, classify, extract
from ledgerlens.pipeline.layout import build_rows
from ledgerlens.pipeline.ocr.base import OCRLine, OCRPage, OCRWord
from ledgerlens.pipeline.preprocess import preprocess
from ledgerlens.pipeline.run import Pipeline
from ledgerlens.schemas import DocType


def _word(text: str, x0: int, y0: int = 0) -> OCRWord:
    return OCRWord(text, 0.95, x0, y0, x0 + 12 * len(text), y0 + 18)


def _page(rows: list[list[tuple[str, int]]], y0: int = 100, step: int = 40, width: int = 1240) -> OCRPage:
    lines = []
    for index, row in enumerate(rows):
        words = [_word(text, x) for text, x in row]
        for w in words:
            w.y0, w.y1 = y0 + index * step, y0 + index * step + 18
        lines.append(
            OCRLine(
                words,
                min(w.x0 for w in words),
                words[0].y0,
                max(w.x1 for w in words),
                words[0].y1,
                float(words[0].y1),
                18.0,
            )
        )
    return OCRPage(lines, width, 1754, "test")


def test_classification_uses_vocabulary_and_paper_shape() -> None:
    invoice = _page([[("TAX", 500), ("INVOICE", 560)], [("Bill", 80), ("To", 130)], [("Due", 80), ("Date:", 130)]])
    assert classify(build_rows(invoice), invoice)[0] is DocType.INVOICE
    statement = _page(
        [[("Account", 80), ("Statement", 180)], [("Opening", 80), ("Balance", 180)], [("Debit", 700), ("Credit", 900)]]
    )
    assert classify(build_rows(statement), statement)[0] is DocType.BANK_STATEMENT
    receipt = _page([[("Corner", 100), ("Bakery", 200)], [("Paid", 50), ("by", 100), ("CARD", 140)]], width=576)
    assert classify(build_rows(receipt), receipt)[0] is DocType.RECEIPT


def test_split_tax_rows_are_summed() -> None:
    page = _page(
        [
            [("INVOICE", 500)],
            [("Subtotal", 800), ("1,000.00", 1050)],
            [("CGST", 800), ("9%", 870), ("90.00", 1050)],
            [("SGST", 800), ("9%", 870), ("90.00", 1050)],
            [("Grand", 800), ("Total", 870), ("1,180.00", 1050)],
        ]
    )
    extraction = extract(page, build_rows(page))
    assert extraction.fields["tax"].value == "180.00"
    assert extraction.fields["total"].value == "1180.00"
    assert extraction.fields["subtotal"].value == "1000.00"


def test_percentage_row_is_a_tax_fallback() -> None:
    page = _page(
        [
            [("INVOICE", 500)],
            [("Subtotal", 800), ("500.00", 1050)],
            [("MwSt.", 800), ("19%", 880), ("95.00", 1050)],
            [("Total", 800), ("595.00", 1050)],
        ]
    )
    extraction = extract(page, build_rows(page))
    assert extraction.fields["tax"].value == "95.00"


def test_stacked_labels_read_the_row_below() -> None:
    page = _page(
        [
            [("Invoice", 80)],
            [
                ("INVOICE", 80),
                ("NUMBER", 180),
                ("ISSUE", 400),
                ("DATE", 470),
                ("DUE", 700),
                ("DATE", 750),
                ("TOTAL", 1000),
                ("DUE", 1080),
            ],
            [("INV-2026-00421", 80), ("2026-03-14", 400), ("2026-04-13", 700), ("USD", 1000), ("5,154.78", 1050)],
        ]
    )
    extraction = extract(page, build_rows(page))
    assert extraction.fields["document_number"].value == "INV-2026-00421"
    assert extraction.fields["document_date"].value == "2026-03-14"
    assert extraction.fields["due_date"].value == "2026-04-13"
    assert extraction.fields["total"].value == "5154.78"
    assert extraction.fields["currency"].value == "USD"


def test_receipt_row_patterns_and_item_arithmetic() -> None:
    item = _receipt_row([_word("2", 30), _word("x", 60), _word("291.03", 80)], Decimal("582.06"), "Shampoo 180ml")
    assert (
        item is not None
        and item.description == "Shampoo 180ml"
        and item.quantity == 2
        and item.unit_price == Decimal("291.03")
    )
    item = _receipt_row(
        [_word("6", 30), _word("x", 60), _word("Veg", 80), _word("sandwich", 140)], Decimal("28.50"), ""
    )
    assert item is not None and item.description == "Veg sandwich" and item.quantity == 6
    item = _receipt_row([_word("1", 30), _word("Masala", 60), _word("chai", 140)], Decimal("1.95"), "")
    assert item is not None and item.description == "Masala chai" and item.quantity == 1
    built = _build_item("Freight", None, Decimal("84.62"), Decimal("3384.80"))
    assert built.quantity == 40
    assert _build_item("x", None, None, Decimal("9.00")).unit_price == Decimal("9.00")


KEY_FIELDS = (
    "total",
    "subtotal",
    "tax",
    "document_date",
    "due_date",
    "closing_balance",
    "opening_balance",
    "period_start",
    "period_end",
    "iban",
    "gstin",
    "currency",
)


def test_end_to_end_fields_on_clean_pages(pipeline: Pipeline, clean_pages) -> None:
    """One clean page per template: type always right, at least 90% of key fields exact, row counts right."""
    checked = hits = rows_right = 0
    for page, truth in clean_pages:
        result = pipeline.process(page)
        assert result.doc_type == truth.doc_type, truth.template
        gold, predicted = canonical_fields(truth.fields), result.field_map()
        present = [name for name in KEY_FIELDS if gold.get(name) is not None]
        checked += len(present)
        hits += sum(1 for name in present if predicted.get(name) == gold[name])
        rows = (len(result.line_items), len(result.transactions))
        rows_right += rows == (len(truth.line_items), len(truth.transactions))
    assert len(clean_pages) == 10 and checked >= 50
    assert hits / checked >= 0.9, (hits, checked)
    assert rows_right >= 9


def test_statement_transactions_are_recovered(pipeline: Pipeline, page_by_type) -> None:
    page, truth = page_by_type[DocType.BANK_STATEMENT]
    result = pipeline.process(page)
    assert len(result.transactions) >= len(truth.transactions) - 1
    assert result.field("opening_balance") is not None


def test_ocr_text_is_available_for_debugging(pipeline: Pipeline, page_by_type) -> None:
    page, _ = page_by_type[DocType.INVOICE]
    prepared = preprocess(page)
    ocr = pipeline.engine.recognize(prepared.gray, prepared.ink)
    assert len(ocr.lines) > 10 and "\n" in ocr.text


def test_missing_total_is_recovered_from_a_printed_amount_that_satisfies_the_sum() -> None:
    page = _page(
        [
            [("INVOICE", 500)],
            [("Subtotal", 800), ("2,238.80", 1050)],
            [("VAT5%", 800), ("111.94", 1050)],
            [("Tot:d", 800), ("2,350.74", 1050)],
        ]
    )
    extraction = extract(page, build_rows(page))
    assert extraction.fields["total"].value == "2350.74"
    assert extraction.fields["total"].evidence == "printed amount matching the sum"
    unrelated = _page(
        [
            [("INVOICE", 500)],
            [("Subtotal", 800), ("2,238.80", 1050)],
            [("VAT5%", 800), ("111.94", 1050)],
            [("Tot:d", 800), ("9,999.99", 1050)],
        ]
    )
    assert "total" not in extract(unrelated, build_rows(unrelated)).fields


def test_quantity_survives_a_misread_qty_header() -> None:
    page = _page(
        [
            [("INVOICE", 500)],
            [("Particulars", 80), ("Uty", 720), ("Rate", 900), ("Amount", 1100)],
            [("Office", 80), ("chair", 160), ("7", 750), ("166.48", 900), ("1,165.36", 1100)],
            [("Subtotal", 800), ("1,165.36", 1100)],
        ]
    )
    items = extract(page, build_rows(page)).line_items
    assert len(items) == 1 and items[0].description == "Office chair" and items[0].quantity == 7
