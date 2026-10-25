"""Five invoice layouts that differ in where labels sit, how tables are ruled and what words are used.

The variety is the point: an extractor that keys on one phrase ("Total") or one geometry
(label left, value right) will fail on some of these, which is what the per-template
benchmark table is designed to reveal.
"""

from __future__ import annotations

import random
from decimal import Decimal

from ledgerlens.generator.canvas import Canvas
from ledgerlens.generator.content import DocumentContent
from ledgerlens.generator.identifiers import format_iban
from ledgerlens.generator.spec import RenderSpec

PAGE_W, PAGE_H = 1240, 1754
MARGIN = 80
RIGHT = PAGE_W - MARGIN


def _identifier_lines(content: DocumentContent, spec: RenderSpec) -> list[str]:
    """Footer identifiers (GSTIN, IBAN) and, on some documents, an explicit currency note."""
    lines: list[str] = []
    if content.fields.gstin:
        lines.append(f"GSTIN: {content.fields.gstin}")
    if content.fields.iban:
        lines.append(f"IBAN: {format_iban(content.fields.iban)}")
    if spec.currency_note and spec.currency_style == "none":
        lines.append(f"All amounts in {spec.currency}")
    return lines


def _table(
    canvas: Canvas,
    content: DocumentContent,
    spec: RenderSpec,
    y: float,
    headers: tuple[str, str, str, str],
    columns: tuple[float, float, float, float],
    ruled: bool,
    grid: bool = False,
) -> float:
    """Line-item table with a header row; returns the baseline after the last row."""
    body, bold = spec.font(), spec.font(bold=True)
    step = spec.leading + 6
    canvas.text(columns[0], y, headers[0], bold, spec.ink)
    for header, x in zip(headers[1:], columns[1:], strict=True):
        canvas.text(x, y, header, bold, spec.ink, align="right")
    if ruled:
        canvas.hline(MARGIN, RIGHT, y + 10, 2, spec.ink)
    y += step
    for item in content.line_items:
        canvas.text(columns[0], y, item.description, body, spec.ink)
        canvas.text(columns[1], y, spec.qty(item.quantity), body, spec.ink, align="right")
        canvas.text(columns[2], y, spec.money(item.unit_price, False), body, spec.ink, align="right")
        canvas.text(columns[3], y, spec.money(item.amount, False), body, spec.ink, align="right")
        if grid:
            canvas.hline(MARGIN, RIGHT, y + 10, 1, spec.ink)
        y += step
    if ruled and not grid:
        canvas.hline(MARGIN, RIGHT, y - step + 12, 1, spec.ink)
    if grid:
        top = y - step * (len(content.line_items) + 1) - spec.body_size
        for x in (MARGIN, columns[1] - 160, columns[1] + 20, columns[2] + 20, RIGHT):
            canvas.vline(x, top, y - step + 10, 1, spec.ink)
        canvas.hline(MARGIN, RIGHT, top, 1, spec.ink)
    return y


def _totals(
    canvas: Canvas,
    content: DocumentContent,
    spec: RenderSpec,
    y: float,
    label_x: float,
    labels: tuple[str, str, str],
    split_tax: bool = False,
) -> float:
    """Subtotal, tax (optionally CGST/SGST halves) and total; returns the next baseline."""
    body, bold = spec.font(), spec.font(bold=True)
    fields = content.fields
    assert fields.subtotal is not None and fields.tax is not None and fields.total is not None
    rows: list[tuple[str, Decimal, bool]] = [(labels[0], fields.subtotal, False)]
    if split_tax:
        half = (fields.tax / 2).quantize(Decimal("0.01"))
        rate = content.extras["tax_label"].split()[-1].rstrip("%")
        half_rate = f"{Decimal(rate) / 2:.0f}" if Decimal(rate) % 2 == 0 else f"{Decimal(rate) / 2}"
        rows += [(f"CGST {half_rate}%", half, False), (f"SGST {half_rate}%", fields.tax - half, False)]
    else:
        rows.append((labels[1], fields.tax, False))
    rows.append((labels[2], fields.total, True))
    for label, value, strong in rows:
        font = bold if strong else body
        canvas.text(label_x, y, label, font, spec.ink)
        canvas.text(RIGHT, y, spec.money(value), font, spec.ink, align="right")
        y += spec.leading + (6 if strong else 2)
    return y


def invoice_classic(canvas: Canvas, content: DocumentContent, spec: RenderSpec, rng: random.Random) -> None:
    """Vendor top-left, INVOICE title top-right, meta block right, Bill To, ruled table, totals right."""
    fields, body, bold = content.fields, spec.font(), spec.font(bold=True)
    canvas.text(MARGIN, 120, fields.vendor_name or "", spec.font(1.5, True), spec.ink)
    canvas.lines(MARGIN, 160, content.vendor_address, body, spec.leading, spec.ink)
    canvas.text(RIGHT, 130, "INVOICE", spec.font(2.2, True), spec.ink, align="right")
    meta = [
        f"Invoice No: {fields.document_number}",
        f"Date: {spec.date(fields.document_date)}",
        f"Due Date: {spec.date(fields.due_date)}",
    ]
    canvas.lines(RIGHT, 190, meta, body, spec.leading, spec.ink, align="right")
    canvas.text(MARGIN, 340, "Bill To:", bold, spec.ink)
    canvas.lines(
        MARGIN,
        340 + spec.leading,
        [fields.customer_name or "", *content.customer_address],
        body,
        spec.leading,
        spec.ink,
    )
    y = _table(
        canvas,
        content,
        spec,
        520,
        ("Description", "Qty", "Unit Price", "Amount"),
        (MARGIN, 760, 960, RIGHT),
        ruled=True,
    )
    y = _totals(canvas, content, spec, y + 30, 820, ("Subtotal", content.extras["tax_label"], "Total"))
    footer = [f"Payment terms: {content.extras['terms']}", *_identifier_lines(content, spec)]
    canvas.lines(MARGIN, y + 60, footer, body, spec.leading, spec.ink)
    canvas.text(PAGE_W / 2, PAGE_H - 90, "Thank you for your business", body, spec.ink, align="center")


def invoice_modern(canvas: Canvas, content: DocumentContent, spec: RenderSpec, rng: random.Random) -> None:
    """Stacked label-over-value meta row, title left, vendor right, unruled table."""
    fields, body, bold = content.fields, spec.font(), spec.font(bold=True)
    small = spec.font(0.8)
    canvas.text(MARGIN, 150, "Invoice", spec.font(2.4, True), spec.ink)
    canvas.text(RIGHT, 120, fields.vendor_name or "", spec.font(1.2, True), spec.ink, align="right")
    canvas.lines(RIGHT, 120 + spec.leading, content.vendor_address, body, spec.leading, spec.ink, align="right")
    canvas.hline(MARGIN, RIGHT, 230, 3, spec.ink)
    meta = (
        ("INVOICE NUMBER", fields.document_number or ""),
        ("ISSUE DATE", spec.date(fields.document_date)),
        ("DUE DATE", spec.date(fields.due_date)),
        ("TOTAL DUE", spec.money(fields.total)),
    )
    for index, (label, value) in enumerate(meta):
        x = MARGIN + index * 270
        canvas.text(x, 280, label, small, spec.ink)
        canvas.text(x, 280 + spec.leading, value, bold, spec.ink)
    canvas.text(MARGIN, 400, "Billed to", small, spec.ink)
    canvas.lines(
        MARGIN,
        400 + spec.leading,
        [fields.customer_name or "", *content.customer_address],
        body,
        spec.leading,
        spec.ink,
    )
    y = _table(
        canvas, content, spec, 560, ("Item", "Quantity", "Price", "Total"), (MARGIN, 780, 960, RIGHT), ruled=True
    )
    y = _totals(canvas, content, spec, y + 30, 820, ("Sub Total", f"Tax ({content.extras['tax_label']})", "Amount Due"))
    canvas.lines(MARGIN, y + 40, _identifier_lines(content, spec), body, spec.leading, spec.ink)


def invoice_boxed(canvas: Canvas, content: DocumentContent, spec: RenderSpec, rng: random.Random) -> None:
    """Everything boxed: key-value grid with colons, fully gridded table, bordered totals."""
    fields, body, _bold = content.fields, spec.font(), spec.font(bold=True)
    canvas.rect(MARGIN, 80, RIGHT, 200, outline=spec.ink, width=2)
    canvas.text(PAGE_W / 2, 135, fields.vendor_name or "", spec.font(1.4, True), spec.ink, align="center")
    canvas.text(PAGE_W / 2, 175, ", ".join(content.vendor_address), body, spec.ink, align="center")
    canvas.rect(MARGIN, 220, RIGHT, 390, outline=spec.ink, width=2)
    canvas.vline(PAGE_W / 2, 220, 390, 2, spec.ink)
    left = [
        f"Invoice No. : {fields.document_number}",
        f"Invoice Date : {spec.date(fields.document_date)}",
        f"Due Date : {spec.date(fields.due_date)}",
        f"PO Number : {content.extras['po_number']}",
    ]
    canvas.lines(MARGIN + 20, 260, left, body, spec.leading, spec.ink)
    right = ["Customer :", fields.customer_name or "", *content.customer_address]
    canvas.lines(PAGE_W / 2 + 20, 260, right, body, spec.leading, spec.ink)
    y = _table(
        canvas,
        content,
        spec,
        460,
        ("Particulars", "Qty", "Rate", "Amount"),
        (MARGIN + 16, 760, 960, RIGHT - 16),
        ruled=True,
        grid=True,
    )
    box_top = y + 20
    y = _totals(canvas, content, spec, y + 50, 800, ("Taxable Value", content.extras["tax_label"], "Grand Total"))
    canvas.rect(780, box_top, RIGHT + 10, y - spec.leading + 12, outline=spec.ink, width=2)
    canvas.lines(
        MARGIN,
        y + 40,
        [*_identifier_lines(content, spec), f"Terms: {content.extras['terms']}"],
        body,
        spec.leading,
        spec.ink,
    )


def invoice_tax(canvas: Canvas, content: DocumentContent, spec: RenderSpec, rng: random.Random) -> None:
    """Indian-style TAX INVOICE with GSTIN under the seller and CGST/SGST split lines."""
    fields, body, bold = content.fields, spec.font(), spec.font(bold=True)
    canvas.text(PAGE_W / 2, 110, "TAX INVOICE", spec.font(1.8, True), spec.ink, align="center")
    canvas.text(MARGIN, 190, fields.vendor_name or "", spec.font(1.3, True), spec.ink)
    header = list(content.vendor_address)
    if fields.gstin:
        header.append(f"GSTIN: {fields.gstin}")
    canvas.lines(MARGIN, 190 + spec.leading, header, body, spec.leading, spec.ink)
    meta = [
        f"Invoice No: {fields.document_number}",
        f"Invoice Date: {spec.date(fields.document_date)}",
        f"Due Date: {spec.date(fields.due_date)}",
    ]
    canvas.lines(RIGHT, 190, meta, body, spec.leading, spec.ink, align="right")
    canvas.text(MARGIN, 380, "Bill To", bold, spec.ink)
    canvas.lines(
        MARGIN,
        380 + spec.leading,
        [fields.customer_name or "", *content.customer_address],
        body,
        spec.leading,
        spec.ink,
    )
    canvas.text(PAGE_W / 2 + 40, 380, "Ship To", bold, spec.ink)
    canvas.lines(
        PAGE_W / 2 + 40,
        380 + spec.leading,
        [fields.customer_name or "", *content.customer_address],
        body,
        spec.leading,
        spec.ink,
    )
    y = _table(
        canvas,
        content,
        spec,
        560,
        ("Description of Goods", "Qty", "Rate", "Amount"),
        (MARGIN, 760, 960, RIGHT),
        ruled=True,
    )
    split = content.country == "IN"
    y = _totals(
        canvas,
        content,
        spec,
        y + 30,
        800,
        ("Taxable Amount", content.extras["tax_label"], "Grand Total"),
        split_tax=split,
    )
    footer = [f"Payment terms: {content.extras['terms']}"]
    if fields.iban:
        footer.append(f"IBAN: {format_iban(fields.iban)}")
    canvas.lines(MARGIN, y + 50, footer, body, spec.leading, spec.ink)
    canvas.text(RIGHT, PAGE_H - 120, f"For {fields.vendor_name}", body, spec.ink, align="right")
    canvas.text(RIGHT, PAGE_H - 120 + spec.leading, "Authorised Signatory", body, spec.ink, align="right")


def invoice_letter(canvas: Canvas, content: DocumentContent, spec: RenderSpec, rng: random.Random) -> None:
    """Letter-like layout: prose header, two-column key-value block, light table, totals left-labelled."""
    fields, body, bold = content.fields, spec.font(), spec.font(bold=True)
    canvas.text(MARGIN, 120, fields.vendor_name or "", spec.font(1.3, True), spec.ink)
    canvas.text(MARGIN, 120 + spec.leading, " | ".join(content.vendor_address), body, spec.ink)
    canvas.hline(MARGIN, RIGHT, 190, 1, spec.ink)
    canvas.text(MARGIN, 250, f"Invoice {fields.document_number}", spec.font(1.6, True), spec.ink)
    pairs = [
        ("Invoice date", spec.date(fields.document_date)),
        ("Payment due", spec.date(fields.due_date)),
        ("Customer", fields.customer_name or ""),
        ("Reference", content.extras["po_number"]),
    ]
    for index, (label, value) in enumerate(pairs):
        x = MARGIN if index % 2 == 0 else PAGE_W / 2 + 40
        y = 320 + (index // 2) * spec.leading
        canvas.text(x, y, f"{label}:", bold, spec.ink)
        canvas.text(x + 220, y, value, body, spec.ink)
    canvas.lines(MARGIN, 320 + 2 * spec.leading + 10, content.customer_address, body, spec.leading, spec.ink)
    y = _table(
        canvas,
        content,
        spec,
        540,
        ("Service / Product", "Units", "Unit Cost", "Line Total"),
        (MARGIN, 780, 980, RIGHT),
        ruled=False,
    )
    canvas.hline(MARGIN, RIGHT, y - spec.leading + 14, 1, spec.ink)
    y = _totals(canvas, content, spec, y + 30, 760, ("Net Amount", content.extras["tax_label"], "Balance Due"))
    footer = [*_identifier_lines(content, spec), f"Please pay within terms ({content.extras['terms']})."]
    canvas.lines(MARGIN, y + 60, footer, body, spec.leading, spec.ink)


INVOICE_TEMPLATES = {
    "invoice_classic": invoice_classic,
    "invoice_modern": invoice_modern,
    "invoice_boxed": invoice_boxed,
    "invoice_tax": invoice_tax,
    "invoice_letter": invoice_letter,
}
