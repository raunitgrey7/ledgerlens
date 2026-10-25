"""Three point-of-sale receipt layouts on narrow paper.

Receipts are the hardest family for layout code: no ruling, centred headers, quantities
written as "2 x", change lines that look like totals, and thermal fonts. The three
templates cover the common conventions seen on shop, supermarket and cafe receipts.
"""

from __future__ import annotations

import random

from ledgerlens.generator.canvas import Canvas
from ledgerlens.generator.content import DocumentContent
from ledgerlens.generator.spec import RenderSpec

PAPER_W = 576
MARGIN = 28
RIGHT = PAPER_W - MARGIN


def receipt_height(content: DocumentContent) -> int:
    """Paper length grows with the number of items."""
    return 800 + 44 * len(content.line_items)


def receipt_thermal(canvas: Canvas, content: DocumentContent, spec: RenderSpec, rng: random.Random) -> None:
    """Centred merchant block, dashed separators, qty x price rows, TOTAL in bold."""
    fields, body = content.fields, spec.font()
    step = spec.leading
    y = 70
    canvas.text(PAPER_W / 2, y, fields.vendor_name or "", spec.font(1.3, True), spec.ink, align="center")
    y = canvas.lines(PAPER_W / 2, y + step + 6, content.vendor_address, body, step, spec.ink, align="center")
    canvas.dashes(MARGIN, RIGHT, y, body, spec.ink)
    y += step
    canvas.text(MARGIN, y, f"Date: {spec.date(fields.document_date)}", body, spec.ink)
    canvas.text(RIGHT, y, f"Time: {content.extras['time']}", body, spec.ink, align="right")
    y += step
    if fields.document_number:
        canvas.text(MARGIN, y, f"Receipt No: {fields.document_number}", body, spec.ink)
        y += step
    canvas.dashes(MARGIN, RIGHT, y, body, spec.ink)
    y += step
    for item in content.line_items:
        canvas.text(MARGIN, y, item.description, body, spec.ink)
        y += step
        canvas.text(MARGIN + 30, y, f"{spec.qty(item.quantity)} x {spec.money(item.unit_price, False)}", body, spec.ink)
        canvas.text(RIGHT, y, spec.money(item.amount, False), body, spec.ink, align="right")
        y += step
    canvas.dashes(MARGIN, RIGHT, y, body, spec.ink)
    y += step
    for label, value in (("Subtotal", fields.subtotal), (content.extras["tax_label"], fields.tax)):
        canvas.text(MARGIN, y, label, body, spec.ink)
        canvas.text(RIGHT, y, spec.money(value, False), body, spec.ink, align="right")
        y += step
    canvas.text(MARGIN, y + 6, "TOTAL", spec.font(1.2, True), spec.ink)
    canvas.text(RIGHT, y + 6, spec.money(fields.total), spec.font(1.2, True), spec.ink, align="right")
    y += step + 16
    canvas.dashes(MARGIN, RIGHT, y, body, spec.ink)
    y += step
    canvas.text(MARGIN, y, f"Paid by {content.extras['payment']}", body, spec.ink)
    y += step
    if fields.gstin:
        canvas.text(MARGIN, y, f"GSTIN: {fields.gstin}", body, spec.ink)
        y += step
    if spec.currency_style == "none":
        canvas.text(MARGIN, y, f"Currency: {spec.currency}", body, spec.ink)
        y += step
    canvas.text(PAPER_W / 2, y + step, "Thank you, visit again", body, spec.ink, align="center")


def receipt_retail(canvas: Canvas, content: DocumentContent, spec: RenderSpec, rng: random.Random) -> None:
    """Left-aligned supermarket style with Qty/Item/Price columns and a Cash/Change block."""
    fields, body, bold = content.fields, spec.font(), spec.font(bold=True)
    step = spec.leading
    y = 70
    canvas.text(MARGIN, y, fields.vendor_name or "", spec.font(1.25, True), spec.ink)
    y = canvas.lines(MARGIN, y + step + 4, content.vendor_address, body, step, spec.ink)
    if fields.gstin:
        canvas.text(MARGIN, y, f"GST No: {fields.gstin}", body, spec.ink)
        y += step
    y += 10
    if fields.document_number:
        canvas.text(MARGIN, y, f"Receipt #: {fields.document_number}", body, spec.ink)
    canvas.text(RIGHT, y, spec.date(fields.document_date), body, spec.ink, align="right")
    y += step + 10
    canvas.hline(MARGIN, RIGHT, y - step + 8, 1, spec.ink)
    canvas.text(MARGIN, y, "Qty", bold, spec.ink)
    canvas.text(MARGIN + 70, y, "Item", bold, spec.ink)
    canvas.text(RIGHT, y, "Price", bold, spec.ink, align="right")
    y += step
    for item in content.line_items:
        canvas.text(MARGIN, y, spec.qty(item.quantity), body, spec.ink)
        canvas.text(MARGIN + 70, y, item.description, body, spec.ink)
        canvas.text(RIGHT, y, spec.money(item.amount, False), body, spec.ink, align="right")
        y += step
    canvas.hline(MARGIN, RIGHT, y - step + 10, 1, spec.ink)
    y += 6
    rows = [("Sub Total", fields.subtotal), ("Tax", fields.tax), ("Total", fields.total)]
    for label, value in rows:
        font = bold if label == "Total" else body
        canvas.text(MARGIN, y, label, font, spec.ink)
        canvas.text(RIGHT, y, spec.money(value, label == "Total"), font, spec.ink, align="right")
        y += step
    if content.extras["payment"] == "CASH":
        tendered = (fields.total // 50 + 1) * 50
        canvas.text(MARGIN, y, "Cash", body, spec.ink)
        canvas.text(RIGHT, y, spec.money(tendered, False), body, spec.ink, align="right")
        y += step
        canvas.text(MARGIN, y, "Change", body, spec.ink)
        canvas.text(RIGHT, y, spec.money(tendered - fields.total, False), body, spec.ink, align="right")
        y += step
    else:
        canvas.text(MARGIN, y, f"{content.extras['payment']} payment approved", body, spec.ink)
        y += step
    if spec.currency_style == "none":
        canvas.text(MARGIN, y, f"All amounts in {spec.currency}", body, spec.ink)
        y += step
    canvas.text(MARGIN, y + step, f"Served by cashier {content.extras['cashier']}", body, spec.ink)


def receipt_cafe(canvas: Canvas, content: DocumentContent, spec: RenderSpec, rng: random.Random) -> None:
    """Cafe style: "2 x Latte" lines, amount right, tax line, Total, payment footer."""
    fields, body = content.fields, spec.font()
    step = spec.leading
    y = 80
    canvas.text(PAPER_W / 2, y, fields.vendor_name or "", spec.font(1.35, True), spec.ink, align="center")
    y = canvas.lines(PAPER_W / 2, y + step + 6, content.vendor_address, body, step, spec.ink, align="center")
    y += 6
    canvas.text(
        PAPER_W / 2, y, f"{spec.date(fields.document_date)} {content.extras['time']}", body, spec.ink, align="center"
    )
    y += step
    if fields.document_number:
        canvas.text(PAPER_W / 2, y, f"Order {fields.document_number}", body, spec.ink, align="center")
        y += step
    canvas.hline(MARGIN, RIGHT, y, 1, spec.ink)
    y += step
    for item in content.line_items:
        canvas.text(MARGIN, y, f"{spec.qty(item.quantity)} x {item.description}", body, spec.ink)
        canvas.text(RIGHT, y, spec.money(item.amount, False), body, spec.ink, align="right")
        y += step
    canvas.hline(MARGIN, RIGHT, y - step + 10, 1, spec.ink)
    y += 4
    canvas.text(MARGIN, y, "Subtotal", body, spec.ink)
    canvas.text(RIGHT, y, spec.money(fields.subtotal, False), body, spec.ink, align="right")
    y += step
    canvas.text(MARGIN, y, content.extras["tax_label"], body, spec.ink)
    canvas.text(RIGHT, y, spec.money(fields.tax, False), body, spec.ink, align="right")
    y += step + 4
    canvas.text(MARGIN, y, "Total", spec.font(1.15, True), spec.ink)
    canvas.text(RIGHT, y, spec.money(fields.total), spec.font(1.15, True), spec.ink, align="right")
    y += step + 10
    canvas.text(MARGIN, y, f"Payment: {content.extras['payment']}", body, spec.ink)
    y += step
    if fields.gstin:
        canvas.text(MARGIN, y, f"GSTIN {fields.gstin}", body, spec.ink)
        y += step
    if spec.currency_style == "none":
        canvas.text(MARGIN, y, f"Currency: {spec.currency}", body, spec.ink)
        y += step
    canvas.text(PAPER_W / 2, y + step, "See you soon", body, spec.ink, align="center")


RECEIPT_TEMPLATES = {
    "receipt_thermal": receipt_thermal,
    "receipt_retail": receipt_retail,
    "receipt_cafe": receipt_cafe,
}
