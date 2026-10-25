"""Two bank statement layouts: a ruled table and a plain zebra-striped ledger.

Statements test the pipeline on long numeric tables where a single misread digit
breaks the running balance chain, which is exactly what the balance rule detects.
"""

from __future__ import annotations

import random

from ledgerlens.generator.canvas import Canvas
from ledgerlens.generator.content import DocumentContent
from ledgerlens.generator.identifiers import format_iban
from ledgerlens.generator.spec import RenderSpec

PAGE_W, PAGE_H = 1240, 1754
MARGIN = 80
RIGHT = PAGE_W - MARGIN


def _account_line(content: DocumentContent) -> str:
    if content.fields.iban:
        return f"IBAN: {format_iban(content.fields.iban)}"
    return f"Account No: {content.extras['account_number']}"


def _rows(
    canvas: Canvas,
    content: DocumentContent,
    spec: RenderSpec,
    y: float,
    columns: tuple[float, float, float, float, float],
    zebra: bool,
) -> float:
    """Transaction rows; returns the baseline after the last row."""
    body = spec.font()
    step = spec.leading + 4
    for index, row in enumerate(content.transactions):
        if zebra and index % 2 == 0:
            canvas.rect(MARGIN, y - spec.body_size - 2, RIGHT, y + 10, fill=(232, 232, 232))
        canvas.text(columns[0], y, spec.date(row.date), body, spec.ink)
        canvas.text(columns[1], y, row.description, body, spec.ink)
        if row.debit is not None:
            canvas.text(columns[2], y, spec.money(row.debit, False), body, spec.ink, align="right")
        if row.credit is not None:
            canvas.text(columns[3], y, spec.money(row.credit, False), body, spec.ink, align="right")
        canvas.text(columns[4], y, spec.money(row.balance, False), body, spec.ink, align="right")
        y += step
    return y


def statement_ruled(canvas: Canvas, content: DocumentContent, spec: RenderSpec, rng: random.Random) -> None:
    """Bank header, holder block, period line, fully ruled Date/Description/Debit/Credit/Balance table."""
    fields, body, bold = content.fields, spec.font(), spec.font(bold=True)
    canvas.text(MARGIN, 120, fields.vendor_name or "", spec.font(1.5, True), spec.ink)
    canvas.lines(MARGIN, 160, content.vendor_address, body, spec.leading, spec.ink)
    canvas.text(RIGHT, 120, "Account Statement", spec.font(1.5, True), spec.ink, align="right")
    canvas.text(RIGHT, 160, f"Statement No: {fields.document_number}", body, spec.ink, align="right")
    canvas.text(
        RIGHT, 160 + spec.leading, f"Statement Date: {spec.date(fields.document_date)}", body, spec.ink, align="right"
    )
    canvas.lines(
        MARGIN,
        300,
        [fields.customer_name or "", *content.customer_address, _account_line(content)],
        body,
        spec.leading,
        spec.ink,
    )
    period = f"Statement Period: {spec.date(fields.period_start)} to {spec.date(fields.period_end)}"
    canvas.text(MARGIN, 460, period, bold, spec.ink)
    canvas.text(MARGIN, 460 + spec.leading, f"Currency: {fields.currency}", body, spec.ink)
    canvas.text(RIGHT, 460, "Opening Balance", bold, spec.ink, align="right")
    canvas.text(RIGHT, 460 + spec.leading, spec.money(fields.opening_balance), body, spec.ink, align="right")
    columns = (MARGIN + 12, MARGIN + 190, 820, 980, RIGHT - 12)
    y = 560
    canvas.hline(MARGIN, RIGHT, y - spec.body_size - 8, 2, spec.ink)
    for header, x, align in (
        ("Date", columns[0], "left"),
        ("Description", columns[1], "left"),
        ("Debit", columns[2], "right"),
        ("Credit", columns[3], "right"),
        ("Balance", columns[4], "right"),
    ):
        canvas.text(x, y, header, bold, spec.ink, align=align)
    canvas.hline(MARGIN, RIGHT, y + 10, 2, spec.ink)
    top = y - spec.body_size - 8
    y = _rows(canvas, content, spec, y + spec.leading + 8, columns, zebra=False)
    canvas.hline(MARGIN, RIGHT, y - spec.leading + 6, 2, spec.ink)
    for x in (MARGIN, columns[1] - 14, 660, 840, 1000, RIGHT):
        canvas.vline(x, top, y - spec.leading + 6, 1, spec.ink)
    canvas.text(RIGHT - 200, y + 20, "Closing Balance", bold, spec.ink, align="right")
    canvas.text(RIGHT, y + 20, spec.money(fields.closing_balance), bold, spec.ink, align="right")


def statement_ledger(canvas: Canvas, content: DocumentContent, spec: RenderSpec, rng: random.Random) -> None:
    """Zebra-striped ledger with Money out / Money in wording and brought/carried forward rows."""
    fields, body, bold = content.fields, spec.font(), spec.font(bold=True)
    canvas.text(PAGE_W / 2, 110, fields.vendor_name or "", spec.font(1.6, True), spec.ink, align="center")
    canvas.text(PAGE_W / 2, 150, ", ".join(content.vendor_address), body, spec.ink, align="center")
    canvas.hline(MARGIN, RIGHT, 180, 1, spec.ink)
    canvas.text(MARGIN, 240, "Your statement", spec.font(1.3, True), spec.ink)
    holder = [fields.customer_name or "", *content.customer_address]
    canvas.lines(MARGIN, 240 + spec.leading + 6, holder, body, spec.leading, spec.ink)
    right = [
        _account_line(content),
        f"Statement number {fields.document_number}",
        f"Issued {spec.date(fields.document_date)}",
        f"Period {spec.date(fields.period_start)} - {spec.date(fields.period_end)}",
    ]
    canvas.lines(RIGHT, 240, right, body, spec.leading, spec.ink, align="right")
    columns = (MARGIN + 12, MARGIN + 190, 840, 1000, RIGHT - 12)
    y = 470
    for header, x, align in (
        ("Date", columns[0], "left"),
        ("Details", columns[1], "left"),
        ("Money out", columns[2], "right"),
        ("Money in", columns[3], "right"),
        ("Balance", columns[4], "right"),
    ):
        canvas.text(x, y, header, bold, spec.ink, align=align)
    canvas.hline(MARGIN, RIGHT, y + 10, 1, spec.ink)
    y += spec.leading + 10
    canvas.text(columns[1], y, "Balance brought forward", body, spec.ink)
    canvas.text(columns[4], y, spec.money(fields.opening_balance, False), body, spec.ink, align="right")
    y = _rows(canvas, content, spec, y + spec.leading + 4, columns, zebra=True)
    canvas.text(columns[1], y + 6, "Balance carried forward", bold, spec.ink)
    canvas.text(columns[4], y + 6, spec.money(fields.closing_balance, False), bold, spec.ink, align="right")
    canvas.text(MARGIN, PAGE_H - 100, f"All amounts in {fields.currency}.", body, spec.ink)


STATEMENT_TEMPLATES = {
    "statement_ruled": statement_ruled,
    "statement_ledger": statement_ledger,
}
