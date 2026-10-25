"""Template registry and the single entry point that turns content into a rendered page."""

from __future__ import annotations

import random
from collections.abc import Callable

from ledgerlens.generator.canvas import Canvas
from ledgerlens.generator.content import DocumentContent
from ledgerlens.generator.spec import RenderSpec
from ledgerlens.generator.templates_invoice import INVOICE_TEMPLATES, PAGE_H, PAGE_W
from ledgerlens.generator.templates_receipt import PAPER_W, RECEIPT_TEMPLATES, receipt_height
from ledgerlens.generator.templates_statement import STATEMENT_TEMPLATES
from ledgerlens.schemas import DocType

TemplateFn = Callable[[Canvas, DocumentContent, RenderSpec, random.Random], None]

TEMPLATES: dict[DocType, dict[str, TemplateFn]] = {
    DocType.INVOICE: INVOICE_TEMPLATES,
    DocType.RECEIPT: RECEIPT_TEMPLATES,
    DocType.BANK_STATEMENT: STATEMENT_TEMPLATES,
}
ALL_TEMPLATE_NAMES: tuple[str, ...] = tuple(name for group in TEMPLATES.values() for name in group)


def page_size(content: DocumentContent) -> tuple[int, int]:
    """A4 at 150 dpi for invoices and statements, 80 mm thermal paper for receipts."""
    if content.doc_type is DocType.RECEIPT:
        return PAPER_W, receipt_height(content)
    return PAGE_W, PAGE_H


def render_document(content: DocumentContent, template: str, spec: RenderSpec, rng: random.Random) -> Canvas:
    """Render ``content`` with the named template; the canvas carries the word boxes."""
    width, height = page_size(content)
    canvas = Canvas(width, height)
    TEMPLATES[content.doc_type][template](canvas, content, spec, rng)
    return canvas


def template_for(content: DocumentContent, rng: random.Random) -> str:
    """Uniformly choose a template for the document's type."""
    return rng.choice(sorted(TEMPLATES[content.doc_type]))
