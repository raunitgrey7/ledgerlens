"""Per-document presentation choices: font family, sizes, ink, separators, date style.

A template describes where things go; a RenderSpec describes how they look. Keeping
them apart lets every template be rendered in every font and locale style, which
multiplies the visual variety the OCR and the parser are tested on.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from datetime import date
from decimal import Decimal

from PIL import ImageFont

from ledgerlens.generator.content import DocumentContent
from ledgerlens.generator.fonts import ALL_FAMILIES, SEEN_FAMILIES, UNSEEN_FAMILIES, FontFamily, family_font
from ledgerlens.generator.format import DATE_STYLES, NUMBER_STYLES, fmt_date, fmt_money, fmt_qty

Color = tuple[int, int, int]


@dataclass(frozen=True)
class RenderSpec:
    """Everything a template needs to turn content into pixels."""

    family: FontFamily
    body_size: int
    ink: Color
    date_style: str
    number_style: str
    currency_style: str
    currency: str

    def font(self, scale: float = 1.0, bold: bool = False) -> ImageFont.FreeTypeFont:
        return family_font(self.family, max(10, round(self.body_size * scale)), bold)

    def money(self, value: Decimal, with_currency: bool = True) -> str:
        style = self.currency_style if with_currency else "none"
        return fmt_money(value, self.currency, self.number_style, style)

    def date(self, value: date) -> str:
        return fmt_date(value, self.date_style)

    @staticmethod
    def qty(value: Decimal) -> str:
        return fmt_qty(value)

    @property
    def leading(self) -> int:
        """Baseline-to-baseline distance for body text."""
        return round(self.body_size * 1.55)


def choose_family(rng: random.Random, unseen_fraction: float) -> FontFamily:
    """Pick a font family; ``unseen_fraction`` of pages use faces the OCR atlas never saw."""
    pool = UNSEEN_FAMILIES if rng.random() < unseen_fraction else SEEN_FAMILIES
    return rng.choice(pool)


def choose_spec(
    rng: random.Random, content: DocumentContent, family: FontFamily | None = None, unseen_fraction: float = 0.15
) -> RenderSpec:
    """Draw presentation choices consistent with the document's country."""
    chosen = family or choose_family(rng, unseen_fraction)
    currency = content.fields.currency or "USD"
    currency_styles = ["code", "code", "none", "none"]
    if chosen.has_rupee or currency != "INR":
        currency_styles += ["symbol", "symbol"]
    body = rng.choice((19, 20, 22, 24)) if content.doc_type.value != "receipt" else rng.choice((19, 20, 21))
    grey = rng.choice((0, 0, 20, 40, 60))
    return RenderSpec(
        family=chosen,
        body_size=body,
        ink=(grey, grey, grey),
        date_style=rng.choice(DATE_STYLES[content.country]),
        number_style=rng.choice(NUMBER_STYLES[content.country]),
        currency_style=rng.choice(currency_styles),
        currency=currency,
    )


def family_by_name(name: str) -> FontFamily:
    return ALL_FAMILIES[name]
