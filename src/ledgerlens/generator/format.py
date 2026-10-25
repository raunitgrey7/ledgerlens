"""Locale-aware printing of money, quantities and dates.

Real documents disagree on separators (1,234.56 against 1.234,56 against 1,23,456.00)
and on date order. The generator reproduces that variety on purpose so the parser in
the pipeline is scored on it; the parser's own rules live in ``pipeline.parse``.
"""

from __future__ import annotations

from datetime import date
from decimal import Decimal

from ledgerlens.generator.vendors import CURRENCY_SYMBOL

MONTHS: tuple[str, ...] = (
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
)

DATE_STYLES: dict[str, tuple[str, ...]] = {
    "IN": ("dmy_slash", "dmy_mon", "dmy_long", "iso"),
    "GB": ("dmy_slash", "dmy_long", "iso"),
    "DE": ("dmy_dot", "iso"),
    "US": ("mdy_slash", "mdy_long", "iso"),
}
NUMBER_STYLES: dict[str, tuple[str, ...]] = {
    "IN": ("west", "lakh", "west"),
    "GB": ("west",),
    "DE": ("de", "de", "west"),
    "US": ("west",),
}


def group_thousands(integer: str, style: str) -> str:
    """Insert separators: western 3-3-3, Indian 3-2-2, or German dots."""
    if style == "lakh":
        if len(integer) <= 3:
            return integer
        head, tail = integer[:-3], integer[-3:]
        parts: list[str] = []
        while len(head) > 2:
            parts.insert(0, head[-2:])
            head = head[:-2]
        return ",".join([head, *parts, tail])
    separator = "." if style == "de" else ","
    return f"{int(integer):,}".replace(",", separator)


def fmt_number(value: Decimal, style: str) -> str:
    """Two-decimal money string in the given separator style."""
    sign = "-" if value < 0 else ""
    integer, cents = f"{abs(value):.2f}".split(".")
    grouped = group_thousands(integer, style)
    decimal_sep = "," if style == "de" else "."
    return f"{sign}{grouped}{decimal_sep}{cents}"


def fmt_money(value: Decimal, currency: str, style: str, currency_style: str) -> str:
    """Money with an optional symbol or ISO code prefix."""
    number = fmt_number(value, style)
    if currency_style == "symbol":
        return f"{CURRENCY_SYMBOL[currency]}{number}"
    if currency_style == "code":
        return f"{currency} {number}"
    return number


def fmt_qty(value: Decimal) -> str:
    """Integers print without a decimal part."""
    return str(int(value)) if value == value.to_integral_value() else f"{value.normalize()}"


def fmt_date(value: date, style: str) -> str:
    """Render a date in one of the supported house styles."""
    if style == "dmy_slash":
        return value.strftime("%d/%m/%Y")
    if style == "dmy_dot":
        return value.strftime("%d.%m.%Y")
    if style == "dmy_mon":
        return f"{value.day:02d}-{MONTHS[value.month - 1][:3]}-{value.year}"
    if style == "dmy_long":
        return f"{value.day} {MONTHS[value.month - 1]} {value.year}"
    if style == "mdy_slash":
        return value.strftime("%m/%d/%Y")
    if style == "mdy_long":
        return f"{MONTHS[value.month - 1]} {value.day}, {value.year}"
    return value.isoformat()
