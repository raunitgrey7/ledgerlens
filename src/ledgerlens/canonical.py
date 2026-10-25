"""Canonical string forms used to compare gold values with predictions.

Both sides of every comparison pass through these functions, so a match means the
same information, not the same formatting. Money becomes a cent-precision decimal
string, dates become ISO 8601, identifiers lose spaces, and names lose case and
spacing noise. Keeping this in one module stops the metric code and the API from
drifting apart in what they call "correct".
"""

from __future__ import annotations

import re
from datetime import date
from decimal import ROUND_HALF_UP, Decimal

from ledgerlens.schemas import DATE_FIELDS, IDENTIFIER_FIELDS, MONEY_FIELDS, DocumentFields

CENT = Decimal("0.01")
_SPACES = re.compile(r"\s+")
_NAME_STRIP = re.compile(r"[^A-Z0-9&' ]")


def canonical_money(value: Decimal | str | float | int) -> str:
    """Cent-precision decimal string without thousands separators, e.g. '1234.50'."""
    return str(Decimal(str(value)).quantize(CENT, rounding=ROUND_HALF_UP))


def canonical_date(value: date | str) -> str:
    """ISO 8601 date string."""
    return value if isinstance(value, str) else value.isoformat()


def canonical_identifier(value: str) -> str:
    """Upper-case identifier with all whitespace removed."""
    return _SPACES.sub("", value).upper()


def canonical_name(value: str) -> str:
    """Upper-case name with punctuation noise removed and single spaces."""
    upper = _NAME_STRIP.sub(" ", value.upper())
    return _SPACES.sub(" ", upper).strip()


def canonical_value(field: str, value: object) -> str | None:
    """Dispatch to the canonical form appropriate for ``field``; None stays None."""
    if value is None:
        return None
    if field in MONEY_FIELDS:
        return canonical_money(value)  # type: ignore[arg-type]
    if field in DATE_FIELDS:
        return canonical_date(value)  # type: ignore[arg-type]
    if field in IDENTIFIER_FIELDS or field == "document_number":
        return canonical_identifier(str(value))
    if field == "currency":
        return str(value).upper().strip()
    return canonical_name(str(value))


def canonical_fields(fields: DocumentFields) -> dict[str, str | None]:
    """Canonical form of every field of a ``DocumentFields`` record."""
    return {name: canonical_value(name, getattr(fields, name)) for name in type(fields).model_fields}


def canonical_description(value: str) -> str:
    """Line-item descriptions compare like names."""
    return canonical_name(value)
