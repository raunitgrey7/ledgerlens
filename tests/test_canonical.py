from datetime import date
from decimal import Decimal

from hypothesis import given
from hypothesis import strategies as st

from ledgerlens.canonical import (
    canonical_fields,
    canonical_identifier,
    canonical_money,
    canonical_name,
    canonical_value,
)
from ledgerlens.schemas import DocumentFields


def test_money_is_cent_precision_without_separators() -> None:
    assert canonical_money(Decimal("1234.5")) == "1234.50"
    assert canonical_money("1234.555") == "1234.56"
    assert canonical_money(7) == "7.00"


def test_names_lose_case_punctuation_and_spacing() -> None:
    assert canonical_name("  Meridian  Textiles, Pvt. Ltd ") == "MERIDIAN TEXTILES PVT LTD"
    assert canonical_name("Bean & Leaf") == "BEAN & LEAF"


def test_identifiers_and_dates() -> None:
    assert canonical_identifier("gb82 west 1234") == "GB82WEST1234"
    assert canonical_value("document_date", date(2026, 3, 14)) == "2026-03-14"
    assert canonical_value("currency", " inr ") == "INR"
    assert canonical_value("total", None) is None


@given(st.decimals(min_value=0, max_value=10**7, places=2, allow_nan=False, allow_infinity=False))
def test_canonical_money_is_idempotent(value: Decimal) -> None:
    once = canonical_money(value)
    assert canonical_money(once) == once


def test_canonical_fields_covers_every_field() -> None:
    fields = DocumentFields(vendor_name="X", total=Decimal("1.0"))
    out = canonical_fields(fields)
    assert set(out) == set(DocumentFields.model_fields)
    assert out["total"] == "1.00" and out["vendor_name"] == "X" and out["tax"] is None
