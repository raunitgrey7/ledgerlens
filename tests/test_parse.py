import random
from datetime import date
from decimal import Decimal

from hypothesis import given, settings
from hypothesis import strategies as st

from ledgerlens.generator.format import fmt_date, fmt_number
from ledgerlens.generator.identifiers import random_gstin, random_iban
from ledgerlens.pipeline.parse import (
    parse_currency,
    parse_date_text,
    parse_dates,
    parse_gstin,
    parse_iban,
    parse_money,
    parse_quantity,
)


def test_money_separator_styles() -> None:
    assert parse_money("1,234.56") == Decimal("1234.56")
    assert parse_money("1.234,56") == Decimal("1234.56")
    assert parse_money("1,23,456.00") == Decimal("123456.00")
    assert parse_money("-45.10") == Decimal("-45.10")
    assert parse_money("0.95") == Decimal("0.95")


def test_money_strips_currency_and_repairs_ocr() -> None:
    assert parse_money("INR 7,37,321.11") == Decimal("737321.11")
    assert parse_money("₹4,311.04") == Decimal("4311.04")
    assert parse_money("USD 392.86") == Decimal("392.86")
    assert parse_money("1O,5O0.OO") == Decimal("10500.00")
    assert parse_money("S5.85") == Decimal("55.85")


def test_money_rejects_non_amounts() -> None:
    for token in ("12", "Total", "2024-10-05", "18%", "", "1,2,3"):
        assert parse_money(token) is None


def test_quantity_and_currency() -> None:
    assert parse_quantity("12") == Decimal("12")
    assert parse_quantity("2x") == Decimal("2")
    assert parse_quantity("2.5") == Decimal("2.5")
    assert parse_quantity("l") == Decimal("1")
    assert parse_quantity("abc") is None
    assert parse_currency("£") == "GBP" and parse_currency("eur") == "EUR" and parse_currency("Qty") is None


def test_date_styles() -> None:
    assert parse_date_text("14/03/2026") == date(2026, 3, 14)
    assert parse_date_text("03/14/2026", prefer_mdy=True) == date(2026, 3, 14)
    assert parse_date_text("14/03/2026", prefer_mdy=True) == date(2026, 3, 14)
    assert parse_date_text("2026-03-14") == date(2026, 3, 14)
    assert parse_date_text("14.03.2026") == date(2026, 3, 14)
    assert parse_date_text("14-Mar-2026") == date(2026, 3, 14)
    assert parse_date_text("14 March 2026") == date(2026, 3, 14)
    assert parse_date_text("March 14, 2026") == date(2026, 3, 14)
    assert parse_date_text("Invoice Date: 27-Apr-2025") == date(2025, 4, 27)
    assert parse_date_text("l4/O3/2026") == date(2026, 3, 14)
    assert parse_date_text("no date here") is None
    assert parse_date_text("31/02/2026") is None


def test_parse_dates_finds_two_dates_around_a_dash() -> None:
    found = parse_dates("Period 1 March 2026 - 31 March 2026")
    assert found == [date(2026, 3, 1), date(2026, 3, 31)]
    assert parse_dates("Statement Period: 01/03/2026 to 31/03/2026") == [date(2026, 3, 1), date(2026, 3, 31)]


@given(st.integers(min_value=0, max_value=10**6), st.sampled_from(["west", "de", "lakh"]))
@settings(max_examples=120)
def test_formatted_money_round_trips(cents: int, style: str) -> None:
    value = Decimal(cents) / Decimal(100)
    assert parse_money(fmt_number(value, style)) == value.quantize(Decimal("0.01"))


@given(
    st.dates(min_value=date(2001, 1, 1), max_value=date(2034, 12, 31)),
    st.sampled_from(["dmy_slash", "dmy_dot", "dmy_mon", "dmy_long", "mdy_slash", "mdy_long", "iso"]),
)
@settings(max_examples=120)
def test_formatted_dates_round_trip(value: date, style: str) -> None:
    assert parse_date_text(fmt_date(value, style), prefer_mdy=style.startswith("mdy")) == value


def test_gstin_repair_and_alternatives() -> None:
    gstin = random_gstin(random.Random(8))
    assert parse_gstin(gstin.replace("0", "O").replace("1", "I")) == gstin
    assert parse_gstin(gstin.lower()) == gstin
    assert parse_gstin(gstin[:-1] + ("A" if gstin[-1] != "A" else "B")) is None
    assert parse_gstin("too short") is None


def test_iban_repair() -> None:
    gb = random_iban(random.Random(2), "GB")
    de = random_iban(random.Random(2), "DE")
    assert parse_iban(gb.replace("1", "l")) == gb
    assert parse_iban(" ".join(de[i : i + 4] for i in range(0, len(de), 4)).replace("0", "O")) == de
    assert parse_iban(de[:-1] + str((int(de[-1]) + 1) % 10)) is None
    assert parse_iban("GB82") is None
