from datetime import date
from decimal import Decimal

from ledgerlens.pipeline.validate import (
    balance_chain,
    currency_known,
    date_sanity,
    gstin_checksum,
    iban_checksum,
    line_item_arithmetic,
    line_items_sum,
    period_sanity,
    run_rules,
    tax_rate_plausible,
    totals_reconcile,
)
from ledgerlens.schemas import DocType, ExtractedField, LineItem, Transaction


def _fields(**values: str | None) -> dict[str, ExtractedField]:
    return {name: ExtractedField(name=name, value=value, confidence=0.9) for name, value in values.items()}


def test_totals_rule_outcomes() -> None:
    assert totals_reconcile(_fields(subtotal="100.00", tax="18.00", total="118.00")).passed is True
    assert totals_reconcile(_fields(subtotal="100.00", tax="18.00", total="119.00")).passed is False
    assert totals_reconcile(_fields(subtotal="100.00", tax="18.00")).passed is None


def test_line_item_rules() -> None:
    items = [
        LineItem(description="a", quantity=Decimal(2), unit_price=Decimal("5.00"), amount=Decimal("10.00")),
        LineItem(description="b", quantity=Decimal(1), unit_price=Decimal("2.50"), amount=Decimal("2.50")),
    ]
    assert line_items_sum(_fields(subtotal="12.50"), items).passed is True
    assert line_items_sum(_fields(subtotal="13.50"), items).passed is False
    assert line_items_sum(_fields(subtotal="12.50"), []).passed is None
    assert line_item_arithmetic(items).passed is True
    bad = [LineItem(description="c", quantity=Decimal(3), unit_price=Decimal("1.00"), amount=Decimal("4.00"))]
    assert line_item_arithmetic(bad).passed is False


def test_date_and_period_rules() -> None:
    assert date_sanity(_fields(document_date="2026-03-14", due_date="2026-04-13")).passed is True
    assert date_sanity(_fields(document_date="2026-03-14", due_date="2026-03-01")).passed is False
    assert date_sanity(_fields(document_date="1999-01-01")).passed is False
    assert date_sanity(_fields()).passed is None
    assert period_sanity(_fields(period_start="2026-03-01", period_end="2026-03-31")).passed is True
    assert period_sanity(_fields(period_start="2026-03-01", period_end="2026-09-30")).passed is False


def test_checksum_rules() -> None:
    assert iban_checksum(_fields(iban="GB82WEST12345698765432")).passed is True
    assert iban_checksum(_fields(iban="GB82WEST12345698765433")).passed is False
    assert gstin_checksum(_fields(gstin="27AAPFU0939F1ZV")).passed is True
    assert gstin_checksum(_fields(gstin="27AAPFU0939F1ZA")).passed is False
    assert gstin_checksum(_fields()).passed is None


def test_balance_chain_rule() -> None:
    rows = [
        Transaction(date=date(2026, 3, 2), description="a", debit=Decimal("10.00"), balance=Decimal("90.00")),
        Transaction(date=date(2026, 3, 5), description="b", credit=Decimal("50.00"), balance=Decimal("140.00")),
    ]
    assert balance_chain(_fields(opening_balance="100.00", closing_balance="140.00"), rows).passed is True
    assert balance_chain(_fields(opening_balance="100.00", closing_balance="141.00"), rows).passed is False
    rows[1] = Transaction(date=date(2026, 3, 5), description="b", credit=Decimal("50.00"), balance=Decimal("150.00"))
    assert balance_chain(_fields(opening_balance="100.00", closing_balance="150.00"), rows).passed is False


def test_tax_rate_and_currency_rules() -> None:
    assert tax_rate_plausible(_fields(subtotal="100.00", tax="18.00")).passed is True
    assert tax_rate_plausible(_fields(subtotal="100.00", tax="80.00")).passed is False
    assert currency_known(_fields(currency="INR")).passed is True
    assert currency_known(_fields(currency="XYZ")).passed is False


def test_run_rules_depends_on_document_type() -> None:
    names = {r.name for r in run_rules(DocType.INVOICE, _fields(), [], [])}
    assert "totals_reconcile" in names and "balance_chain" not in names
    names = {r.name for r in run_rules(DocType.BANK_STATEMENT, _fields(), [], [])}
    assert "balance_chain" in names and "totals_reconcile" not in names
