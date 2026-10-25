import random
from datetime import date
from decimal import Decimal

from ledgerlens.generator.content import generate_content, make_invoice, make_receipt, make_statement, poisson
from ledgerlens.schemas import DocType


def test_invoice_totals_reconcile_and_items_multiply() -> None:
    for seed in range(25):
        content = make_invoice(random.Random(seed))
        fields = content.fields
        assert fields.subtotal + fields.tax == fields.total
        assert sum(item.amount for item in content.line_items) == fields.subtotal
        for item in content.line_items:
            assert (item.quantity * item.unit_price).quantize(Decimal("0.01")) == item.amount
        assert fields.due_date >= fields.document_date
        assert date(2024, 1, 1) <= fields.document_date <= date(2026, 9, 30)


def test_receipt_has_items_and_consistent_totals() -> None:
    for seed in range(25):
        content = make_receipt(random.Random(seed))
        assert 1 <= len(content.line_items) <= 10
        assert content.fields.subtotal + content.fields.tax == content.fields.total
        assert content.fields.due_date is None


def test_statement_balance_chain_closes() -> None:
    for seed in range(25):
        content = make_statement(random.Random(seed))
        running = content.fields.opening_balance
        for row in content.transactions:
            assert (row.debit is None) != (row.credit is None)
            running += (row.credit or Decimal("0")) - (row.debit or Decimal("0"))
            assert running == row.balance
            assert row.balance >= 0
        assert running == content.fields.closing_balance
        assert content.fields.period_start.day == 1
        assert content.fields.period_start <= content.transactions[0].date <= content.fields.period_end


def test_identifiers_follow_country() -> None:
    for seed in range(40):
        content = make_invoice(random.Random(seed))
        if content.country == "IN":
            assert content.fields.iban is None
        if content.country in ("US",):
            assert content.fields.gstin is None and content.fields.iban is None


def test_generation_is_deterministic_per_seed() -> None:
    a = generate_content(random.Random(99), DocType.INVOICE)
    b = generate_content(random.Random(99), DocType.INVOICE)
    assert a.fields == b.fields and a.line_items == b.line_items


def test_poisson_sampler_mean_is_close_to_lambda() -> None:
    rng = random.Random(0)
    samples = [poisson(rng, 2.6) for _ in range(4000)]
    assert abs(sum(samples) / len(samples) - 2.6) < 0.15
