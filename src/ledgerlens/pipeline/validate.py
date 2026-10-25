"""Accounting and sanity rules that confirm or contradict extracted values.

Rules are the cheapest source of truth available: arithmetic on a page must hold,
identifiers carry checksums, and dates must be plausible. A rule that passes raises
confidence in every field it touched; one that fails lowers it and forces review,
because an inconsistent record is worse than a missing one.
"""

from __future__ import annotations

from datetime import date, timedelta
from decimal import Decimal

from ledgerlens.generator.identifiers import gstin_is_valid, iban_is_valid
from ledgerlens.schemas import DocType, ExtractedField, LineItem, RuleResult, Transaction

CENT = Decimal("0.01")
KNOWN_CURRENCIES = frozenset({"INR", "GBP", "EUR", "USD"})
HARD_RULES = frozenset({"totals_reconcile", "iban_checksum", "gstin_checksum", "balance_chain", "line_items_sum"})


def _money(fields: dict[str, ExtractedField], name: str) -> Decimal | None:
    field = fields.get(name)
    return Decimal(field.value) if field is not None and field.value is not None else None


def _date(fields: dict[str, ExtractedField], name: str) -> date | None:
    field = fields.get(name)
    return date.fromisoformat(field.value) if field is not None and field.value is not None else None


def totals_reconcile(fields: dict[str, ExtractedField]) -> RuleResult:
    subtotal, tax, total = _money(fields, "subtotal"), _money(fields, "tax"), _money(fields, "total")
    names = ["subtotal", "tax", "total"]
    if subtotal is None or tax is None or total is None:
        return RuleResult(name="totals_reconcile", passed=None, detail="missing operand", fields=names)
    gap = abs(subtotal + tax - total)
    return RuleResult(
        name="totals_reconcile", passed=gap <= CENT, detail=f"subtotal + tax - total = {gap}", fields=names
    )


def line_items_sum(fields: dict[str, ExtractedField], items: list[LineItem]) -> RuleResult:
    subtotal = _money(fields, "subtotal")
    if subtotal is None or not items:
        return RuleResult(name="line_items_sum", passed=None, detail="no items or subtotal", fields=["subtotal"])
    total = sum((item.amount for item in items), Decimal("0"))
    gap = abs(total - subtotal)
    return RuleResult(
        name="line_items_sum", passed=gap <= CENT * len(items), detail=f"items - subtotal = {gap}", fields=["subtotal"]
    )


def line_item_arithmetic(items: list[LineItem]) -> RuleResult:
    if not items:
        return RuleResult(name="line_item_arithmetic", passed=None, detail="no items")
    bad = [item for item in items if abs(item.quantity * item.unit_price - item.amount) > CENT]
    return RuleResult(name="line_item_arithmetic", passed=not bad, detail=f"{len(bad)} of {len(items)} rows off")


def date_sanity(fields: dict[str, ExtractedField]) -> RuleResult:
    issued, due = _date(fields, "document_date"), _date(fields, "due_date")
    names = ["document_date"] + (["due_date"] if due is not None else [])
    if issued is None:
        return RuleResult(name="date_sanity", passed=None, detail="no date", fields=names)
    ok = date(2000, 1, 1) <= issued <= date.today() + timedelta(days=366)
    if due is not None:
        ok = ok and issued <= due <= issued + timedelta(days=180)
    return RuleResult(name="date_sanity", passed=ok, detail=f"issued {issued}, due {due}", fields=names)


def period_sanity(fields: dict[str, ExtractedField]) -> RuleResult:
    start, end = _date(fields, "period_start"), _date(fields, "period_end")
    names = ["period_start", "period_end"]
    if start is None or end is None:
        return RuleResult(name="period_sanity", passed=None, detail="no period", fields=names)
    span = (end - start).days
    return RuleResult(name="period_sanity", passed=0 <= span <= 92, detail=f"{span} days", fields=names)


def iban_checksum(fields: dict[str, ExtractedField]) -> RuleResult:
    field = fields.get("iban")
    if field is None or field.value is None:
        return RuleResult(name="iban_checksum", passed=None, detail="no iban", fields=["iban"])
    return RuleResult(name="iban_checksum", passed=iban_is_valid(field.value), detail=field.value, fields=["iban"])


def gstin_checksum(fields: dict[str, ExtractedField]) -> RuleResult:
    field = fields.get("gstin")
    if field is None or field.value is None:
        return RuleResult(name="gstin_checksum", passed=None, detail="no gstin", fields=["gstin"])
    return RuleResult(name="gstin_checksum", passed=gstin_is_valid(field.value), detail=field.value, fields=["gstin"])


def balance_chain(fields: dict[str, ExtractedField], transactions: list[Transaction]) -> RuleResult:
    opening, closing = _money(fields, "opening_balance"), _money(fields, "closing_balance")
    names = ["opening_balance", "closing_balance"]
    if opening is None or closing is None or not transactions:
        return RuleResult(name="balance_chain", passed=None, detail="incomplete chain", fields=names)
    running = opening
    broken = 0
    for row in transactions:
        running = running + (row.credit or Decimal("0")) - (row.debit or Decimal("0"))
        if abs(running - row.balance) > CENT:
            broken += 1
            running = row.balance
    closes = abs(transactions[-1].balance - closing) <= CENT
    return RuleResult(
        name="balance_chain",
        passed=broken == 0 and closes,
        detail=f"{broken} broken links, closing {'matches' if closes else 'differs'}",
        fields=names,
    )


def tax_rate_plausible(fields: dict[str, ExtractedField]) -> RuleResult:
    subtotal, tax = _money(fields, "subtotal"), _money(fields, "tax")
    if subtotal is None or tax is None or subtotal <= 0:
        return RuleResult(name="tax_rate_plausible", passed=None, detail="missing operand", fields=["tax"])
    rate = tax / subtotal
    return RuleResult(
        name="tax_rate_plausible",
        passed=Decimal("0") <= rate <= Decimal("0.35"),
        detail=f"rate {rate:.4f}",
        fields=["tax"],
    )


def currency_known(fields: dict[str, ExtractedField]) -> RuleResult:
    field = fields.get("currency")
    if field is None or field.value is None:
        return RuleResult(name="currency_known", passed=None, detail="no currency", fields=["currency"])
    return RuleResult(
        name="currency_known", passed=field.value in KNOWN_CURRENCIES, detail=field.value, fields=["currency"]
    )


def run_rules(
    doc_type: DocType, fields: dict[str, ExtractedField], items: list[LineItem], transactions: list[Transaction]
) -> list[RuleResult]:
    """All rules relevant to the document type, in a fixed order."""
    common = [date_sanity(fields), iban_checksum(fields), gstin_checksum(fields), currency_known(fields)]
    if doc_type is DocType.BANK_STATEMENT:
        return [period_sanity(fields), balance_chain(fields, transactions), *common]
    return [
        totals_reconcile(fields),
        line_items_sum(fields, items),
        line_item_arithmetic(items),
        tax_rate_plausible(fields),
        *common,
    ]
