"""Ground-truth content for invoices, receipts and bank statements.

Distributions are chosen to look like real paperwork rather than uniform noise:
line counts are 1 + Poisson (few lines dominate, long invoices are rare), prices are
log-normal around a catalogue base price scaled by local purchasing power, quantities
are mostly one with a geometric tail, and statement balances form a consistent running
chain. Every amount is a cent-quantised Decimal so arithmetic rules hold exactly.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from datetime import date, timedelta
from decimal import ROUND_HALF_UP, Decimal

from ledgerlens.generator import vendors
from ledgerlens.generator.identifiers import random_document_number, random_gstin, random_iban
from ledgerlens.schemas import DocType, DocumentFields, LineItem, Transaction

CENT = Decimal("0.01")
COUNTRIES: tuple[str, ...] = ("IN", "GB", "DE", "US")
COUNTRY_WEIGHTS: tuple[float, ...] = (0.4, 0.2, 0.2, 0.2)
TAX_RATES: dict[str, tuple[tuple[Decimal, str], ...]] = {
    "IN": (
        (Decimal("0.05"), "GST 5%"),
        (Decimal("0.12"), "GST 12%"),
        (Decimal("0.18"), "GST 18%"),
        (Decimal("0.28"), "GST 28%"),
    ),
    "GB": ((Decimal("0.20"), "VAT 20%"), (Decimal("0.05"), "VAT 5%")),
    "DE": ((Decimal("0.19"), "MwSt. 19%"), (Decimal("0.07"), "MwSt. 7%")),
    "US": (
        (Decimal("0.0825"), "Sales Tax 8.25%"),
        (Decimal("0.06"), "Sales Tax 6%"),
        (Decimal("0.0725"), "Sales Tax 7.25%"),
    ),
}
PAYMENT_METHODS: tuple[str, ...] = ("CASH", "CARD", "UPI", "CARD", "CASH")


@dataclass
class DocumentContent:
    """Fields plus the presentation details a template needs but the schema does not score."""

    doc_type: DocType
    country: str
    fields: DocumentFields
    line_items: list[LineItem] = field(default_factory=list)
    transactions: list[Transaction] = field(default_factory=list)
    extras: dict[str, str] = field(default_factory=dict)
    vendor_address: list[str] = field(default_factory=list)
    customer_address: list[str] = field(default_factory=list)


def money(value: float | Decimal) -> Decimal:
    """Quantise to cents with half-up rounding, the convention on printed documents."""
    return Decimal(str(value)).quantize(CENT, rounding=ROUND_HALF_UP)


def poisson(rng: random.Random, lam: float) -> int:
    """Knuth's Poisson sampler; small lambda only, which is all the generator needs."""
    limit = math.exp(-lam)
    count, product = 0, rng.random()
    while product > limit:
        count += 1
        product *= rng.random()
    return count


def pick_country(rng: random.Random) -> str:
    return rng.choices(COUNTRIES, weights=COUNTRY_WEIGHTS, k=1)[0]


def company_name(rng: random.Random, country: str) -> str:
    head = rng.choice(vendors.COMPANY_HEADS)
    trade = rng.choice(vendors.COMPANY_TRADES)
    return f"{head} {trade} {rng.choice(vendors.COMPANY_SUFFIX[country])}"


def address(rng: random.Random, country: str) -> vendors.Address:
    city, postcode = rng.choice(vendors.CITIES[country])
    return vendors.Address(rng.choice(vendors.STREETS[country]), city, postcode, country)


def random_date(rng: random.Random, start: date = date(2024, 1, 1), end: date = date(2026, 9, 30)) -> date:
    return start + timedelta(days=rng.randint(0, (end - start).days))


def make_line_items(rng: random.Random, currency: str, count: int) -> list[LineItem]:
    """Catalogue items with log-normal price jitter; services bill hours, goods bill units."""
    scale = vendors.PRICE_SCALE[currency]
    chosen = rng.sample(vendors.PRODUCTS, k=min(count, len(vendors.PRODUCTS)))
    items: list[LineItem] = []
    for description, base, is_service in chosen:
        unit_price = money(base * scale * rng.lognormvariate(0.0, 0.35))
        if is_service:
            quantity = Decimal(rng.choice((1, 2, 4, 8, 12, 20, 40)))
        else:
            quantity = Decimal(1 if rng.random() < 0.45 else rng.randint(2, 12))
        items.append(
            LineItem(
                description=description, quantity=quantity, unit_price=unit_price, amount=money(quantity * unit_price)
            )
        )
    return items


def totals(items: list[LineItem], rate: Decimal) -> tuple[Decimal, Decimal, Decimal]:
    """Subtotal, tax and total; tax is rounded once, as on a printed invoice."""
    subtotal = money(sum((item.amount for item in items), Decimal("0")))
    tax = money(subtotal * rate)
    return subtotal, tax, money(subtotal + tax)


def make_invoice(rng: random.Random) -> DocumentContent:
    """A B2B invoice: seller, buyer, number, dates, line items, tax, payment identifiers."""
    country = pick_country(rng)
    currency = vendors.CURRENCY[country]
    issued = random_date(rng)
    items = make_line_items(rng, currency, min(1 + poisson(rng, 2.6), 11))
    rate, tax_label = rng.choice(TAX_RATES[country])
    subtotal, tax, total = totals(items, rate)
    fields = DocumentFields(
        vendor_name=company_name(rng, country),
        customer_name=company_name(rng, country) if rng.random() < 0.7 else rng.choice(vendors.PERSON_NAMES),
        document_number=random_document_number(rng, "invoice", issued.year),
        document_date=issued,
        due_date=issued + timedelta(days=rng.choice((0, 7, 14, 15, 30, 45, 60))),
        currency=currency,
        subtotal=subtotal,
        tax=tax,
        total=total,
        gstin=random_gstin(rng) if country == "IN" and rng.random() < 0.9 else None,
        iban=random_iban(rng, country) if country in ("GB", "DE") and rng.random() < 0.85 else None,
    )
    extras = {
        "tax_label": tax_label,
        "terms": rng.choice(("Net 30", "Due on receipt", "Net 15", "Net 45")),
        "po_number": f"PO-{rng.randint(1000, 99999)}",
    }
    return DocumentContent(
        DocType.INVOICE,
        country,
        fields,
        items,
        [],
        extras,
        address(rng, country).lines(),
        address(rng, country).lines(),
    )


def make_receipt(rng: random.Random) -> DocumentContent:
    """A point-of-sale receipt: merchant, time, few items, tax, total, payment method."""
    country = pick_country(rng)
    currency = vendors.CURRENCY[country]
    scale = vendors.PRICE_SCALE[currency]
    picks = rng.sample(vendors.RETAIL_ITEMS, k=min(1 + poisson(rng, 3.0), 10))
    items: list[LineItem] = []
    for description, base in picks:
        unit_price = money(base * scale * rng.lognormvariate(0.0, 0.25))
        quantity = Decimal(1 if rng.random() < 0.6 else rng.randint(2, 6))
        items.append(
            LineItem(
                description=description, quantity=quantity, unit_price=unit_price, amount=money(quantity * unit_price)
            )
        )
    rate, tax_label = rng.choice(TAX_RATES[country])
    subtotal, tax, total = totals(items, rate)
    issued = random_date(rng)
    fields = DocumentFields(
        vendor_name=rng.choice(vendors.RETAIL_HEADS),
        document_number=random_document_number(rng, "receipt", issued.year) if rng.random() < 0.75 else None,
        document_date=issued,
        currency=currency,
        subtotal=subtotal,
        tax=tax,
        total=total,
        gstin=random_gstin(rng) if country == "IN" and rng.random() < 0.5 else None,
    )
    extras = {
        "tax_label": tax_label,
        "time": f"{rng.randint(7, 22):02d}:{rng.randint(0, 59):02d}",
        "payment": rng.choice(PAYMENT_METHODS),
        "cashier": f"{rng.randint(1, 9)}",
    }
    return DocumentContent(DocType.RECEIPT, country, fields, items, [], extras, address(rng, country).lines(), [])


def _transaction_description(rng: random.Random, country: str) -> tuple[str, str]:
    pattern, kind = rng.choice(vendors.TRANSACTION_TEMPLATES[country])
    text = pattern.format(
        ref=rng.randint(100000000, 999999999),
        ref4=rng.randint(1000, 9999),
        company=rng.choice(vendors.COMPANY_HEADS).upper(),
        person=rng.choice(vendors.PERSON_NAMES).upper(),
    )
    if kind == "either":
        kind = "debit" if rng.random() < 0.6 else "credit"
    return text, kind


def make_transactions(rng: random.Random, country: str, start: date, end: date, count: int) -> list[Transaction]:
    """Chronological rows with a consistent running balance that never goes negative."""
    scale = vendors.PRICE_SCALE[vendors.CURRENCY[country]]
    days = (end - start).days
    rows: list[tuple[date, str, str, Decimal]] = []
    for _ in range(count):
        text, kind = _transaction_description(rng, country)
        magnitude = money(scale * rng.lognormvariate(math.log(45.0 if kind == "debit" else 400.0), 0.9))
        rows.append((start + timedelta(days=rng.randint(0, days)), text, kind, magnitude))
    rows.sort(key=lambda row: row[0])
    running, lowest = Decimal("0"), Decimal("0")
    for _, _, kind, magnitude in rows:
        running += magnitude if kind == "credit" else -magnitude
        lowest = min(lowest, running)
    balance = money(scale * rng.lognormvariate(math.log(2500.0), 0.6)) - lowest
    transactions: list[Transaction] = []
    for day, text, kind, magnitude in rows:
        balance = money(balance + magnitude) if kind == "credit" else money(balance - magnitude)
        transactions.append(
            Transaction(
                date=day,
                description=text,
                balance=balance,
                debit=magnitude if kind == "debit" else None,
                credit=magnitude if kind == "credit" else None,
            )
        )
    return transactions


def make_statement(rng: random.Random) -> DocumentContent:
    """A one-month bank statement with opening and closing balances that reconcile."""
    country = pick_country(rng)
    anchor = random_date(rng, date(2024, 1, 1), date(2026, 8, 31))
    start = anchor.replace(day=1)
    end = (start.replace(month=start.month % 12 + 1, year=start.year + start.month // 12)) - timedelta(days=1)
    transactions = make_transactions(rng, country, start, end, rng.randint(6, 18))
    first = transactions[0]
    delta = (first.credit or Decimal("0")) - (first.debit or Decimal("0"))
    opening = money(first.balance - delta)
    iban = random_iban(rng, country) if country in ("GB", "DE") else None
    fields = DocumentFields(
        vendor_name=rng.choice(vendors.BANK_NAMES[country]),
        customer_name=rng.choice(vendors.PERSON_NAMES),
        document_number=random_document_number(rng, "statement", end.year),
        document_date=end + timedelta(days=rng.randint(1, 3)),
        period_start=start,
        period_end=end,
        currency=vendors.CURRENCY[country],
        opening_balance=opening,
        closing_balance=transactions[-1].balance,
        iban=iban,
    )
    account = f"{rng.randint(10**9, 10**11 - 1)}" if iban is None else ""
    return DocumentContent(
        DocType.BANK_STATEMENT,
        country,
        fields,
        [],
        transactions,
        {"account_number": account},
        address(rng, country).lines(),
        address(rng, country).lines(),
    )


def generate_content(rng: random.Random, doc_type: DocType) -> DocumentContent:
    """Dispatch on document type."""
    makers = {DocType.INVOICE: make_invoice, DocType.RECEIPT: make_receipt, DocType.BANK_STATEMENT: make_statement}
    return makers[doc_type](rng)
