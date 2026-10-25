"""Field, line-item and transaction extraction from OCR rows.

The extractor is a set of anchored searches rather than a learned model so that every
value it returns can be explained by the label it was found next to. Header fields
are anchored on fuzzy labels; line items on a table header; receipt items on the
"qty x price" conventions of till printers; transactions on a statement table header.
Document type is decided first from vocabulary and paper shape because the label
lexicons differ per type.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from decimal import Decimal

import numpy as np

from ledgerlens.canonical import canonical_value
from ledgerlens.pipeline.layout import (
    Column,
    LabelMatch,
    Row,
    TableHeader,
    find_label,
    find_table_header,
    line_below,
    normalise,
    similarity,
    strip_separators,
    words_below,
)
from ledgerlens.pipeline.ocr.base import OCRPage, OCRWord
from ledgerlens.pipeline.parse import (
    parse_currency,
    parse_date_text,
    parse_dates,
    parse_gstin,
    parse_iban,
    parse_money,
    parse_quantity,
)
from ledgerlens.schemas import DocType, ExtractedField, LineItem, Transaction

DOC_TYPE_CUES: dict[DocType, tuple[tuple[str, float], ...]] = {
    DocType.INVOICE: (
        ("invoice", 2.0),
        ("tax invoice", 2.0),
        ("bill to", 1.5),
        ("billed to", 1.5),
        ("due date", 1.0),
        ("payment due", 1.0),
        ("amount due", 1.0),
        ("invoice number", 1.0),
        ("payment terms", 1.0),
    ),
    DocType.RECEIPT: (
        ("receipt", 2.0),
        ("thank you", 1.0),
        ("cash", 1.0),
        ("change", 1.0),
        ("paid by", 1.5),
        ("payment", 0.5),
        ("visit again", 1.0),
        ("cashier", 1.5),
        ("order", 0.5),
        ("see you soon", 1.0),
    ),
    DocType.BANK_STATEMENT: (
        ("statement", 2.0),
        ("opening balance", 2.0),
        ("closing balance", 2.0),
        ("brought forward", 2.0),
        ("carried forward", 2.0),
        ("money out", 1.5),
        ("money in", 1.5),
        ("debit", 1.0),
        ("credit", 1.0),
        ("balance", 1.0),
    ),
}
TITLE_WORDS: tuple[str, ...] = (
    "invoice",
    "tax invoice",
    "account statement",
    "your statement",
    "receipt",
    "statement",
    "for",
)
NUMBER_LABELS: tuple[str, ...] = (
    "invoice no",
    "invoice no.",
    "invoice number",
    "invoice #",
    "receipt no",
    "receipt #",
    "receipt number",
    "order",
    "statement no",
    "statement number",
    "bill no",
    "invoice",
)
NUMBER_EXCLUDE: tuple[str, ...] = (
    "invoice date",
    "due date",
    "issue date",
    "statement date",
    "statement period",
    "tax invoice",
    "payment terms",
)
DATE_LABELS: tuple[str, ...] = ("invoice date", "issue date", "statement date", "date", "issued", "dated", "date:")
DATE_EXCLUDE: tuple[str, ...] = ("due date", "payment due", "period", "statement period")
DUE_LABELS: tuple[str, ...] = ("due date", "payment due", "due")
SUBTOTAL_LABELS: tuple[str, ...] = (
    "subtotal",
    "sub total",
    "sub-total",
    "taxable value",
    "taxable amount",
    "net amount",
    "net",
)
TAX_LABELS: tuple[str, ...] = ("tax", "gst", "vat", "mwst", "mwst.", "sales tax", "total tax")
TAX_PART_LABELS: tuple[str, ...] = ("cgst", "sgst", "igst")
TOTAL_LABELS: tuple[str, ...] = ("total", "grand total", "amount due", "balance due", "total due", "total amount")
TOTAL_EXCLUDE: tuple[str, ...] = ("sub total", "subtotal", "total tax", "taxable", "line total", "sub")
OPENING_LABELS: tuple[str, ...] = ("opening balance", "balance brought forward", "brought forward")
CLOSING_LABELS: tuple[str, ...] = ("closing balance", "balance carried forward", "carried forward")
PERIOD_LABELS: tuple[str, ...] = ("statement period", "period")
CUSTOMER_LABELS: tuple[str, ...] = ("bill to", "billed to", "customer", "sold to", "bill to:")
CUSTOMER_EXCLUDE: tuple[str, ...] = ("ship to",)
CURRENCY_LABELS: tuple[str, ...] = ("currency", "all amounts in")
GSTIN_LABELS: tuple[str, ...] = ("gstin", "gst no", "gst no.", "gstin:")
IBAN_LABELS: tuple[str, ...] = ("iban", "iban:")
ITEM_COLUMNS: dict[str, tuple[str, ...]] = {
    "description": (
        "description",
        "description of goods",
        "item",
        "particulars",
        "service / product",
        "service product",
        "details",
    ),
    "quantity": ("qty", "quantity", "units"),
    "unit_price": ("unit price", "price", "rate", "unit cost"),
    "amount": ("amount", "total", "line total"),
}
TRANSACTION_COLUMNS: dict[str, tuple[str, ...]] = {
    "date": ("date",),
    "description": ("description", "details", "particulars", "narration"),
    "debit": ("debit", "money out", "withdrawals", "withdrawal", "paid out"),
    "credit": ("credit", "money in", "deposits", "paid in"),
    "balance": ("balance",),
}
STOP_LABELS: tuple[str, ...] = (
    SUBTOTAL_LABELS
    + TAX_LABELS
    + TAX_PART_LABELS
    + TOTAL_LABELS
    + CLOSING_LABELS
    + ("payment", "paid by", "cash", "thank you")
)
_HAS_DIGIT = re.compile(r"\d")
_NUMBER_REPAIRS: dict[str, str] = {"O": "0", "o": "0", "I": "1", "l": "1", "|": "1"}
_LETTER_NEXT_TO_DIGIT = re.compile(r"(?<=\d)[OoIl|]|[OoIl|](?=\d)")
_PERCENT = re.compile(r"^\d{1,2}(?:[.,]\d{1,2})?%$")
_NAME_TOKEN = re.compile(r"^[A-Z][a-zA-Z'&.-]+$")
LABEL_THRESHOLD = 0.8
RECEIPT_NUMBER_LABELS: tuple[str, ...] = ("receipt no", "receipt #", "receipt number", "order")
LABELLED_OPTIONAL: tuple[tuple[str, tuple[str, ...]], ...] = (("gstin", GSTIN_LABELS), ("iban", IBAN_LABELS))
CODE_VOTE = 2.0
ARITHMETIC_FACTOR = 0.9
DETACHED_GAP = 4.0
SYMBOL_VOTE = 1.0
IDENTIFIER_VOTE = 2.0


@dataclass
class Extraction:
    """Raw extraction before validation and confidence scoring."""

    doc_type: DocType
    doc_type_confidence: float
    fields: dict[str, ExtractedField]
    line_items: list[LineItem]
    transactions: list[Transaction]
    flagged: list[str] = dataclass_field(default_factory=list)


def classify(rows: list[Row], page: OCRPage) -> tuple[DocType, float]:
    """Vocabulary votes per type plus a paper-shape prior for narrow receipts."""
    scores: dict[DocType, float] = dict.fromkeys(DocType, 0.0)
    for kind, cues in DOC_TYPE_CUES.items():
        for phrase, weight in cues:
            if find_label(rows, (phrase,), threshold=0.85):
                scores[kind] += weight
    if page.width < 700 and page.height > 1.3 * page.width:
        scores[DocType.RECEIPT] += 2.5
    best = max(scores, key=lambda kind: scores[kind])
    total = sum(scores.values())
    confidence = scores[best] / total if total > 0 else 0.34
    return best, round(confidence, 3)


def extract(page: OCRPage, rows: list[Row]) -> Extraction:
    """Dispatch on the classified document type."""
    doc_type, confidence = classify(rows, page)
    fields: dict[str, ExtractedField] = {}
    line_items: list[LineItem] = []
    transactions: list[Transaction] = []
    _put(fields, "vendor_name", *vendor_name(rows))
    if doc_type is not DocType.BANK_STATEMENT:
        _put(fields, "gstin", *identifier_field(rows, GSTIN_LABELS, parse_gstin, 15))
    _put(fields, "iban", *identifier_field(rows, IBAN_LABELS, parse_iban, 18))
    _put(fields, "currency", *currency(rows, fields))
    prefer_mdy = fields["currency"].value == "USD" if "currency" in fields else False
    if doc_type is DocType.BANK_STATEMENT:
        _extract_statement_fields(rows, fields, prefer_mdy)
        transactions = extract_transactions(rows, page, prefer_mdy)
    else:
        _extract_trade_fields(rows, fields, doc_type, prefer_mdy)
        line_items = extract_line_items(rows, page) if doc_type is DocType.INVOICE else extract_receipt_items(rows)
    return Extraction(
        doc_type, confidence, fields, line_items, transactions, labelled_but_missing(rows, fields, doc_type)
    )


def labelled_but_missing(rows: list[Row], fields: dict[str, ExtractedField], doc_type: DocType) -> list[str]:
    """Optional fields whose label is printed but whose value was not read: an OCR miss, not an absence."""
    flagged = [name for name, labels in LABELLED_OPTIONAL if name not in fields and find_label(rows, labels, 0.85)]
    if (
        doc_type is DocType.RECEIPT
        and "document_number" not in fields
        and find_label(rows, RECEIPT_NUMBER_LABELS, 0.85, NUMBER_EXCLUDE)
    ):
        flagged.append("document_number")
    return flagged


def _put(fields: dict[str, ExtractedField], name: str, value: object, confidence: float, evidence: str) -> None:
    if value is None:
        return
    fields[name] = ExtractedField(
        name=name, value=canonical_value(name, value), confidence=round(confidence, 4), evidence=evidence
    )


def _extract_trade_fields(
    rows: list[Row], fields: dict[str, ExtractedField], doc_type: DocType, prefer_mdy: bool
) -> None:
    _put(fields, "document_number", *document_number(rows))
    _put(fields, "document_date", *date_field(rows, DATE_LABELS, DATE_EXCLUDE, prefer_mdy, fallback=True))
    if doc_type is DocType.INVOICE:
        _put(fields, "due_date", *date_field(rows, DUE_LABELS, (), prefer_mdy))
        _put(fields, "customer_name", *customer_name(rows))
    _put(fields, "subtotal", *money_field(rows, SUBTOTAL_LABELS, ()))
    _put(fields, "tax", *tax_field(rows))
    _put(fields, "total", *money_field(rows, TOTAL_LABELS, TOTAL_EXCLUDE, prefer_last=True))
    recover_by_arithmetic(rows, fields)


def recover_by_arithmetic(rows: list[Row], fields: dict[str, ExtractedField]) -> None:
    """Fill one missing amount of subtotal + tax = total with a printed amount that satisfies the sum.

    A misread label ("Tot:d") hides a value the OCR read correctly. When exactly one of
    the three is missing, the amount it must have is known, and it is accepted only if
    that amount is actually printed on the page, so nothing is invented.
    """
    names = ("subtotal", "tax", "total")
    missing = [name for name in names if name not in fields or fields[name].value is None]
    if len(missing) != 1:
        return
    known = {name: Decimal(fields[name].value) for name in names if name not in missing}  # type: ignore[arg-type]
    target = {
        "total": lambda: known["subtotal"] + known["tax"],
        "subtotal": lambda: known["total"] - known["tax"],
        "tax": lambda: known["total"] - known["subtotal"],
    }[missing[0]]()
    for word in (w for row in rows for w in row.words):
        amount = parse_money(word.text)
        if amount is not None and amount == target and target > 0:
            support = min(fields[name].confidence for name in names if name not in missing)
            _put(
                fields,
                missing[0],
                amount,
                min(word.confidence, support) * ARITHMETIC_FACTOR,
                "printed amount matching the sum",
            )
            return


def _extract_statement_fields(rows: list[Row], fields: dict[str, ExtractedField], prefer_mdy: bool) -> None:
    _put(fields, "document_number", *document_number(rows))
    _put(fields, "document_date", *date_field(rows, DATE_LABELS, DATE_EXCLUDE, prefer_mdy))
    _put(fields, "customer_name", *statement_holder(rows))
    start, end, confidence, evidence = period(rows, prefer_mdy)
    _put(fields, "period_start", start, confidence, evidence)
    _put(fields, "period_end", end, confidence, evidence)
    _put(fields, "opening_balance", *money_field(rows, OPENING_LABELS, ()))
    _put(fields, "closing_balance", *money_field(rows, CLOSING_LABELS, (), prefer_last=True))


def value_words(rows: list[Row], match: LabelMatch) -> list[OCRWord]:
    """Words to the right of a label, or below it when the label stands alone (stacked layout)."""
    rest = strip_separators(match.rest)
    if rest:
        return rest
    x0, x1 = match.words[0].x0, match.words[-1].x1
    return strip_separators(words_below(rows, match.row_index, x0, x1))


def value_options(rows: list[Row], match: LabelMatch) -> list[list[OCRWord]]:
    """Both places a value can live: right of the label, then directly below it."""
    x0, x1 = match.words[0].x0, match.words[-1].x1
    options = [strip_separators(match.rest), strip_separators(words_below(rows, match.row_index, x0, x1))]
    return [option for option in options if option]


def _first_amount(words: list[OCRWord]) -> tuple[Decimal, OCRWord] | None:
    """The right-most token that parses as money, with its word."""
    for word in reversed(words):
        amount = parse_money(word.text)
        if amount is not None:
            return amount, word
    joined = "".join(w.text for w in words)
    amount = parse_money(joined)
    if amount is not None and words:
        return amount, min(words, key=lambda w: w.confidence)
    return None


def money_field(
    rows: list[Row], labels: tuple[str, ...], exclude: tuple[str, ...], prefer_last: bool = False
) -> tuple[Decimal | None, float, str]:
    """Best labelled amount: the lowest matching row when ``prefer_last`` (totals), else the best label score."""
    best: tuple[float, Decimal, float, str] | None = None
    for match in find_label(rows, labels, LABEL_THRESHOLD, exclude):
        found = next((f for f in (_first_amount(option) for option in value_options(rows, match)) if f), None)
        if found is None:
            continue
        amount, word = found
        rank = match.row_index + match.score if prefer_last else match.score - 0.001 * match.row_index
        if best is None or rank > best[0]:
            confidence = word.confidence * (0.7 + 0.3 * match.score)
            best = (rank, amount, confidence, f"label '{' '.join(w.text for w in match.words)}' row {match.row_index}")
    if best is None:
        return None, 0.0, ""
    return best[1], best[2], best[3]


def tax_field(rows: list[Row]) -> tuple[Decimal | None, float, str]:
    """A single tax line, the sum of CGST and SGST rows, or any row with a percentage and an amount."""
    parts: dict[int, tuple[Decimal, float]] = {}
    for match in find_label(rows, TAX_PART_LABELS, 0.9):
        found = _first_amount(value_words(rows, match))
        if found is not None and match.row_index not in parts:
            parts[match.row_index] = (found[0], found[1].confidence)
    if len(parts) >= 2:
        chosen = [parts[index] for index in sorted(parts)[:2]]
        return sum((amount for amount, _ in chosen), Decimal("0")), min(c for _, c in chosen), "sum of cgst and sgst"
    labelled = money_field(rows, TAX_LABELS, (*TOTAL_EXCLUDE, "taxable"))
    if labelled[0] is not None:
        return labelled
    for index, row in enumerate(rows):
        words = row.words
        if len(words) >= 2 and _PERCENT.match(words[-2].text) and not _is_total_row(row):
            amount = parse_money(words[-1].text)
            if amount is not None:
                return amount, words[-1].confidence * 0.85, f"percentage row {index}"
    return None, 0.0, ""


def _is_total_row(row: Row) -> bool:
    head = " ".join(row.text.split()[:2])
    return any(similarity(head, phrase) >= 0.85 for phrase in TOTAL_LABELS + SUBTOTAL_LABELS)


def date_field(
    rows: list[Row], labels: tuple[str, ...], exclude: tuple[str, ...], prefer_mdy: bool, fallback: bool = False
) -> tuple[object, float, str]:
    """Labelled date; with ``fallback`` the first parseable date in the top third of the page."""
    for match in find_label(rows, labels, LABEL_THRESHOLD, exclude):
        for words in value_options(rows, match):
            text = " ".join(w.text for w in words[:4])
            found = parse_date_text(text, prefer_mdy)
            if found is None:
                continue
            digits = [w for w in words[:4] if _HAS_DIGIT.search(w.text)]
            confidence = min((w.confidence for w in digits), default=0.5) * (0.7 + 0.3 * match.score)
            return found, confidence, f"label '{' '.join(w.text for w in match.words)}'"
    if fallback:
        for row in rows[: max(3, len(rows) // 3)]:
            found = parse_date_text(row.text, prefer_mdy)
            if found is not None:
                digits = [w for w in row.words if _HAS_DIGIT.search(w.text)]
                return found, 0.6 * min((w.confidence for w in digits), default=0.5), "unlabelled date"
    return None, 0.0, ""


def document_number(rows: list[Row]) -> tuple[str | None, float, str]:
    """First token with a digit after a number label; the bare 'Invoice <number>' title also counts."""
    for match in find_label(rows, NUMBER_LABELS, LABEL_THRESHOLD, NUMBER_EXCLUDE):
        for words in value_options(rows, match):
            for word in words[:2]:
                token = repair_number(word.text.strip(":#"))
                if _HAS_DIGIT.search(token) and parse_money(token) is None and len(token) >= 4:
                    return token, word.confidence * (0.7 + 0.3 * match.score), f"label '{match.words[0].text}'"
    return None, 0.0, ""


def repair_number(token: str) -> str:
    """Letters that touch a digit inside a document number are digits misread: RO79423 is R079423."""
    previous = None
    while previous != token:
        previous = token
        token = _LETTER_NEXT_TO_DIGIT.sub(lambda m: _NUMBER_REPAIRS[m.group(0)], token)
    return token


def identifier_field(
    rows: list[Row], labels: tuple[str, ...], parser: object, min_len: int
) -> tuple[str | None, float, str]:
    """Checksummed identifier next to its label, else anywhere on the page."""
    for match in find_label(rows, labels, LABEL_THRESHOLD):
        words = value_words(rows, match)
        candidates = ["".join(w.text for w in words[:6])] + [w.text for w in words[:2]]
        for candidate in candidates:
            parsed = parser(candidate)  # type: ignore[operator]
            if parsed is not None:
                confidence = min((w.confidence for w in words[:6]), default=0.5)
                return parsed, max(confidence, 0.9), f"label '{match.words[0].text}', checksum ok"
    for row in rows:
        for word in row.words:
            if len(word.text) >= min_len:
                parsed = parser(word.text)  # type: ignore[operator]
                if parsed is not None:
                    return parsed, max(word.confidence, 0.85), "unlabelled, checksum ok"
    return None, 0.0, ""


def vendor_name(rows: list[Row]) -> tuple[str | None, float, str]:
    """The top-most block-local line that is not merely a document title, with title words removed."""
    lines = [line for row in rows[:6] for line in row.lines]
    for line in sorted(lines, key=lambda line: (line.y0, line.x0)):
        kept = _strip_title_words(strip_separators(list(line.words)))
        text = " ".join(w.text for w in kept)
        if len(normalise(text)) >= 3 and not _HAS_DIGIT.search(text) and not _looks_like_label(text):
            return text, _joint(kept), "first text line"
    return None, 0.0, ""


def _looks_like_label(text: str) -> bool:
    head = " ".join(text.split()[:2])
    labels = NUMBER_LABELS + DATE_LABELS + CUSTOMER_LABELS + CUSTOMER_EXCLUDE + TOTAL_LABELS + SUBTOTAL_LABELS
    return any(similarity(head, phrase) >= 0.85 for phrase in labels)


def _strip_title_words(words: list[OCRWord]) -> list[OCRWord]:
    kept = list(words)
    for phrase in sorted(TITLE_WORDS, key=len, reverse=True):
        length = len(phrase.split())
        index = 0
        while index + length <= len(kept):
            run = " ".join(w.text for w in kept[index : index + length])
            if similarity(run, phrase) >= 0.85:
                del kept[index : index + length]
            else:
                index += 1
    return kept


def customer_name(rows: list[Row]) -> tuple[str | None, float, str]:
    """Name next to or under a Bill To style label."""
    for match in find_label(rows, CUSTOMER_LABELS, LABEL_THRESHOLD, CUSTOMER_EXCLUDE):
        words = strip_separators(_same_block_rest(rows[match.row_index], match))
        if not words:
            words = strip_separators(line_below(rows, match.row_index, match.words[0].x0, match.words[-1].x1))
        if words and not any(similarity(w.text, "to") >= 0.9 for w in words[:1]):
            return " ".join(w.text for w in words), _joint(words), f"label '{match.words[0].text}'"
    return None, 0.0, ""


def _same_block_rest(row: Row, match: LabelMatch) -> list[OCRWord]:
    """The value right of a label: the rest of the label's own line, else the next block on the row.

    The next block is refused when it is itself a label (a "Ship To" column beside
    "Bill To"), so a neighbouring column cannot leak into the value.
    """
    last = match.words[-1]
    own = next((line for line in row.lines if any(word is last for word in line.words)), None)
    rest = [word for word in own.words if word.x0 > last.x0] if own is not None else []
    if strip_separators(rest):
        return rest
    following = sorted((line for line in row.lines if line.x0 > last.x1), key=lambda line: line.x0)
    if not following or _looks_like_label(following[0].text) or following[0].text.rstrip().endswith(":"):
        return []
    return list(following[0].words)


def _joint(words: list[OCRWord]) -> float:
    """Probability that every word of a multi-word value is right: the product of word confidences."""
    return float(np.prod([word.confidence for word in words])) if words else 0.0


def statement_holder(rows: list[Row]) -> tuple[str | None, float, str]:
    """Account holder: first row of two to four capitalised words without digits below the bank header."""
    for index, row in enumerate(rows[1:12], start=1):
        words = list(min(row.lines, key=lambda line: line.x0).words)
        tokens = [w.text for w in words]
        name_shaped = 2 <= len(tokens) <= 4 and all(_NAME_TOKEN.match(t) for t in tokens) and words[0].x0 < 400
        is_title = any(
            similarity(" ".join(tokens), phrase) >= 0.8 for phrase in ("your statement", "account statement")
        )
        if name_shaped and not is_title:
            return " ".join(tokens), _joint(words), f"name-shaped row {index}"
    return None, 0.0, ""


def period(rows: list[Row], prefer_mdy: bool) -> tuple[object, object, float, str]:
    """Two dates after a period label, split on 'to' or a dash."""
    for match in find_label(rows, PERIOD_LABELS, LABEL_THRESHOLD):
        words = value_words(rows, match)
        dates = parse_dates(" ".join(w.text for w in words), prefer_mdy)
        if len(dates) >= 2:
            digits = [w for w in words if _HAS_DIGIT.search(w.text)]
            return dates[0], dates[1], min((w.confidence for w in digits), default=0.5), "period label"
    for index, row in enumerate(rows):
        for line in row.lines:
            dates = parse_dates(line.text, prefer_mdy)
            if len(dates) == 2 and 0 <= (dates[1] - dates[0]).days <= 92:
                digits = [w for w in line.words if _HAS_DIGIT.search(w.text)]
                return (
                    dates[0],
                    dates[1],
                    0.7 * min((w.confidence for w in digits), default=0.5),
                    f"date pair row {index}",
                )
    return None, None, 0.0, ""


def currency(rows: list[Row], fields: dict[str, ExtractedField]) -> tuple[str | None, float, str]:
    """Vote over every currency cue on the page: labels and ISO codes count double, symbols once.

    Currency symbols are small and OCR confuses them with each other, so a single
    misread symbol must not outvote the codes, labels and identifiers elsewhere on the
    page. Confidence is the winner's share of the votes, capped by the cue quality.
    """
    votes: dict[str, float] = {}
    for match in find_label(rows, CURRENCY_LABELS, LABEL_THRESHOLD):
        for word in value_words(rows, match)[:2]:
            code = parse_currency(word.text.strip(".,"))
            if code and len(word.text.strip(".,")) == 3:
                votes[code] = votes.get(code, 0.0) + CODE_VOTE
    for row in rows:
        for word in row.words:
            code = parse_currency(word.text)
            if code and len(word.text) == 3:
                votes[code] = votes.get(code, 0.0) + CODE_VOTE
            elif word.text[:1] in "\u20b9\u00a3\u20ac$" and parse_money(word.text) is not None:
                symbol_code = parse_currency(word.text[:1])
                if symbol_code:
                    votes[symbol_code] = votes.get(symbol_code, 0.0) + SYMBOL_VOTE
    for code in _identifier_currency(fields):
        votes[code] = votes.get(code, 0.0) + IDENTIFIER_VOTE
    if not votes:
        return None, 0.0, ""
    winner = max(sorted(votes), key=lambda code: votes[code])
    share = votes[winner] / sum(votes.values())
    strength = min(1.0, votes[winner] / CODE_VOTE)
    confidence = share * (0.6 + 0.35 * strength)
    return winner, confidence, f"{votes[winner]:.0f} of {sum(votes.values()):.0f} currency votes"


def _identifier_currency(fields: dict[str, ExtractedField]) -> list[str]:
    """Currencies implied by checksummed identifiers: GSTIN means India, IBAN gives the country."""
    implied = []
    if "gstin" in fields:
        implied.append("INR")
    iban = fields.get("iban")
    if iban is not None and iban.value:
        country = {"GB": "GBP", "DE": "EUR", "FR": "EUR", "NL": "EUR"}.get(iban.value[:2])
        if country:
            implied.append(country)
    return implied


def _is_stop_row(row: Row) -> bool:
    text = row.text
    return any(
        similarity(" ".join(text.split()[:2]), phrase) >= 0.85
        or similarity(text.split()[0] if text else "", phrase) >= 0.9
        for phrase in STOP_LABELS
    )


def extract_line_items(rows: list[Row], page: OCRPage) -> list[LineItem]:
    """Rows under a column header, assigned to columns by x position, until a totals label.

    Without a readable header the body is still recovered from the numeric shape of
    the rows above the totals block (description, optional quantity, price, amount).
    """
    header = find_table_header(rows, ITEM_COLUMNS, page.width, minimum=2)
    if header is None:
        return _items_without_header(rows)
    _ensure_description_column(header)
    items: list[LineItem] = []
    misses = 0
    for row in rows[header.row_index + 1 :]:
        if _is_stop_row(row) or _is_separator(row) or misses >= 2:
            break
        item = _row_to_item(row, header)
        if item is None:
            misses += 1
            continue
        items.append(item)
    return items


def _ensure_description_column(header: TableHeader) -> None:
    """Descriptions are the left-most column; add it when its header word was unreadable."""
    if any(column.name == "description" for column in header.columns):
        return
    first = min(header.columns, key=lambda column: column.x0)
    first.x0 = max(first.x0, first.header.x0 - 10)
    header.columns.append(Column("description", 0.0, first.x0, first.header))


def _row_to_item(row: Row, header: TableHeader) -> LineItem | None:
    cells: dict[str, list[OCRWord]] = {column.name: [] for column in header.columns}
    for word in row.words:
        column = header.column_for(word, right_aligned=parse_money(word.text) is not None)
        if column is not None:
            cells[column.name].append(word)
    amount = _first_amount(cells.get("amount", []))
    if amount is None:
        return None
    words = cells.get("description", [])
    quantity_words = cells.get("quantity", [])
    if not quantity_words and _detached_number(words):
        quantity_words, words = [words[-1]], words[:-1]
    description = " ".join(w.text for w in words)
    if not description:
        return None
    quantity = next((q for q in (parse_quantity(w.text) for w in quantity_words) if q is not None), None)
    price = _first_amount(cells.get("unit_price", []))
    return _build_item(description, quantity, price[0] if price else None, amount[0])


def _detached_number(words: list[OCRWord]) -> bool:
    """A trailing number far right of the description text: a quantity whose column header was misread."""
    if len(words) < 2 or parse_quantity(words[-1].text) is None:
        return False
    gap = words[-1].x0 - words[-2].x1
    return gap > DETACHED_GAP * max(words[-1].y1 - words[-1].y0, 8)


def _build_item(description: str, quantity: Decimal | None, unit_price: Decimal | None, amount: Decimal) -> LineItem:
    """Fill a missing quantity or price from the other two when the arithmetic allows it."""
    if quantity is None and unit_price is not None and unit_price > 0:
        ratio = amount / unit_price
        rounded = ratio.quantize(Decimal("0.001"))
        quantity = rounded.normalize() if abs(ratio - rounded) < Decimal("0.0005") and rounded > 0 else Decimal(1)
    if quantity is None or quantity <= 0:
        quantity = Decimal(1)
    if unit_price is None:
        unit_price = (amount / quantity).quantize(Decimal("0.01"))
    return LineItem(description=description, quantity=quantity, unit_price=unit_price, amount=amount)


def _items_without_header(rows: list[Row]) -> list[LineItem]:
    """Numeric-shaped rows directly above the totals block: text left, one to three numbers right."""
    stop = next((index for index, row in enumerate(rows) if index > 3 and _is_stop_row(row)), None)
    if stop is None:
        return []
    items: list[LineItem] = []
    for row in reversed(rows[:stop]):
        words = row.words
        numbers = _trailing_numbers(words)
        if not numbers or len(numbers) == len(words):
            if items:
                break
            continue
        description = " ".join(w.text for w in words[: len(words) - len(numbers)])
        amount = parse_money(numbers[-1].text)
        if amount is None or (_HAS_DIGIT.search(description) and len(description) < 4):
            if items:
                break
            continue
        price = parse_money(numbers[-2].text) if len(numbers) >= 2 else None
        quantity = parse_quantity(numbers[0].text) if len(numbers) == 3 else None
        items.append(_build_item(description, quantity, price, amount))
    return list(reversed(items))


def _trailing_numbers(words: list[OCRWord]) -> list[OCRWord]:
    """Right-most run of tokens that parse as money or quantity."""
    run: list[OCRWord] = []
    for word in reversed(words):
        if parse_money(word.text) is not None or parse_quantity(word.text) is not None:
            run.insert(0, word)
        else:
            break
    return run


def _is_separator(row: Row) -> bool:
    """A row of dashes or dots, as till printers use between sections."""
    text = row.text.replace(" ", "")
    return len(text) >= 10 and all(ch in "-_.=~" for ch in text)


def extract_receipt_items(rows: list[Row]) -> list[LineItem]:
    """Till-style items: 'qty x price amount', 'qty x name amount', 'qty name amount', or name then amount row."""
    region = _receipt_region(rows)
    items: list[LineItem] = []
    pending = ""
    for row in region:
        words = row.words
        if _is_separator(row):
            if items:
                break
            continue
        amount = parse_money(words[-1].text) if words else None
        if amount is None:
            pending = " ".join(w.text for w in words)
            continue
        item = _receipt_row(words[:-1], amount, pending)
        pending = ""
        if item is not None:
            items.append(item)
    return items


def _receipt_region(rows: list[Row]) -> list[Row]:
    """Rows between the header block (name, address, date, number) and the first totals label."""
    start = 0
    for index, row in enumerate(rows):
        if any(
            similarity(" ".join(row.text.split()[:2]), phrase) >= 0.85
            for phrase in ("receipt no", "receipt #", "order", "date", "qty item")
        ):
            start = index + 1
    end = len(rows)
    for index in range(start, len(rows)):
        if _is_stop_row(rows[index]):
            end = index
            break
    return rows[start:end]


def _receipt_row(words: list[OCRWord], amount: Decimal, pending: str) -> LineItem | None:
    tokens = [w.text for w in words]
    quantity: Decimal | None = None
    unit_price: Decimal | None = None
    description = pending
    if len(tokens) >= 3 and normalise(tokens[1]) == "x":
        quantity = parse_quantity(tokens[0])
        unit_price = parse_money(tokens[2])
        if unit_price is None:
            description = " ".join(tokens[2:])
    elif (
        len(tokens) >= 2
        and parse_quantity(tokens[0]) is not None
        and not tokens[1][:1].isdigit()
        and parse_money(tokens[1]) is None
    ):
        quantity = parse_quantity(tokens[0])
        description = " ".join(tokens[1:])
    elif tokens:
        description = " ".join(tokens)
    if not description:
        return None
    return _build_item(description, quantity, unit_price, amount)


def extract_transactions(rows: list[Row], page: OCRPage, prefer_mdy: bool = False) -> list[Transaction]:
    """Statement rows under the column header: date, description, debit or credit, balance."""
    header = find_table_header(rows, TRANSACTION_COLUMNS, page.width, minimum=3)
    if header is None:
        return []
    transactions: list[Transaction] = []
    misses = 0
    for row in rows[header.row_index + 1 :]:
        if misses >= 2:
            break
        text = row.text
        if any(similarity(" ".join(text.split()[:3]), phrase) >= 0.85 for phrase in OPENING_LABELS + CLOSING_LABELS):
            if transactions:
                break
            continue
        transaction = _row_to_transaction(row, header, prefer_mdy)
        if transaction is None:
            misses += 1
            continue
        transactions.append(transaction)
    return transactions


def _row_to_transaction(row: Row, header: TableHeader, prefer_mdy: bool) -> Transaction | None:
    cells: dict[str, list[OCRWord]] = {column.name: [] for column in header.columns}
    for word in row.words:
        column = header.column_for(word, right_aligned=parse_money(word.text) is not None)
        if column is not None:
            cells[column.name].append(word)
    when = parse_date_text(" ".join(w.text for w in cells.get("date", [])[:3]), prefer_mdy)
    balance = _first_amount(cells.get("balance", []))
    if when is None or balance is None:
        return None
    debit = _first_amount(cells.get("debit", []))
    credit = _first_amount(cells.get("credit", []))
    return Transaction(
        date=when,
        description=" ".join(w.text for w in cells.get("description", [])),
        debit=debit[0] if debit else None,
        credit=credit[0] if credit else None,
        balance=balance[0],
    )
