"""Turn OCR tokens into typed values: money, dates, quantities, identifiers.

OCR confuses characters within a class (0/O, 1/l/I, 5/S, 8/B, 2/Z, 6/G), so every
parser first maps letters to digits where a digit is required by the pattern, then
validates the result. Money parsing decides the decimal separator from the trailing
two-digit group, which handles 1,234.56 and 1.234,56 and 1,23,456.00 alike.
"""

from __future__ import annotations

import re
from datetime import date
from decimal import Decimal, InvalidOperation

from ledgerlens.generator.identifiers import GSTIN_PATTERN, IBAN_LENGTHS, gstin_is_valid, iban_is_valid

LETTER_TO_DIGIT: dict[str, str] = {
    "O": "0",
    "o": "0",
    "D": "0",
    "Q": "0",
    "I": "1",
    "l": "1",
    "L": "1",
    "i": "1",
    "|": "1",
    "!": "1",
    "Z": "2",
    "z": "2",
    "S": "5",
    "s": "5",
    "B": "8",
    "G": "6",
    "g": "9",
    "q": "9",
    "T": "7",
    "A": "4",
}
DIGIT_TO_LETTER: dict[str, str] = {"0": "O", "1": "I", "2": "Z", "5": "S", "8": "B", "6": "G", "4": "A", "7": "T"}
CURRENCY_TOKENS: dict[str, str] = {
    "INR": "INR",
    "RS": "INR",
    "RS.": "INR",
    "₹": "INR",
    "GBP": "GBP",
    "£": "GBP",
    "EUR": "EUR",
    "€": "EUR",
    "USD": "USD",
    "$": "USD",
    "US$": "USD",
}
MONTHS: dict[str, int] = {
    name: index
    for index, name in enumerate(
        (
            "january",
            "february",
            "march",
            "april",
            "may",
            "june",
            "july",
            "august",
            "september",
            "october",
            "november",
            "december",
        ),
        start=1,
    )
}
MONTHS.update({name[:3]: index for name, index in list(MONTHS.items())})
_MONEY_SHAPE = re.compile(r"^-?[0-9]{1,3}(?:[.,][0-9]{2,3})*[.,][0-9]{2}$|^-?[0-9]+[.,][0-9]{2}$")
_NUMERIC = re.compile(r"^-?[0-9]+(?:[.,][0-9]+)?$")
_DATE_NUMERIC = re.compile(r"^(\d{1,4})[./-](\d{1,2})[./-](\d{1,4})$")
_YEAR_RANGE = (2000, 2035)


def digitise(token: str) -> str:
    """Map OCR letter confusions to digits; keep separators and signs."""
    return "".join(LETTER_TO_DIGIT.get(ch, ch) for ch in token)


def strip_currency(token: str) -> tuple[str, str | None]:
    """Remove leading or trailing currency symbols or codes; return (rest, first currency seen).

    Two symbols in a row happen when OCR reads a digit as a currency glyph, so the
    strip repeats; the digits that remain still have to pass the money shape test.
    """
    text = token.strip()
    currency: str | None = None
    for _ in range(2):
        for symbol, code in CURRENCY_TOKENS.items():
            if text.upper().startswith(symbol) and len(text) > len(symbol):
                text, currency = text[len(symbol) :].strip(), currency or code
                break
            if text.upper().endswith(symbol) and len(text) > len(symbol):
                text, currency = text[: -len(symbol)].strip(), currency or code
                break
    return text, currency


def parse_money(token: str) -> Decimal | None:
    """Cent-precision Decimal from a printed amount in any supported separator style."""
    text, _ = strip_currency(token)
    text = digitise(text.replace(" ", ""))
    if not _MONEY_SHAPE.match(text):
        return None
    sign = -1 if text.startswith("-") else 1
    body = text.lstrip("-")
    integer, cents = body[:-3], body[-2:]
    digits = re.sub(r"[.,]", "", integer)
    if not digits:
        return None
    try:
        return sign * Decimal(f"{digits}.{cents}")
    except InvalidOperation:
        return None


def parse_quantity(token: str) -> Decimal | None:
    """Integer or simple decimal quantity; rejects amounts with two trailing decimals."""
    text = digitise(token.strip().rstrip("xX"))
    if not _NUMERIC.match(text):
        return None
    try:
        value = Decimal(text.replace(",", "."))
    except InvalidOperation:
        return None
    return value if value >= 0 else None


def parse_currency(token: str) -> str | None:
    """ISO code when the token is a known currency symbol or code."""
    return CURRENCY_TOKENS.get(token.strip().upper())


def _valid_date(year: int, month: int, day: int) -> date | None:
    if not (_YEAR_RANGE[0] <= year <= _YEAR_RANGE[1]):
        return None
    try:
        return date(year, month, day)
    except ValueError:
        return None


def parse_date_tokens(tokens: list[str], prefer_mdy: bool = False) -> date | None:
    """Try numeric and month-name date shapes over a run of one to three tokens."""
    joined = "".join(tokens).strip(",.")
    numeric = _DATE_NUMERIC.match(digitise(joined))
    if numeric:
        return _numeric_date(*(int(part) for part in numeric.groups()), prefer_mdy=prefer_mdy)
    if len(tokens) == 3:
        return _named_date(tokens)
    if len(tokens) == 1 and "-" in tokens[0]:
        return _named_date(tokens[0].split("-"))
    return None


def _numeric_date(a: int, b: int, c: int, prefer_mdy: bool) -> date | None:
    """Resolve the day/month order: ISO when the first group is a year, else locale with sanity swap."""
    if a > 31:
        return _valid_date(a, b, c)
    year = c if c > 99 else 2000 + c
    first, second = (b, a) if prefer_mdy else (a, b)
    day, month = first, second
    if month > 12 and day <= 12:
        day, month = month, day
    return _valid_date(year, month, day)


def _named_date(parts: list[str]) -> date | None:
    """'14 March 2026', '14-Mar-2026' or 'March 14, 2026'."""
    cleaned = [part.strip(",.") for part in parts]
    if len(cleaned) != 3:
        return None
    month_first = cleaned[0].lower() in MONTHS or cleaned[0].lower()[:3] in MONTHS
    month_token = cleaned[0] if month_first else cleaned[1]
    day_token = cleaned[1] if month_first else cleaned[0]
    month = MONTHS.get(month_token.lower()) or MONTHS.get(month_token.lower()[:3])
    if month is None:
        return None
    try:
        return _valid_date(int(digitise(cleaned[2])), month, int(digitise(day_token)))
    except ValueError:
        return None


def parse_dates(text: str, prefer_mdy: bool = False) -> list[date]:
    """Every date in a short text, left to right, without overlapping token runs."""
    tokens = text.replace(",", " ").split()
    found: list[date] = []
    start = 0
    while start < len(tokens):
        for size in (3, 1, 2):
            run = tokens[start : start + size]
            parsed = parse_date_tokens(run, prefer_mdy) if len(run) == size else None
            if parsed is not None:
                found.append(parsed)
                start += size
                break
        else:
            start += 1
    return found


def parse_date_text(text: str, prefer_mdy: bool = False) -> date | None:
    """The first date in a short text, or None."""
    dates = parse_dates(text, prefer_mdy)
    return dates[0] if dates else None


def _repair_by_pattern(value: str, shape: str) -> str:
    """Force each position to the class the pattern expects: d = digit, a = letter, x = either, else a literal."""
    out = []
    for ch, want in zip(value, shape, strict=False):
        if want == "d" and not ch.isdigit():
            out.append(LETTER_TO_DIGIT.get(ch, ch))
        elif want == "a" and not ch.isalpha():
            out.append(DIGIT_TO_LETTER.get(ch, ch))
        elif want.isupper() and ch in (want, DIGIT_TO_LETTER.get(ch, ""), "2" if want == "Z" else ""):
            out.append(want)
        else:
            out.append(ch)
    return "".join(out) + value[len(shape) :]


def _twin(ch: str) -> str | None:
    """The look-alike character of the other class, if any (0/O, 1/I, 5/S, 8/B, 2/Z, 6/G)."""
    return DIGIT_TO_LETTER.get(ch) or LETTER_TO_DIGIT.get(ch)


def _alternatives(value: str, free_positions: tuple[int, ...]) -> list[str]:
    """All variants where each free position may take its look-alike twin (at most 2^n strings)."""
    variants = [value]
    for position in free_positions:
        twin = _twin(value[position]) if position < len(value) else None
        if twin is None:
            continue
        variants += [v[:position] + twin + v[position + 1 :] for v in variants]
    return variants


def parse_gstin(token: str) -> str | None:
    """Fifteen-character GSTIN, repaired position by position and verified by checksum.

    The entity position and the check character admit both letters and digits, so their
    look-alike variants are tried until one satisfies the checksum.
    """
    cleaned = re.sub(r"[^0-9A-Za-z]", "", token).upper()
    if len(cleaned) != 15:
        return None
    repaired = _repair_by_pattern(cleaned, "ddaaaaaddddaxZx")
    for candidate in _alternatives(repaired, (12, 14)):
        if GSTIN_PATTERN.match(candidate) and gstin_is_valid(candidate):
            return candidate
    return None


def parse_iban(token: str) -> str | None:
    """IBAN with country-specific repair (GB: 4 letters then digits; DE: digits) and mod-97 check."""
    cleaned = re.sub(r"[^0-9A-Za-z]", "", token).upper()
    if len(cleaned) < 15:
        return None
    country = _repair_by_pattern(cleaned[:2], "aa")
    expected = IBAN_LENGTHS.get(country)
    if expected is None or len(cleaned) != expected:
        return None
    shape = "aadd" + ("aaaa" + "d" * 14 if country == "GB" else "d" * (expected - 4))
    repaired = _repair_by_pattern(country + cleaned[2:], shape)
    return repaired if iban_is_valid(repaired) else None


def looks_like_amount(token: str) -> bool:
    """Cheap shape test used by the layout code before committing to a parse."""
    return parse_money(token) is not None
