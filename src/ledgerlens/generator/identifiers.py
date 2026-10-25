"""Identifier generation and checksum validation (IBAN mod-97, GSTIN base-36).

Checksums matter twice: the generator uses them to emit valid identifiers, and the
pipeline uses the same functions as a validation rule. Sharing one implementation
guarantees the rule accepts exactly what the generator produces, and rejects a
single OCR substitution with probability close to 1 - 1/97 (IBAN) or 1 - 1/36 (GSTIN).
"""

from __future__ import annotations

import random
import re
import string

ALNUM36 = string.digits + string.ascii_uppercase
IBAN_LENGTHS: dict[str, int] = {"GB": 22, "DE": 22, "FR": 27, "NL": 18}
GSTIN_PATTERN = re.compile(r"^[0-9]{2}[A-Z]{5}[0-9]{4}[A-Z][1-9A-Z]Z[0-9A-Z]$")
IBAN_PATTERN = re.compile(r"^[A-Z]{2}[0-9]{2}[A-Z0-9]{11,30}$")

# GST state codes used by the generator (a realistic subset of the official list).
GST_STATE_CODES: tuple[str, ...] = ("07", "27", "29", "33", "36", "19", "24", "08", "09", "06")


def _iban_numeric(value: str) -> int:
    """Convert an IBAN (check digits in place) into the integer used by the mod-97 test."""
    rearranged = value[4:] + value[:4]
    digits = "".join(str(ALNUM36.index(ch)) if ch.isalpha() else ch for ch in rearranged)
    return int(digits)


def iban_is_valid(value: str) -> bool:
    """True when the IBAN has a known country length and passes the ISO 13616 mod-97 test."""
    cleaned = value.replace(" ", "").upper()
    if not IBAN_PATTERN.match(cleaned):
        return False
    expected = IBAN_LENGTHS.get(cleaned[:2])
    if expected is not None and len(cleaned) != expected:
        return False
    return _iban_numeric(cleaned) % 97 == 1


def make_iban(country: str, bban: str) -> str:
    """Attach the two ISO 7064 check digits to a country code and BBAN."""
    provisional = f"{country}00{bban}"
    check = 98 - _iban_numeric(provisional) % 97
    return f"{country}{check:02d}{bban}"


def random_iban(rng: random.Random, country: str) -> str:
    """Generate a structurally valid IBAN for GB (bank letters, sort code, account) or DE."""
    if country == "GB":
        bank = "".join(rng.choice(string.ascii_uppercase) for _ in range(4))
        bban = bank + "".join(rng.choice(string.digits) for _ in range(14))
    elif country == "DE":
        bban = "".join(rng.choice(string.digits) for _ in range(18))
    else:
        raise ValueError(f"no IBAN scheme for country {country!r}")
    return make_iban(country, bban)


def format_iban(value: str) -> str:
    """Group an IBAN in blocks of four, the way it is printed on documents."""
    return " ".join(value[i : i + 4] for i in range(0, len(value), 4))


def gstin_check_char(body: str) -> str:
    """Compute the 15th GSTIN character from the first 14 (alternating 1, 2 factors, base 36)."""
    total = 0
    for index, ch in enumerate(body):
        factor = 1 if index % 2 == 0 else 2
        product = factor * ALNUM36.index(ch)
        total += product // 36 + product % 36
    return ALNUM36[(36 - total % 36) % 36]


def gstin_is_valid(value: str) -> bool:
    """True when the GSTIN matches the official pattern and its check character."""
    cleaned = value.replace(" ", "").upper()
    if not GSTIN_PATTERN.match(cleaned):
        return False
    return gstin_check_char(cleaned[:14]) == cleaned[14]


def random_gstin(rng: random.Random) -> str:
    """Generate a valid GSTIN: state code, PAN-shaped block, entity digit, Z, check char."""
    state = rng.choice(GST_STATE_CODES)
    pan = (
        "".join(rng.choice(string.ascii_uppercase) for _ in range(3))
        + rng.choice("PCHFAT")
        + rng.choice(string.ascii_uppercase)
        + "".join(rng.choice(string.digits) for _ in range(4))
        + rng.choice(string.ascii_uppercase)
    )
    body = state + pan + rng.choice("123456789") + "Z"
    return body + gstin_check_char(body)


def random_document_number(rng: random.Random, kind: str, year: int) -> str:
    """Produce a document number in one of several house styles seen on real paperwork."""
    serial = rng.randint(1, 99999)
    styles = {
        "invoice": [
            f"INV-{year}-{serial:05d}",
            f"IN/{year % 100:02d}-{(year + 1) % 100:02d}/{serial:04d}",
            f"{rng.randint(100000, 999999)}",
            f"INV{serial:06d}",
        ],
        "receipt": [f"R{serial:06d}", f"{rng.randint(1000, 9999)}-{serial:05d}", f"RCPT-{serial:05d}"],
        "statement": [f"ST-{year}-{serial:05d}", f"{rng.randint(10000000, 99999999)}"],
    }
    return rng.choice(styles[kind])
