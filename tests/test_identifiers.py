import random

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from ledgerlens.generator.identifiers import (
    format_iban,
    gstin_check_char,
    gstin_is_valid,
    iban_is_valid,
    make_iban,
    random_document_number,
    random_gstin,
    random_iban,
)


def test_known_gstin_is_valid() -> None:
    assert gstin_is_valid("27AAPFU0939F1ZV")
    assert gstin_check_char("27AAPFU0939F1ZV"[:14]) == "V"


def test_gstin_rejects_bad_shape_and_bad_checksum() -> None:
    assert not gstin_is_valid("27AAPFU0939F1ZA")
    assert not gstin_is_valid("27AAPFU0939F1Z")
    assert not gstin_is_valid("27aapfu0939f1zv".upper()[:-1] + "0")


@given(st.integers(min_value=0, max_value=10**9))
@settings(max_examples=60, deadline=None)
def test_generated_gstin_always_valid(seed: int) -> None:
    assert gstin_is_valid(random_gstin(random.Random(seed)))


@given(st.integers(min_value=0, max_value=10**9), st.sampled_from(["GB", "DE"]))
@settings(max_examples=60, deadline=None)
def test_generated_iban_always_valid(seed: int, country: str) -> None:
    iban = random_iban(random.Random(seed), country)
    assert iban_is_valid(iban)
    assert iban_is_valid(format_iban(iban))


def test_single_digit_substitution_breaks_iban_checksum() -> None:
    iban = random_iban(random.Random(5), "DE")
    broken = iban[:10] + str((int(iban[10]) + 1) % 10) + iban[11:]
    assert not iban_is_valid(broken)


def test_make_iban_matches_reference_example() -> None:
    assert make_iban("GB", "WEST12345698765432") == "GB82WEST12345698765432"


def test_iban_country_length_enforced() -> None:
    assert not iban_is_valid("GB82WEST1234569876543")
    assert not iban_is_valid("XX00123456789012")


def test_random_iban_unknown_country_raises() -> None:
    with pytest.raises(ValueError):
        random_iban(random.Random(1), "IN")


def test_document_numbers_have_expected_shapes() -> None:
    rng = random.Random(3)
    for kind in ("invoice", "receipt", "statement"):
        number = random_document_number(rng, kind, 2026)
        assert any(ch.isdigit() for ch in number)
        assert " " not in number
