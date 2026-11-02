"""Exercise exact storage, rounding, overflow rejection, and reconciled installments."""

from decimal import Decimal

import pytest

from emporio_agent.money import (
    MAX_CENTS,
    canonical_rate,
    check_cents,
    discounted_cents,
    format_brl,
    parse_customer_amount,
    parse_money_cents,
    round_cents,
    split_installments,
)


def test_money_never_round_trips_through_float() -> None:
    """Preserve cents beyond binary floating-point integer precision."""
    assert parse_money_cents("90071992547409.93") == 9007199254740993
    assert parse_money_cents("92233720368547758.07") == MAX_CENTS
    assert format_brl(219900) == "R$ 2.199,00"
    assert format_brl(5) == "R$ 0,05"
    assert format_brl(129990) == "R$ 1.299,90"
    assert format_brl(100000000) == "R$ 1.000.000,00"


@pytest.mark.parametrize(
    "value", ["NaN", "Infinity", "-1", "1e2", "1,20", "1.001", "92233720368547758.08", ""]
)
def test_invalid_money_is_rejected(value: str) -> None:
    """Reject ambiguous, non-finite, negative, overprecise, and overflowing amounts."""
    with pytest.raises(ValueError):
        parse_money_cents(value)


@pytest.mark.parametrize("value", [True, 1.1, -1, MAX_CENTS + 1, "100"])
def test_integer_boundary_rejects_coercions(value: object) -> None:
    """Do not accept floats, booleans, or strings as already-parsed cents."""
    with pytest.raises(ValueError):
        check_cents(value)


def test_exact_discounts_and_half_up_rounding() -> None:
    """Apply fractional percentages and round only the final amount."""
    assert canonical_rate("2.500") == "2.5"
    assert discounted_cents(54900, "20") == 43920
    assert discounted_cents(101, "50") == 51
    assert discounted_cents(MAX_CENTS, "0") == MAX_CENTS
    assert round_cents(Decimal("100.5")) == 101


def test_installments_reconcile_including_tiny_amounts() -> None:
    """Distribute remainder cents without negative final installments or lost money."""
    assert split_installments(100, 3) == (34, 33, 33)
    assert split_installments(6, 12) == (1,) * 6 + (0,) * 6
    assert sum(split_installments(239918, 12)) == 239918
    with pytest.raises(ValueError):
        split_installments(100, 0)


@pytest.mark.parametrize("rate", ["-1", "100.01", "NaN", "5e1"])
def test_invalid_percentages_are_rejected(rate: str) -> None:
    """Keep percentage inputs finite and explicitly decimal."""
    with pytest.raises(ValueError):
        discounted_cents(100, rate)


@pytest.mark.parametrize(
    ("typed", "cents"),
    [
        ("1000", 100000),
        ("1.000", 100000),
        ("1.000,50", 100050),
        ("1000,5", 100050),
        ("R$ 1.000,00", 100000),
        ("  r$1.299,90 ", 129990),
        ("12.345.678", 1234567800),
        ("1000.50", 100050),
        ("1,000.50", 100050),
        ("0,05", 5),
        ("150", 15000),
    ],
)
def test_customer_amounts_read_brazilian_and_english_forms(typed: str, cents: int) -> None:
    """A dot followed by three digits is a thousands separator, never R$ 1,00."""
    assert parse_customer_amount(typed) == cents


@pytest.mark.parametrize(
    "typed", ["1,000", "1.00.0", "1.0000", "1000,505", "-100", "mil reais", "", "1.000.50", "1e3"]
)
def test_ambiguous_customer_amounts_ask_to_write_again(typed: str) -> None:
    """Ambiguous or invalid amounts are refused with a kind request for a valid value."""
    with pytest.raises(ValueError, match="Kindly ask the customer to write it again"):
        parse_customer_amount(typed)
