"""Exercise exact storage, rounding, overflow rejection, and reconciled installments."""

from decimal import Decimal

import pytest

from emporio_agent.money import (
    MAX_CENTS,
    canonical_rate,
    check_cents,
    discounted_cents,
    format_brl,
    parse_money_cents,
    round_cents,
    split_installments,
)


def test_money_never_round_trips_through_float() -> None:
    """Preserve cents beyond binary floating-point integer precision."""
    assert parse_money_cents("90071992547409.93") == 9007199254740993
    assert parse_money_cents("92233720368547758.07") == MAX_CENTS
    assert format_brl(219900) == "BRL 2,199.00"


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
