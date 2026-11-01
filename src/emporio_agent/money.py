"""Exact monetary parsing, rounding, discounts, and installment allocation."""

import re
from decimal import ROUND_HALF_UP, Decimal, InvalidOperation, localcontext

MAX_CENTS = 2**63 - 1
_DECIMAL = re.compile(r"\d+(?:\.\d+)?\Z")


def _decimal(value: str) -> Decimal:
    if len(value) > 80 or not _DECIMAL.fullmatch(value.strip()):
        raise ValueError("A nonnegative decimal string is required.")
    try:
        result = Decimal(value.strip())
    except InvalidOperation as exc:
        raise ValueError("Invalid decimal value.") from exc
    if not result.is_finite():
        raise ValueError("A finite decimal value is required.")
    return result


def check_cents(value: object) -> int:
    """Validate a nonnegative integer amount within SQLite's exact integer range."""
    if not isinstance(value, int) or isinstance(value, bool) or not 0 <= value <= MAX_CENTS:
        raise ValueError("The monetary amount is outside the supported range.")
    return value


def parse_money_cents(value: str) -> int:
    """Parse decimal currency without accepting or constructing binary floats."""
    amount = _decimal(value)
    with localcontext() as context:
        context.prec = 100
        cents = amount * 100
        if cents != cents.to_integral_value():
            raise ValueError("Currency amounts must have at most two decimal places.")
        if cents > MAX_CENTS:
            raise ValueError("The monetary amount is outside the supported range.")
        return check_cents(int(cents))


def canonical_rate(value: str) -> str:
    """Validate a percentage and retain its exact decimal representation."""
    rate = _decimal(value)
    if rate > 100:
        raise ValueError("A percentage must be between zero and one hundred.")
    rendered = format(rate, "f")
    return rendered.rstrip("0").rstrip(".") if "." in rendered else rendered


def round_cents(value: Decimal) -> int:
    """Round a decimal amount expressed in cents with explicit HALF_UP rounding."""
    if not value.is_finite() or value < 0:
        raise ValueError("A finite nonnegative amount is required.")
    with localcontext() as context:
        context.prec = max(100, len(value.as_tuple().digits) + 5)
        if value > MAX_CENTS:
            raise ValueError("The monetary amount is outside the supported range.")
        return check_cents(int(value.quantize(Decimal(1), rounding=ROUND_HALF_UP)))


def discounted_cents(amount_cents: int, rate: str) -> int:
    """Apply an exact decimal percentage and round the resulting amount to cents."""
    check_cents(amount_cents)
    percentage = Decimal(canonical_rate(rate))
    with localcontext() as context:
        context.prec = 120
        return round_cents(Decimal(amount_cents) * (100 - percentage) / 100)


def split_installments(amount_cents: int, count: int) -> tuple[int, ...]:
    """Allocate cents across installments, preserving the exact total and nonnegative parts."""
    check_cents(amount_cents)
    if type(count) is not int or not 1 <= count <= 120:
        raise ValueError("The installment count must be between one and 120.")
    base, remainder = divmod(amount_cents, count)
    return tuple(base + (index < remainder) for index in range(count))


def format_brl(amount_cents: int) -> str:
    """Format exact integer cents as an English-language BRL amount."""
    check_cents(amount_cents)
    whole, fraction = divmod(amount_cents, 100)
    return f"BRL {whole:,}.{fraction:02d}"
