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


_BRAZILIAN_AMOUNT = re.compile(r"(\d{1,3}(?:\.\d{3})+|\d+)(?:,(\d{1,2}))?")
_ENGLISH_AMOUNT = re.compile(r"(\d{1,3}(?:,\d{3})+|\d+)\.(\d{1,2})")
AMOUNT_HINT = (
    "Could not read that amount. Kindly ask the customer to write it again, for example "
    "1.000 or 1.000,50."
)


def parse_customer_amount(value: str) -> int:
    """Read an amount typed by a customer, Brazilian style first, as exact integer cents.

    Accepts 1000, 1.000, 1.000,50, 1000,50, R$ 1.000,00 and the English 1,000.50 or 1000.50.
    A dot followed by exactly three digits is a thousands separator; ambiguous forms such as
    1,000 are refused instead of guessed.

    Raises:
        ValueError: The amount cannot be read unambiguously.
    """
    text = value.strip().removeprefix("R$").removeprefix("r$").strip()
    if len(text) > 40:
        raise ValueError(AMOUNT_HINT)
    if match := _BRAZILIAN_AMOUNT.fullmatch(text):
        whole, fraction = match.group(1).replace(".", ""), match.group(2) or "0"
    elif match := _ENGLISH_AMOUNT.fullmatch(text):
        whole, fraction = match.group(1).replace(",", ""), match.group(2)
    else:
        raise ValueError(AMOUNT_HINT)
    return parse_money_cents(f"{whole}.{fraction}")


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
    """Format exact integer cents as Brazilian reais, for example R$ 1.299,90."""
    check_cents(amount_cents)
    whole, fraction = divmod(amount_cents, 100)
    return f"R$ {whole:,}".replace(",", ".") + f",{fraction:02d}"
