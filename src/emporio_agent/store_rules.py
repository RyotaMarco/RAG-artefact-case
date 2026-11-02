"""Deterministic store rules: numbers come from the manual, choices from the store rulings."""

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta

_WEEKDAYS = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")


def opening_status(
    now: datetime, weekly_hours: Mapping[int, tuple[time, time]]
) -> dict[str, object]:
    """Apply the manual's weekly schedule to a store-local time, without holiday knowledge."""
    hours = weekly_hours.get(now.weekday())
    is_open = hours is not None and hours[0] <= now.time() < hours[1]
    following = now
    while True:
        slot = weekly_hours.get(following.weekday())
        start = (
            following.replace(hour=slot[0].hour, minute=slot[0].minute, second=0, microsecond=0)
            if slot is not None
            else None
        )
        if start is not None and start > now:
            break
        following = (following + timedelta(days=1)).replace(hour=0, minute=0)
    status: dict[str, object] = {
        "store_local_time": now.isoformat(timespec="minutes"),
        "weekday": _WEEKDAYS[now.weekday()],
        "open_by_regular_hours": is_open,
        "today": f"{hours[0]:%H:%M}-{hours[1]:%H:%M}" if hours is not None else "closed",
        "next_opening": f"{_WEEKDAYS[following.weekday()]} {start:%Y-%m-%d %H:%M}",
        "warning": "Holidays and special dates are not known here: the store closes on "
        "holidays and may extend hours on special dates. Ask the customer to confirm on "
        "WhatsApp.",
    }
    if is_open and hours is not None:
        status["closes_at"] = f"{hours[1]:%H:%M}"
    return status


def metro_shipping_cents(
    amount_cents: int, *, threshold_cents: int, fee_cents: int, free_at_threshold: bool
) -> int:
    """Return the metropolitan fee for an amount, using the manual's threshold and fee."""
    free = amount_cents >= threshold_cents if free_at_threshold else amount_cents > threshold_cents
    return 0 if free else fee_cents


RETURN_REASONS = ("regret", "preference", "defect")
_CHANNELS = ("online", "store")


@dataclass(frozen=True)
class ReturnTerms:
    """Deadlines read from the manual plus the store's rulings on exchanges."""

    regret_days: int
    refund_business_days: int
    preference_days: int
    defect_days: int
    manufacturer_warranty: str
    preference_counts_from: str
    in_store_preference: str


def _day(value: str | None, label: str, today: date) -> date | None:
    if value is None or not value.strip():
        return None
    try:
        parsed = date.fromisoformat(value.strip())
    except ValueError as error:
        raise ValueError(f"The {label} must be a YYYY-MM-DD date.") from error
    if parsed > today:
        raise ValueError(f"The {label} cannot be in the future.")
    return parsed


def _missing_date(channel: str, *, from_purchase: bool = False) -> str:
    if channel == "store":
        return (
            "Ask the customer for the purchase date; in-store purchases have no separate "
            "receipt date."
        )
    if from_purchase:
        return "Ask the customer for the purchase date."
    return "Ask the customer for the date the product was received."


def _dates(
    channel: str, purchased_on: str | None, received_on: str | None, today: date
) -> tuple[date | None, date | None]:
    purchased = _day(purchased_on, "purchase date", today)
    received = _day(received_on, "receipt date", today)
    if channel == "store":
        received = received or purchased
    if purchased is not None and received is not None and received < purchased:
        raise ValueError("The receipt date cannot be before the purchase date.")
    return purchased, received


def _in_store_referral(reason: str, channel: str, terms: ReturnTerms) -> str | None:
    if channel != "store":
        return None
    if reason == "regret":
        return (
            "The right of withdrawal covers online purchases only. Refer the customer to "
            "WhatsApp for clarification; do not offer an exchange or alternative yourself."
        )
    if reason == "preference" and terms.in_store_preference == "whatsapp":
        return (
            "Preference exchanges for in-store purchases are handled by staff. Refer the "
            "customer to WhatsApp; do not calculate a deadline or promise an exchange."
        )
    return None


def _window(start: date, days: int, today: date) -> dict[str, object]:
    deadline = start + timedelta(days=days)
    return {
        "deadline": deadline.isoformat(),
        "within_window": today <= deadline,
        "days_left": max((deadline - today).days, 0),
    }


def return_window(
    reason: str,
    channel: str,
    purchased_on: str | None,
    received_on: str | None,
    today: date,
    terms: ReturnTerms,
) -> dict[str, object]:
    """Evaluate customer-supplied dates; never infer receipt from order or delivery estimates.

    Raises:
        ValueError: The reason, channel or a required date is missing or invalid.
    """
    if reason not in RETURN_REASONS:
        raise ValueError("Reason must be regret, preference or defect.")
    if channel not in _CHANNELS:
        raise ValueError("Ask whether the purchase was online or in the physical store.")
    purchased, received = _dates(channel, purchased_on, received_on, today)
    result: dict[str, object] = {"reason": reason, "channel": channel}
    referral = _in_store_referral(reason, channel, terms)
    if referral is not None:
        result["applicable"] = False
        result["explanation"] = referral
        return result
    if reason == "regret":
        if received is None:
            raise ValueError(_missing_date(channel))
        result.update(_window(received, terms.regret_days, today))
        result["conditions"] = (
            "Original packaging, unused, with all accessories and manuals. Refund with the "
            f"original payment method within {terms.refund_business_days} business days; the "
            "store pays the return shipping. The customer starts the return on WhatsApp."
        )
        return result
    if reason == "preference":
        from_purchase = terms.preference_counts_from == "purchase"
        start = purchased if from_purchase else received
        if start is None:
            raise ValueError(_missing_date(channel, from_purchase=from_purchase))
        result.update(_window(start, terms.preference_days, today))
        result["conditions"] = (
            "Perfect condition, original packaging, and the desired product available."
        )
        return result
    if purchased is None:
        raise ValueError(_missing_date(channel, from_purchase=True))
    result["store_exchange"] = _window(purchased, terms.defect_days, today)
    result["after_store_exchange"] = (
        "The warranty is the manufacturer's; its term is in the product's warranty "
        f"certificate (manual: {terms.manufacturer_warranty}). The store can intermediate the "
        "request on WhatsApp. Never say the warranty has expired."
    )
    result["conditions"] = (
        "Manufacturing defects only; misuse, falls, humidity and unauthorized changes are "
        "not covered."
    )
    return result
