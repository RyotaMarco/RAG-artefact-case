"""Deterministic store rules, applied by tools only while policy_evidence confirms them."""

from datetime import date, datetime, time, timedelta

# Owner rulings for combined payments: two different methods, catalog prices without the PIX
# discount, and no boleto because it takes up to three business days to clear.
SPLIT_METHODS = ("pix", "credit", "debit")


# Regular hours from manual section 2, applied only while the "hours" evidence is found.
WEEKLY_HOURS = {day: (time(9), time(18)) for day in range(5)} | {5: (time(9), time(13))}
_WEEKDAYS = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")


def opening_status(now: datetime) -> dict[str, object]:
    """Apply the regular weekly schedule to a store-local time, without holiday knowledge."""
    hours = WEEKLY_HOURS.get(now.weekday())
    is_open = hours is not None and hours[0] <= now.time() < hours[1]
    following = now
    while True:
        slot = WEEKLY_HOURS.get(following.weekday())
        start = (
            following.replace(hour=slot[0].hour, minute=0, second=0, microsecond=0)
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


# Manual section 5.1; the limit applies to the final amount paid, per the owner's ruling.
METRO_FREE_ABOVE_CENTS = 50000
METRO_FEE_CENTS = 3500


def metro_shipping_cents(final_total_cents: int) -> int:
    """Return the metropolitan fee: free strictly above R$ 500,00, R$ 35,00 otherwise."""
    return 0 if final_total_cents > METRO_FREE_ABOVE_CENTS else METRO_FEE_CENTS


# Calendar-day windows from manual sections 4.1-4.3 and 8.1 and the owner's rulings.
RETURN_REASONS = ("regret", "preference", "defect")
_CHANNELS = ("online", "store")


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


def _missing_date(channel: str, *, defect: bool = False) -> str:
    if channel == "store":
        return (
            "Ask the customer for the purchase date; in-store purchases have no separate "
            "receipt date."
        )
    if defect:
        return "Ask the customer for the purchase date."
    return "Ask the customer for the date the product was received."


def _window(start: date, days: int, today: date) -> dict[str, object]:
    deadline = start + timedelta(days=days)
    return {
        "deadline": deadline.isoformat(),
        "within_window": today <= deadline,
        "days_left": max((deadline - today).days, 0),
    }


def return_window(
    reason: str, channel: str, purchased_on: str | None, received_on: str | None, today: date
) -> dict[str, object]:
    """Evaluate customer-supplied dates; never infer receipt from order or delivery estimates.

    Raises:
        ValueError: The reason, channel or a required date is missing or invalid.
    """
    if reason not in RETURN_REASONS:
        raise ValueError("Reason must be regret, preference or defect.")
    if channel not in _CHANNELS:
        raise ValueError("Ask whether the purchase was online or in the physical store.")
    purchased = _day(purchased_on, "purchase date", today)
    received = _day(received_on, "receipt date", today)
    if channel == "store":
        received = received or purchased
    if purchased is not None and received is not None and received < purchased:
        raise ValueError("The receipt date cannot be before the purchase date.")
    result: dict[str, object] = {"reason": reason, "channel": channel}
    if reason == "regret" and channel == "store":
        result["applicable"] = False
        result["explanation"] = (
            "The right of withdrawal covers online purchases only. Refer the customer to "
            "WhatsApp for clarification; do not offer an exchange or alternative yourself."
        )
        return result
    if reason in {"regret", "preference"}:
        if received is None:
            raise ValueError(_missing_date(channel))
        result.update(_window(received, 7, today))
        result["conditions"] = (
            "Original packaging, unused, with all accessories and manuals. Refund with the "
            "original payment method within 10 business days; the store pays the return "
            "shipping. The customer starts the return on WhatsApp."
            if reason == "regret"
            else "Perfect condition, original packaging, and the desired product available."
        )
        return result
    if purchased is None:
        raise ValueError(_missing_date(channel, defect=True))
    # Owner ruling: after the store's 30 days there is no store deadline; the store can
    # intermediate the manufacturer's warranty, whose term is in the product certificate.
    result["store_exchange"] = _window(purchased, 30, today)
    result["after_store_exchange"] = (
        "The warranty is the manufacturer's; its term is in the product's warranty "
        "certificate (6 months to 2 years). The store can intermediate the request on "
        "WhatsApp. Never say the warranty has expired."
    )
    result["conditions"] = (
        "Manufacturing defects only; misuse, falls, humidity and unauthorized changes are "
        "not covered."
    )
    return result
