"""Manual quotes each deterministic store rule depends on, and contact masking."""

import re
from collections.abc import Sequence
from datetime import date, datetime, time, timedelta

from emporio_agent.documents import Citation, PolicySection

MASK = "[contato interno omitido]"
_PHONE = re.compile(r"(?:\(\d{2}\)|\b\d{2})\s*\d{4,5}-\d{4}\b")
_EMAIL = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
_WHATSAPP = re.compile(r"canal de whatsapp\s+(\(\d{2}\)\s*\d{4,5}-\d{4})", re.IGNORECASE)

# A tool applies each rule only while the active manual still contains these quotes in the
# named section, so a changed manual disables the rule instead of keeping a stale value.
RULE_EVIDENCE: dict[str, tuple[tuple[str, str], ...]] = {
    "hours": (
        ("2.", "Segunda a Sexta-feira 09:00 às 18:00"),
        ("2.", "Sábado 09:00 às 13:00"),
        ("2.", "Domingo e Feriados Fechado"),
    ),
    "regret": (
        ("4.1", "Direito de Arrependimento (Compras Online)"),
        ("4.1", "em até 7 (sete) dias corridos após o recebimento do produto"),
    ),
    "preference": (
        ("4.3", "Trocas por preferência (cor, modelo, tamanho) são permitidas em até 7"),
    ),
    "not_exchangeable": (
        ("4.4", "Instrumentos com personalização ou ajustes sob encomenda"),
        ("4.4", "venda final"),
        ("4.4", "Boquilhas de instrumentos de sopro"),
    ),
    "defect": (
        ("4.2", "podem ser trocados em até 30 (trinta) dias corridos após a compra"),
        ("4.2", "pode intermediar o processo mediante solicitação"),
        ("8.1", "garantia legal de 90 (noventa) dias contra defeitos de fabricação"),
        ("8.1", "contados a partir da data de recebimento pelo cliente"),
    ),
    "metro_shipping": (
        ("5.1", "Frete grátis para pedidos acima de R$ 500,00"),
        ("5.1", "taxa fixa de R$ 35,00"),
    ),
    "complaints": (("7.3", "Prazo de retorno: 24 horas úteis"),),
    "mixed_payment": (
        (
            "3.1",
            "Combinação de formas de pagamento: permitida (ex.: PIX + cartão) "
            "para compras acima de R$ 2.000,00",
        ),
    ),
}

# Owner rulings for combined payments: two different methods, catalog prices without the PIX
# discount, and no boleto because it takes up to three business days to clear.
SPLIT_METHODS = ("pix", "credit", "debit")


def _flat(text: str) -> str:
    return " ".join(text.split())


def _matches(title: str, prefix: str) -> bool:
    return title == prefix or title.split(maxsplit=1)[0] == prefix


def _digits(text: str) -> str:
    return re.sub(r"\D", "", text)


class PolicyEvidence:
    """Locate rule evidence and the public contacts in the active manual sections."""

    def __init__(self, sections: Sequence[PolicySection]) -> None:
        """Find each rule's quotes; rules with any missing quote stay disabled."""
        self._sections = tuple(sections)
        self.whatsapp = self._whatsapp()
        self.email = self._email()
        self._found: dict[str, tuple[Citation, ...]] = {}
        for rule, quotes in RULE_EVIDENCE.items():
            located = [self._locate(section, quote) for section, quote in quotes]
            if all(located):
                self._found[rule] = tuple(
                    dict.fromkeys(c for group in located if group for c in group)
                )

    def _locate(self, prefix: str, quote: str) -> tuple[Citation, ...] | None:
        for section in self._sections:
            body = _flat(f"{section.title} {section.text}")
            if _matches(section.title, prefix) and _flat(quote) in body:
                return section.citations
        return None

    def _whatsapp(self) -> str | None:
        found = {
            match.group(1)
            for section in self._sections
            if _matches(section.title, "7.")
            for match in _WHATSAPP.finditer(_flat(section.text))
        }
        return found.pop() if len(found) == 1 else None

    def _email(self) -> str | None:
        found = {
            match.group(0)
            for section in self._sections
            if section.title == "Document overview"
            for match in _EMAIL.finditer(section.text)
        }
        return found.pop() if len(found) == 1 else None

    def has(self, *rules: str) -> bool:
        """Report whether the active manual contains the evidence for every named rule."""
        return all(rule in self._found for rule in rules)

    def citations(self, *rules: str) -> tuple[Citation, ...]:
        """Return the manual locations behind the named rules."""
        return tuple(
            dict.fromkeys(c for rule in rules if rule in self._found for c in self._found[rule])
        )

    def mask(self, text: str) -> str:
        """Hide every phone and email except the section 7 WhatsApp and the cover email."""
        phone = _digits(self.whatsapp) if self.whatsapp else None

        def phones(match: re.Match[str]) -> str:
            return match.group(0) if _digits(match.group(0)) == phone else MASK

        def emails(match: re.Match[str]) -> str:
            return match.group(0) if match.group(0) == self.email else MASK

        return _EMAIL.sub(emails, _PHONE.sub(phones, text))


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
            "The right of withdrawal covers online purchases only. Offer a preference exchange."
        )
        return result
    if reason in {"regret", "preference"}:
        if received is None:
            raise ValueError("Ask the customer for the date the product was received.")
        result.update(_window(received, 7, today))
        result["conditions"] = (
            "Original packaging, unused, with all accessories and manuals."
            if reason == "regret"
            else "Perfect condition, original packaging, and the desired product available."
        )
        return result
    if purchased is None or received is None:
        raise ValueError("Ask the customer for the purchase date and the receipt date.")
    result["store_exchange"] = _window(purchased, 30, today)
    result["legal_warranty_intermediation"] = _window(received, 90, today)
    result["conditions"] = (
        "Manufacturing defects only; misuse, falls, humidity and unauthorized changes are "
        "not covered."
    )
    return result
