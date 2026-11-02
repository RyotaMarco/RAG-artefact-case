"""Manual values and quotes each deterministic store rule depends on, and contact masking."""

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import time

from emporio_agent.contacts import EMAIL, PHONE, phone_digits
from emporio_agent.documents import Citation, PolicySection
from emporio_agent.money import parse_customer_amount

MASK = "[contato interno omitido]"
_WHATSAPP = re.compile(r"canal de whatsapp\s+(\(\d{2}\)\s*\d{4,5}-\d{4})", re.IGNORECASE)
_MONEY = r"(\d[\d.]*,\d{2})"
_CLOCK = r"(\d{1,2}):(\d{2})\s+às\s+(\d{1,2}):(\d{2})"

# A tool applies each rule only while the active manual still contains these quotes in the
# named section. The numbers themselves are read from the manual (ManualValues), so a changed
# amount or deadline is picked up, and a missing or contradictory one disables the rule.
RULE_EVIDENCE: dict[str, tuple[tuple[str, str], ...]] = {
    "hours": (("2.", "Domingo e Feriados Fechado"),),
    "regret": (
        ("4.1", "Direito de Arrependimento (Compras Online)"),
        ("4.1", "embalagem original, sem sinais de uso, com todos os acessórios e manuais"),
        ("4.1", "na mesma forma de pagamento original"),
        ("4.1", "O frete de devolução é por conta da loja em caso de arrependimento"),
    ),
    "preference": (("4.3", "Trocas por preferência (cor, modelo, tamanho) são permitidas"),),
    "not_exchangeable": (
        ("4.4", "Instrumentos com personalização ou ajustes sob encomenda"),
        ("4.4", "venda final"),
        ("4.4", "Boquilhas de instrumentos de sopro"),
    ),
    "defect": (
        ("4.2", "o cliente deve acionar a garantia diretamente com o fabricante"),
        ("4.2", "pode intermediar o processo mediante solicitação"),
        ("8.2", "garantia própria"),
    ),
    "metro_shipping": (("5.1", "Frete grátis para pedidos acima de"),),
    "complaints": (("7.3", "Prazo de retorno"),),
    "mixed_payment": (("3.1", "Combinação de formas de pagamento: permitida (ex.: PIX + cartão)"),),
}


@dataclass(frozen=True)
class ManualValues:
    """Numbers and terms read from the manual, or None when absent or contradictory."""

    weekly_hours: Mapping[int, tuple[time, time]] | None = None
    metro_threshold_cents: int | None = None
    metro_fee_cents: int | None = None
    regret_days: int | None = None
    refund_business_days: int | None = None
    preference_days: int | None = None
    defect_days: int | None = None
    complaint_hours: int | None = None
    manufacturer_warranty: str | None = None


# Values each rule needs in addition to its quotes.
RULE_VALUES: dict[str, tuple[str, ...]] = {
    "hours": ("weekly_hours",),
    "regret": ("regret_days", "refund_business_days"),
    "preference": ("preference_days",),
    "defect": ("defect_days", "manufacturer_warranty"),
    "metro_shipping": ("metro_threshold_cents", "metro_fee_cents"),
    "complaints": ("complaint_hours",),
}


def _flat(text: str) -> str:
    return " ".join(text.split())


def _matches(title: str, prefix: str) -> bool:
    return title == prefix or title.split(maxsplit=1)[0] == prefix


def _unique(pattern: str, text: str) -> tuple[str, ...] | None:
    """Return the groups of a pattern stated exactly once (repeats must agree), else None."""
    found = {match.groups() for match in re.finditer(pattern, text, re.IGNORECASE)}
    return found.pop() if len(found) == 1 else None


def _slot(groups: tuple[str, ...] | None) -> tuple[time, time] | None:
    if groups is None:
        return None
    opening, closing = time(int(groups[0]), int(groups[1])), time(int(groups[2]), int(groups[3]))
    return (opening, closing) if opening < closing else None


class PolicyEvidence:
    """Locate rule evidence, manual values and the public contacts in the manual sections."""

    def __init__(self, sections: Sequence[PolicySection]) -> None:
        """Find each rule's quotes and values; rules missing either stay disabled."""
        self._sections = tuple(sections)
        self.whatsapp = self._whatsapp()
        self.email = self._email()
        self.values = self._values()
        self._found: dict[str, tuple[Citation, ...]] = {}
        for rule, quotes in RULE_EVIDENCE.items():
            located = [self._locate(section, quote) for section, quote in quotes]
            needed = RULE_VALUES.get(rule, ())
            if all(located) and all(getattr(self.values, name) is not None for name in needed):
                self._found[rule] = tuple(
                    dict.fromkeys(c for group in located if group for c in group)
                )

    def _locate(self, prefix: str, quote: str) -> tuple[Citation, ...] | None:
        for section in self._sections:
            body = _flat(f"{section.title} {section.text}")
            if _matches(section.title, prefix) and _flat(quote) in body:
                return section.citations
        return None

    def _text(self, prefix: str) -> str:
        return " ".join(
            _flat(section.text) for section in self._sections if _matches(section.title, prefix)
        )

    def _days(self, prefix: str, pattern: str) -> int | None:
        groups = _unique(pattern, self._text(prefix))
        return int(groups[0]) if groups else None

    def _values(self) -> ManualValues:
        hours = self._text("2.")
        weekdays = _slot(_unique(r"segunda a sexta-feira\s+" + _CLOCK, hours))
        saturday = _slot(_unique(r"sábado\s+" + _CLOCK, hours))
        sunday_closed = re.search(r"domingo e feriados fechado", hours, re.IGNORECASE)
        weekly = (
            {day: weekdays for day in range(5)} | {5: saturday}
            if weekdays and saturday and sunday_closed
            else None
        )
        shipping = self._text("5.1")
        free = _unique(r"frete grátis para pedidos acima de r\$\s*" + _MONEY, shipping)
        fee = _unique(
            r"pedidos abaixo de r\$\s*" + _MONEY + r",\s*taxa fixa de r\$\s*" + _MONEY, shipping
        )
        threshold = parse_customer_amount(free[0]) if free else None
        below = parse_customer_amount(fee[0]) if fee else None
        defect = self._days("4.2", r"trocados em até (\d+) \(\w+\) dias corridos após a compra")
        after = self._days("4.2", r"após os (\d+) dias")
        warranty = _unique(r"pode variar de ([^.]+?)\.", self._text("8.2"))
        return ManualValues(
            weekly_hours=weekly,
            metro_threshold_cents=threshold if threshold == below else None,
            metro_fee_cents=parse_customer_amount(fee[1]) if fee and threshold == below else None,
            regret_days=self._days("4.1", r"em até (\d+) \(\w+\) dias corridos após o recebimento"),
            refund_business_days=self._days("4.1", r"em até (\d+) dias úteis"),
            preference_days=self._days("4.3", r"permitidas em até (\d+) dias"),
            defect_days=defect if defect == after else None,
            complaint_hours=self._days("7.3", r"prazo de retorno:\s*(\d+) horas úteis"),
            manufacturer_warranty=warranty[0] if warranty else None,
        )

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
            for match in EMAIL.finditer(section.text)
        }
        return found.pop() if len(found) == 1 else None

    def has(self, *rules: str) -> bool:
        """Report whether the active manual supports every named rule (quotes and values)."""
        return all(rule in self._found for rule in rules)

    def citations(self, *rules: str) -> tuple[Citation, ...]:
        """Return the manual locations behind the named rules."""
        return tuple(
            dict.fromkeys(c for rule in rules if rule in self._found for c in self._found[rule])
        )

    def mask(self, text: str) -> str:
        """Hide every phone and email except the section 7 WhatsApp and the cover email."""
        phone = phone_digits(self.whatsapp) if self.whatsapp else None

        def phones(match: re.Match[str]) -> str:
            return match.group(0) if phone_digits(match.group(0)) == phone else MASK

        def emails(match: re.Match[str]) -> str:
            return match.group(0) if match.group(0) == self.email else MASK

        return EMAIL.sub(emails, PHONE.sub(phones, text))
