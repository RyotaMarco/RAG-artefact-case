"""Owner-approved store facts, each confirmed verbatim against the active policy PDF."""

import re
from collections.abc import Sequence
from dataclasses import dataclass

from emporio_agent.documents import Citation, PolicySection

MASK = "[contato interno omitido]"
_PHONE = re.compile(r"(?:\(\d{2}\)|\b\d{2})\s*\d{4,5}-\d{4}\b")
_EMAIL = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
_WHATSAPP = re.compile(r"canal de whatsapp\s+(\(\d{2}\)\s*\d{4,5}-\d{4})", re.IGNORECASE)


@dataclass(frozen=True)
class Evidence:
    """Locate one verbatim quote inside a numbered manual section."""

    section: str
    quote: str


@dataclass(frozen=True)
class StoreFact:
    """State what the agent may say, the manual quotes behind it and any owner ruling."""

    key: str
    topic: str
    statement: str
    evidence: tuple[Evidence, ...]
    clarification: str | None = None


TOPICS = (
    "contact",
    "hours",
    "payments",
    "returns",
    "warranty",
    "shipping",
    "assortment",
    "complaints",
)

FACTS = (
    StoreFact(
        "whatsapp",
        "contact",
        "Customer support and special-date hour announcements use WhatsApp {whatsapp}.",
        (Evidence("7.", "O canal de WhatsApp {whatsapp}"),),
        "Company registration contacts are internal and are never given to customers.",
    ),
    StoreFact(
        "email",
        "contact",
        "The store email is {email}. WhatsApp {whatsapp} is faster.",
        (Evidence("Document overview", "{email}"), Evidence("7.", "{whatsapp}")),
    ),
    StoreFact(
        "address",
        "contact",
        "Address: Rua 14 de Maio, 3200, Centro, Campo Grande - MS, CEP 79202-333.",
        (Evidence("1.2", "Rua 14 de Maio, 3200 — Centro, Campo Grande - MS, 79202-333"),),
    ),
    StoreFact(
        "hours",
        "hours",
        "Monday to Friday 09:00-18:00, Saturday 09:00-13:00, closed on Sundays and holidays. "
        "Hours may be extended on special dates, announced on social media and WhatsApp "
        "{whatsapp}. Holidays are not known to the assistant; customers should confirm there.",
        (
            Evidence("2.", "Segunda a Sexta-feira 09:00 às 18:00"),
            Evidence("2.", "Sábado 09:00 às 13:00"),
            Evidence("2.", "Domingo e Feriados Fechado"),
            Evidence("2.", "o horário pode ser estendido"),
        ),
    ),
    StoreFact(
        "installments",
        "payments",
        "Credit card up to 12x without interest. Minimum installment: R$ 50,00 for 1x-3x, "
        "R$ 80,00 for 4x-6x and R$ 100,00 for 7x-12x. Use calculate_quote for any amount.",
        (
            Evidence("3.", "Cartão de Crédito Até 12x sem juros Parcela mínima de R$ 100,00"),
            Evidence("3.1", "Parcelamento em até 3x: sem juros"),
            Evidence("3.1", "(exceto abaixo de R$ 50,00)"),
            Evidence("3.1", "Parcelamento de 4x a 6x: sem juros, parcela mínima de R$ 80,00"),
            Evidence("3.1", "Parcelamento de 7x a 12x: sem juros, parcela mínima de R$ 100,00"),
        ),
        "The 'exceto abaixo de R$ 50,00' wording is a R$ 50,00 minimum installment; the "
        "table's R$ 100,00 minimum refers to the 7x-12x tier.",
    ),
    StoreFact(
        "mixed_payment",
        "payments",
        "Combining payment methods (for example PIX + card) is allowed only for purchases "
        "above R$ 2.000,00. The assistant does not split the amounts; staff close the values "
        "on WhatsApp {whatsapp}.",
        (
            Evidence(
                "3.1",
                "Combinação de formas de pagamento: permitida (ex.: PIX + cartão) "
                "para compras acima de R$ 2.000,00",
            ),
        ),
    ),
    StoreFact(
        "regret",
        "returns",
        "Right of withdrawal applies to online purchases only: up to 7 calendar days after "
        "receipt, without justification, in the original packaging and unused. The store pays "
        "the return shipping and refunds with the original payment method within 10 business "
        "days. In-store purchases have no withdrawal right; offer a preference exchange.",
        (
            Evidence("4.1", "Direito de Arrependimento (Compras Online)"),
            Evidence("4.1", "em até 7 (sete) dias corridos após o recebimento do produto"),
            Evidence("4.1", "embalagem original, sem sinais de uso"),
            Evidence("4.1", "na mesma forma de pagamento original em até 10 dias úteis"),
            Evidence("4.1", "O frete de devolução é por conta da loja em caso de arrependimento"),
        ),
    ),
    StoreFact(
        "preference_exchange",
        "returns",
        "Preference exchanges (color, model, size) are allowed up to 7 days after receipt, "
        "subject to availability, with the item in perfect condition and original packaging. "
        "Price differences are charged or refunded.",
        (
            Evidence(
                "4.3", "Trocas por preferência (cor, modelo, tamanho) são permitidas em até 7"
            ),
            Evidence("4.3", "na embalagem original"),
        ),
        "The 7 days count from receipt; for in-store purchases receipt is the purchase date.",
    ),
    StoreFact(
        "not_exchangeable",
        "returns",
        "Not eligible for exchange: customized instruments or special setup, items sold in "
        "clearance marked 'venda final', and wind instrument mouthpieces (hygiene).",
        (
            Evidence("4.4", "Instrumentos com personalização ou ajustes sob encomenda"),
            Evidence("4.4", "venda final"),
            Evidence("4.4", "Boquilhas de instrumentos de sopro"),
        ),
    ),
    StoreFact(
        "defect",
        "warranty",
        "Manufacturing defects: the store exchanges the item up to 30 calendar days after the "
        "purchase. Up to 90 days after receipt, the legal warranty of the Consumer Defense Code "
        "applies; it is the manufacturer's warranty and the store intermediates the request. "
        "Misuse, falls, humidity and unauthorized modifications are not covered. Contact "
        "WhatsApp {whatsapp}.",
        (
            Evidence("4.2", "podem ser trocados em até 30 (trinta) dias corridos após a compra"),
            Evidence("4.2", "pode intermediar o processo mediante solicitação"),
            Evidence("8.1", "garantia legal de 90 (noventa) dias contra defeitos de fabricação"),
            Evidence("8.1", "contados a partir da data de recebimento pelo cliente"),
        ),
    ),
    StoreFact(
        "metro_shipping",
        "shipping",
        "Campo Grande metropolitan region: free shipping for orders above R$ 500,00; R$ 35,00 "
        "otherwise, including exactly R$ 500,00. Delivery in 1 to 3 business days by the "
        "store's own courier. calculate_quote applies this to the final amount paid.",
        (
            Evidence("5.1", "Frete grátis para pedidos acima de R$ 500,00"),
            Evidence("5.1", "taxa fixa de R$ 35,00"),
            Evidence("5.1", "Prazo de entrega: 1 a 3 dias úteis"),
        ),
        "The R$ 500,00 limit uses the final amount paid, after promotions and the PIX discount.",
    ),
    StoreFact(
        "other_cities_shipping",
        "shipping",
        "Other cities ship by Correios (PAC, SEDEX) or Jadlog; the fee depends on destination "
        "CEP, weight and size, so staff quote it on WhatsApp {whatsapp}. Large instruments may "
        "need a special quote.",
        (
            Evidence("5.2", "O valor do frete é calculado automaticamente com base no CEP"),
            Evidence("5.2", "podem exigir frete especial com cotação individual"),
        ),
    ),
    StoreFact(
        "assortment",
        "assortment",
        "The catalog covers more than 300 instruments from national and international brands. "
        "Use query_data for current availability and prices.",
        (Evidence("1.", "mais de 300 instrumentos"),),
    ),
    StoreFact(
        "no_accessories",
        "assortment",
        "The store sells only musical instruments, not accessories such as strings, picks, "
        "cables, cases, pedals or amplifiers. Staff can recommend a partner store on WhatsApp "
        "{whatsapp}; never name partner stores.",
        (
            Evidence("1.", "não comercializamos acessórios como cordas, palhetas, cabos"),
            Evidence("7.1", "Sugerir lojas parceiras quando possível"),
        ),
    ),
    StoreFact(
        "complaints",
        "complaints",
        "Complaints are answered within 24 business hours. With a verified order, "
        "register_complaint records it and returns a protocol to send on WhatsApp {whatsapp}.",
        (Evidence("7.3", "Prazo de retorno: 24 horas úteis"),),
    ),
)


def _flat(text: str) -> str:
    return " ".join(text.split())


def _matches(title: str, prefix: str) -> bool:
    return title == prefix or title.split(maxsplit=1)[0] == prefix


def _digits(text: str) -> str:
    return re.sub(r"\D", "", text)


class StoreFacts:
    """Expose only facts whose quotes, and extracted contacts, the active manual confirms."""

    def __init__(self, sections: Sequence[PolicySection]) -> None:
        """Verify every fact against the supplied sections; unconfirmed facts are withheld."""
        self._sections = tuple(sections)
        self.whatsapp = self._whatsapp()
        self.email = self._email()
        values = {
            name: value
            for name, value in (("whatsapp", self.whatsapp), ("email", self.email))
            if value is not None
        }
        self._verified: dict[str, tuple[StoreFact, str, tuple[Citation, ...]]] = {}
        for fact in FACTS:
            try:
                statement = fact.statement.format(**values)
                quotes = [(item.section, item.quote.format(**values)) for item in fact.evidence]
            except KeyError:
                continue
            citations = [self._locate(section, quote) for section, quote in quotes]
            if all(citations):
                found = tuple(dict.fromkeys(c for group in citations if group for c in group))
                self._verified[fact.key] = (fact, statement, found)

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

    def has(self, *keys: str) -> bool:
        """Report whether every named fact is confirmed by the active manual."""
        return all(key in self._verified for key in keys)

    def keys(self, topic: str) -> tuple[str, ...]:
        """Return the confirmed fact keys for one topic."""
        return tuple(key for key, (fact, _, _) in self._verified.items() if fact.topic == topic)

    def citations(self, *keys: str) -> tuple[Citation, ...]:
        """Return the manual locations behind the named confirmed facts."""
        return tuple(
            dict.fromkeys(
                c for key in keys if key in self._verified for c in self._verified[key][2]
            )
        )

    def topic(self, name: str) -> list[dict[str, object]]:
        """Return confirmed facts for one topic with statements, rulings and sources."""
        return [
            {
                "statement": statement,
                "clarification": fact.clarification,
                "sources": [
                    {"file": c.source, "pages": c.pages, "section": c.section} for c in citations
                ],
            }
            for fact, statement, citations in self._verified.values()
            if fact.topic == name
        ]

    def mask(self, text: str) -> str:
        """Hide every phone and email except the confirmed public WhatsApp and store email."""
        phone = _digits(self.whatsapp) if self.whatsapp else None

        def phones(match: re.Match[str]) -> str:
            return match.group(0) if _digits(match.group(0)) == phone else MASK

        def emails(match: re.Match[str]) -> str:
            return match.group(0) if match.group(0) == self.email else MASK

        return _EMAIL.sub(emails, _PHONE.sub(phones, text))
