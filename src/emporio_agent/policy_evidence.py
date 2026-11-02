"""Manual quotes each deterministic store rule depends on, and internal-contact masking."""

import re
from collections.abc import Sequence

from emporio_agent.contacts import EMAIL, PHONE, phone_digits
from emporio_agent.documents import Citation, PolicySection

MASK = "[contato interno omitido]"
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
        ("4.1", "embalagem original, sem sinais de uso, com todos os acessórios e manuais"),
        ("4.1", "na mesma forma de pagamento original em até 10 dias úteis"),
        ("4.1", "O frete de devolução é por conta da loja em caso de arrependimento"),
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
        ("4.2", "Após os 30 dias, o cliente deve acionar a garantia diretamente com o fabricante"),
        ("4.2", "pode intermediar o processo mediante solicitação"),
        ("8.2", "pode variar de 6 meses a 2 anos"),
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


def _flat(text: str) -> str:
    return " ".join(text.split())


def _matches(title: str, prefix: str) -> bool:
    return title == prefix or title.split(maxsplit=1)[0] == prefix


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
            for match in EMAIL.finditer(section.text)
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
        phone = phone_digits(self.whatsapp) if self.whatsapp else None

        def phones(match: re.Match[str]) -> str:
            return match.group(0) if phone_digits(match.group(0)) == phone else MASK

        def emails(match: re.Match[str]) -> str:
            return match.group(0) if match.group(0) == self.email else MASK

        return EMAIL.sub(emails, PHONE.sub(phones, text))
