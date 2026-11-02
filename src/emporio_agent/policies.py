"""Extract supported monetary clauses and calculate only unambiguous quotes."""

import re
import unicodedata
from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal, localcontext

from emporio_agent.documents import Citation, PolicySection
from emporio_agent.money import canonical_rate


class PolicyUnavailableError(ValueError):
    """A source policy is missing, contradictory or ambiguous for a calculation."""


@dataclass(frozen=True)
class InstallmentTier:
    """Preserve a sourced installment range and any unresolved minimum wording."""

    minimum_count: int
    maximum_count: int
    minimum_amount: Decimal | None
    ambiguous: bool
    citations: tuple[Citation, ...]


@dataclass(frozen=True)
class PaymentQuote:
    """Return exact totals with the policy evidence used to calculate them."""

    total: Decimal
    installments: int
    installment_amount: Decimal
    discount_rate: Decimal
    citations: tuple[Citation, ...]
    note: str | None = None


@dataclass(frozen=True)
class FinancialRuleSet:
    """Store only values extracted from the active source documents."""

    pix_rate: Decimal | None
    installment_tiers: tuple[InstallmentTier, ...]
    non_cumulative: bool | None
    mixed_payment_minimum: Decimal | None
    conflicts: tuple[str, ...]
    citations: tuple[Citation, ...]
    single_payment_methods: tuple[str, ...] = ()

    def quote(
        self,
        price: Decimal,
        *,
        payment_method: str = "pix",
        installments: int = 1,
        promotional: bool = False,
    ) -> PaymentQuote:
        """Apply sourced payment rules without guessing missing conditions."""
        cents = Decimal("0.01")
        if not price.is_finite() or price < 0 or price != price.quantize(cents):
            raise ValueError(
                "Price must be a nonnegative, finite amount with at most two decimals."
            )
        if installments < 1:
            raise ValueError("Installment count must be positive.")
        if self.conflicts:
            raise PolicyUnavailableError("Conflicting source policies require staff confirmation.")
        method = payment_method.casefold().strip()
        method = {"debit_card": "debit", "bank_slip": "boleto"}.get(method, method)
        if method in self.single_payment_methods:
            if installments != 1:
                raise PolicyUnavailableError("This sourced payment method requires one payment.")
            return PaymentQuote(price, 1, price, Decimal(0), self.citations)
        if method == "pix":
            return self._pix_quote(price, installments, promotional)
        if method not in {"credit", "credit_card"}:
            raise PolicyUnavailableError("This payment method has no supported calculation rule.")
        return self._credit_quote(price, installments)

    def _pix_quote(self, price: Decimal, installments: int, promotional: bool) -> PaymentQuote:
        if installments != 1:
            raise PolicyUnavailableError("The supported PIX clause is a single payment.")
        if self.pix_rate is None:
            raise PolicyUnavailableError("No unambiguous PIX discount was extracted.")
        if promotional and self.non_cumulative is None:
            raise PolicyUnavailableError("Discount stacking is not established by the sources.")
        rate = Decimal(0) if promotional and self.non_cumulative else self.pix_rate
        total = (price * (Decimal(1) - rate)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
        note = (
            "PIX discount does not stack with a promotional price."
            if promotional and self.non_cumulative
            else None
        )
        return PaymentQuote(total, 1, total, rate, self.citations, note)

    def _credit_quote(self, price: Decimal, installments: int) -> PaymentQuote:
        matches = [
            tier
            for tier in self.installment_tiers
            if tier.minimum_count <= installments <= tier.maximum_count
        ]
        if len(matches) != 1 or matches[0].ambiguous:
            raise PolicyUnavailableError("Installment conditions are missing or ambiguous.")
        minimum = matches[0].minimum_amount
        exact_installment = price / installments
        if minimum is not None and exact_installment < minimum:
            raise PolicyUnavailableError(
                "The requested plan is below the sourced installment minimum."
            )
        rounded = exact_installment.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
        note = None
        if rounded * installments != price:
            note = "Installment amounts are approximate; checkout allocates any cent difference."
        return PaymentQuote(price, installments, rounded, Decimal(0), matches[0].citations, note)


def _normalized(text: str) -> str:
    decomposed = unicodedata.normalize("NFKD", text.casefold())
    return " ".join("".join(char for char in decomposed if not unicodedata.combining(char)).split())


def _amount(text: str) -> Decimal:
    return Decimal(text.replace(".", "").replace(",", "."))


def _pix_rules(text: str) -> tuple[set[Decimal], list[str]]:
    rates: set[Decimal] = set()
    conflicts: list[str] = []
    for pattern in (
        r"\bpix\b\s*:?\s*(?:pagamento\s+a\s+vista\s+)?"
        r"([+−-]?[0-9]+(?:[.,][0-9]+)?)\s*%\s+de\s+desconto",
        r"desconto\s+de\s+pix\s*\(\s*([+−-]?[0-9]+(?:[.,][0-9]+)?)\s*%\s*\)",
    ):
        for match in re.finditer(pattern, text):
            try:
                percentage = canonical_rate(match.group(1).replace(",", "."))
                with localcontext() as context:
                    context.prec = 100
                    rates.add(Decimal(percentage) / 100)
            except ValueError:
                conflicts.append("PIX discount is invalid or has unsupported precision.")
    return rates, conflicts


def _single_payment_rules(text: str) -> tuple[set[str], list[str]]:
    methods: set[str] = set()
    conflicts: list[str] = []
    boundary = (
        r"(?=cartao de credito|cartao de debito|boleto bancario|\bpix\b|\d+(?:\.\d+)+\s+[a-z]|$)"
    )
    for label, method, observation in (
        ("cartao de debito", "debit", r"todas as bandeiras aceitas"),
        ("boleto bancario", "boleto", r"compensacao em ate \d+ (?:dias uteis|dia util)"),
    ):
        affirmative = r"pagamento a vista(?: " + observation + r")?\.?"
        for row in re.finditer(label + r"\s+(.*?)" + boundary, text):
            if re.fullmatch(affirmative, row.group(1).strip()):
                methods.add(method)
            else:
                conflicts.append(f"Unsupported or contradictory conditions for {method} payments.")
    return methods, conflicts


def _stacking_rules(text: str) -> set[bool]:
    stacking: set[bool] = set()
    if (
        "promocoes nao sao cumulativas" in text
        or "pix" in text
        and "nao se aplica sobre precos ja promocionais" in text
    ):
        stacking.add(True)
    if "promocoes sao cumulativas" in text:
        stacking.add(False)
    return stacking


_MONEY_PATTERN = r"([0-9]+(?:\.[0-9]{3})*(?:,[0-9]{1,2})?)"


def _installment_rules(
    text: str, citations: tuple[Citation, ...]
) -> tuple[list[InstallmentTier], list[str]]:
    tiers: list[InstallmentTier] = []
    conflicts: list[str] = []
    tier_pattern = re.compile(
        r"parcelamento\s+(?:em\s+ate\s+(\d+)x|de\s+(\d+)x\s+a\s+(\d+)x)"
        r"\s*:\s*(.*?)(?=\s*•|\s*parcelamento\s|\s*combinacao\s|$)"
    )
    for match in tier_pattern.finditer(text):
        low = 1 if match.group(1) else int(match.group(2))
        high = int(match.group(1) or match.group(3))
        if low < 1 or high < low:
            conflicts.append("An installment range is invalid.")
            continue
        terms = match.group(4)
        amount_match = re.search(r"parcela\s+minima\s+de\s+r\$\s*" + _MONEY_PATTERN, terms)
        minimum = _amount(amount_match.group(1)) if amount_match else None
        # "Sem valor minimo (exceto abaixo de R$ X)" states a R$ X installment floor.
        floor = re.search(
            r"sem valor minimo de parcela\s*\(\s*exceto abaixo de r\$\s*"
            + _MONEY_PATTERN
            + r"\s*\)",
            terms,
        )
        if minimum is None and floor is not None:
            minimum = _amount(floor.group(1))
            terms = terms.replace(floor.group(0), "")
        ambiguous = "exceto" in terms or "sem juros" not in terms
        if minimum is None and "sem valor minimo" not in terms:
            ambiguous = True
        tiers.append(InstallmentTier(low, high, minimum, ambiguous, citations))
    return tiers, conflicts


def _merge_tiers(
    tiers: dict[tuple[int, int], InstallmentTier], candidates: list[InstallmentTier]
) -> list[str]:
    conflicts: list[str] = []
    for tier in candidates:
        key = (tier.minimum_count, tier.maximum_count)
        previous = tiers.get(key)
        if previous is not None:
            if (previous.minimum_amount, previous.ambiguous) != (
                tier.minimum_amount,
                tier.ambiguous,
            ):
                conflicts.append(
                    f"Conflicting installment conditions for {key[0]} to {key[1]} payments."
                )
            else:
                tier = InstallmentTier(
                    *key,
                    tier.minimum_amount,
                    tier.ambiguous,
                    tuple(dict.fromkeys((*previous.citations, *tier.citations))),
                )
        tiers[key] = tier
    return conflicts


def _rule_conflicts(
    rates: set[Decimal], stacking: set[bool], mixed: set[Decimal], tiers: list[InstallmentTier]
) -> list[str]:
    conflicts: list[str] = []
    if len(rates) > 1:
        conflicts.append("Conflicting PIX discount percentages in the active PDFs.")
    if len(stacking) > 1:
        conflicts.append("Conflicting discount stacking conditions in the active PDFs.")
    if len(mixed) > 1:
        conflicts.append("Conflicting mixed-payment thresholds in the active PDFs.")
    for left, right in zip(tiers, tiers[1:]):
        if left.maximum_count >= right.minimum_count:
            conflicts.append("Installment ranges overlap and need staff confirmation.")
    return conflicts


def extract_rules(sections: tuple[PolicySection, ...]) -> FinancialRuleSet:
    """Parse supported same-template clauses; never fill gaps with fixture defaults."""
    rates: set[Decimal] = set()
    stacking: set[bool] = set()
    mixed: set[Decimal] = set()
    tiers: dict[tuple[int, int], InstallmentTier] = {}
    citations: list[Citation] = []
    single_payment_methods: set[str] = set()
    conflicts: list[str] = []
    for section in sections:
        text = _normalized(section.text)
        section_rates, pix_errors = _pix_rules(text)
        methods, method_errors = _single_payment_rules(text)
        section_tiers, tier_errors = _installment_rules(text, section.citations)
        section_stacking = _stacking_rules(text)
        rates.update(section_rates)
        single_payment_methods.update(methods)
        stacking.update(section_stacking)
        conflicts.extend(
            pix_errors + method_errors + tier_errors + _merge_tiers(tiers, section_tiers)
        )
        combined = re.search(
            r"combinacao de formas de pagamento:.*?acima de\s+r\$\s*" + _MONEY_PATTERN, text
        )
        if combined:
            mixed.add(_amount(combined.group(1)))
        if any(
            (
                section_rates,
                methods,
                section_tiers,
                section_stacking,
                pix_errors,
                method_errors,
                tier_errors,
                combined,
            )
        ):
            citations.extend(section.citations)
    ordered = sorted(tiers.values(), key=lambda tier: (tier.minimum_count, tier.maximum_count))
    conflicts.extend(_rule_conflicts(rates, stacking, mixed, ordered))
    return FinancialRuleSet(
        next(iter(rates)) if len(rates) == 1 else None,
        tuple(ordered),
        next(iter(stacking)) if len(stacking) == 1 else None,
        next(iter(mixed)) if len(mixed) == 1 else None,
        tuple(dict.fromkeys(conflicts)),
        tuple(dict.fromkeys(citations)),
        tuple(sorted(single_payment_methods)),
    )
