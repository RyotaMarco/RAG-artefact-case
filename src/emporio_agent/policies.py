"""Extract supported monetary clauses and calculate only unambiguous quotes."""

import re
import unicodedata
from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal

from emporio_agent.documents import Citation, PolicySection


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
            if installments != 1:
                raise PolicyUnavailableError("The supported PIX clause is a single payment.")
            if self.pix_rate is None:
                raise PolicyUnavailableError("No unambiguous PIX discount was extracted.")
            if promotional and self.non_cumulative is None:
                raise PolicyUnavailableError("Discount stacking is not established by the sources.")
            rate = Decimal(0) if promotional and self.non_cumulative else self.pix_rate
            total = (price * (Decimal(1) - rate)).quantize(cents, rounding=ROUND_HALF_UP)
            note = (
                "PIX discount does not stack with a promotional price."
                if promotional and self.non_cumulative
                else None
            )
            return PaymentQuote(total, 1, total, rate, self.citations, note)
        if method not in {"credit", "credit_card"}:
            raise PolicyUnavailableError("This payment method has no supported calculation rule.")
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
        rounded = exact_installment.quantize(cents, rounding=ROUND_HALF_UP)
        note = None
        if rounded * installments != price:
            note = "Installment amounts are approximate; checkout allocates any cent difference."
        return PaymentQuote(price, installments, rounded, Decimal(0), matches[0].citations, note)


def _normalized(text: str) -> str:
    decomposed = unicodedata.normalize("NFKD", text.casefold())
    return " ".join("".join(char for char in decomposed if not unicodedata.combining(char)).split())


def _amount(text: str) -> Decimal:
    return Decimal(text.replace(".", "").replace(",", "."))


def extract_rules(sections: tuple[PolicySection, ...]) -> FinancialRuleSet:
    """Parse supported same-template clauses; never fill gaps with fixture defaults."""
    rates: set[Decimal] = set()
    stacking: set[bool] = set()
    mixed: set[Decimal] = set()
    tiers: dict[tuple[int, int], InstallmentTier] = {}
    citations: list[Citation] = []
    single_payment_methods: set[str] = set()
    conflicts: list[str] = []
    money = r"([0-9]+(?:\.[0-9]{3})*(?:,[0-9]{1,2})?)"
    tier_pattern = re.compile(
        r"parcelamento\s+(?:em\s+ate\s+(\d+)x|de\s+(\d+)x\s+a\s+(\d+)x)"
        r"\s*:\s*(.*?)(?=\s*•|\s*parcelamento\s|\s*combinacao\s|$)"
    )
    for section in sections:
        text = _normalized(section.text)
        found = False
        for pattern in (
            r"\bpix\b[^%]{0,100}?([0-9]+(?:[.,][0-9]+)?)\s*%\s+de\s+desconto",
            r"desconto\s+de\s+pix\s*\(\s*([0-9]+(?:[.,][0-9]+)?)\s*%\s*\)",
        ):
            for match in re.finditer(pattern, text):
                rate = Decimal(match.group(1).replace(",", ".")) / 100
                if not Decimal(0) <= rate <= Decimal(1):
                    conflicts.append("PIX discount is outside the valid percentage range.")
                else:
                    rates.add(rate)
                found = True
        if (
            "promocoes nao sao cumulativas" in text
            or "pix" in text
            and "nao se aplica sobre precos ja promocionais" in text
        ):
            stacking.add(True)
            found = True
        if "promocoes sao cumulativas" in text:
            stacking.add(False)
            found = True
        for label, method in (("cartao de debito", "debit"), ("boleto bancario", "boleto")):
            row = re.search(
                label
                + r"\s+(.*?)(?=cartao de credito|cartao de debito|boleto bancario|\bpix\b|\d+(?:\.\d+)+\s+[a-z]|$)",
                text,
            )
            if (
                row is not None
                and "pagamento a vista" in row.group(1)
                and not re.search(r"desconto|juros|taxa|r\$", row.group(1))
            ):
                single_payment_methods.add(method)
                found = True
            elif row is not None and "pagamento a vista" in row.group(1):
                conflicts.append(f"Unsupported additional conditions for {method} payments.")
        for match in tier_pattern.finditer(text):
            low = 1 if match.group(1) else int(match.group(2))
            high = int(match.group(1) or match.group(3))
            terms = match.group(4)
            amount_match = re.search(r"parcela\s+minima\s+de\s+r\$\s*" + money, terms)
            minimum = _amount(amount_match.group(1)) if amount_match else None
            ambiguous = "exceto" in terms or "sem juros" not in terms
            if minimum is None and "sem valor minimo" not in terms:
                ambiguous = True
            if low < 1 or high < low:
                conflicts.append("An installment range is invalid.")
                continue
            tier = InstallmentTier(low, high, minimum, ambiguous, section.citations)
            previous = tiers.get((low, high))
            if previous is not None:
                if (previous.minimum_amount, previous.ambiguous) != (minimum, ambiguous):
                    conflicts.append(
                        f"Conflicting installment conditions for {low} to {high} payments."
                    )
                else:
                    tier = InstallmentTier(
                        low,
                        high,
                        minimum,
                        ambiguous,
                        tuple(dict.fromkeys((*previous.citations, *section.citations))),
                    )
            tiers[(low, high)] = tier
            found = True
        combined = re.search(
            r"combinacao de formas de pagamento:.*?acima de\s+r\$\s*" + money, text
        )
        if combined:
            mixed.add(_amount(combined.group(1)))
            found = True
        if found:
            citations.extend(section.citations)
    if len(rates) > 1:
        conflicts.append("Conflicting PIX discount percentages in the active PDFs.")
    if len(stacking) > 1:
        conflicts.append("Conflicting discount stacking conditions in the active PDFs.")
    if len(mixed) > 1:
        conflicts.append("Conflicting mixed-payment thresholds in the active PDFs.")
    ordered = sorted(tiers.values(), key=lambda tier: (tier.minimum_count, tier.maximum_count))
    for left, right in zip(ordered, ordered[1:]):
        if left.maximum_count >= right.minimum_count:
            conflicts.append("Installment ranges overlap and need staff confirmation.")
    return FinancialRuleSet(
        next(iter(rates)) if len(rates) == 1 else None,
        tuple(ordered),
        next(iter(stacking)) if len(stacking) == 1 else None,
        next(iter(mixed)) if len(mixed) == 1 else None,
        tuple(dict.fromkeys(conflicts)),
        tuple(dict.fromkeys(citations)),
        tuple(sorted(single_payment_methods)),
    )
