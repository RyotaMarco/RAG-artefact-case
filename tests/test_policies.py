"""Check dynamic source rules, contradictions and exact Decimal payment quotes."""

from decimal import Decimal
from pathlib import Path

import pytest
from pypdf import PdfReader

from emporio_agent.documents import Citation, PolicySection
from emporio_agent.policies import PolicyUnavailableError, extract_rules


def section(text: str, source: str = "policy.pdf") -> PolicySection:
    """Attach evidence to an independently supplied policy clause."""
    return PolicySection("Payments", text, (Citation(source, (3,), "Payments"),))


@pytest.mark.parametrize("percentage", ["-5", "−5", "105", "50." + "0" * 150 + "1"])
def test_invalid_source_percentage_never_becomes_a_different_discount(percentage: str) -> None:
    """Keep invalid signs, ranges or unsupported precision from producing a usable quote."""
    rules = extract_rules((section(f"PIX {percentage}% de desconto."),))
    assert rules.pix_rate is None
    assert rules.conflicts
    with pytest.raises(PolicyUnavailableError):
        rules.quote(Decimal("100.00"))


def test_source_percentage_preserves_valid_decimal_digits() -> None:
    """Rule extraction must not inherit Decimal's default 28-digit rounding context."""
    percentage = "7." + "1" * 45
    rules = extract_rules((section(f"PIX {percentage}% de desconto."),))
    assert rules.pix_rate == Decimal(percentage + "E-2")


@pytest.mark.parametrize(
    "text", ["PIX não oferece 5% de desconto.", "PIX sem desconto. Cartão: 5% de desconto."]
)
def test_unrecognized_payment_wording_does_not_borrow_a_nearby_percentage(text: str) -> None:
    """Only supported affirmative PIX clauses may enable automatic discount calculations."""
    rules = extract_rules((section(text),))
    assert rules.pix_rate is None
    with pytest.raises(PolicyUnavailableError):
        rules.quote(Decimal("100.00"))


def test_rates_and_thresholds_come_from_source_values() -> None:
    """Modified input amounts must change the quote without implementation changes."""
    rules = extract_rules(
        (
            section(
                "PIX Pagamento à vista 7,5% de desconto sobre o preço de tabela. "
                "Parcelamento de 4x a 8x: sem juros, parcela mínima de R$ 90,00. "
                "Promoções não são cumulativas."
            ),
        )
    )
    assert rules.pix_rate == Decimal("0.075")
    assert rules.quote(Decimal("1234.56")).total == Decimal("1141.97")
    assert rules.quote(
        Decimal("800.00"), payment_method="credit", installments=8
    ).installment_amount == Decimal("100.00")
    assert rules.quote(Decimal("800.00"), promotional=True).total == Decimal("800.00")
    with pytest.raises(PolicyUnavailableError, match="minimum"):
        rules.quote(Decimal("700.00"), payment_method="credit", installments=8)


def test_conflicting_added_pdf_disables_quotes_and_removal_restores_them() -> None:
    """Never pick the newest filename or retrieval score as monetary authority."""
    original = section("PIX 5% de desconto sobre o preço de tabela.", "old.pdf")
    replacement = section("PIX 9% de desconto sobre o preço de tabela.", "new.pdf")
    conflicting = extract_rules((original, replacement))
    assert conflicting.pix_rate is None
    assert conflicting.conflicts
    with pytest.raises(PolicyUnavailableError, match="Conflicting"):
        conflicting.quote(Decimal("100.00"))
    assert extract_rules((replacement,)).quote(Decimal("100.00")).total == Decimal("91.00")


def test_duplicate_rules_merge_evidence_without_conflict() -> None:
    """Equivalent policy sources may share one interpreted value."""
    rules = extract_rules(
        (
            section("PIX 6% de desconto sobre o preço de tabela.", "a.pdf"),
            section("PIX 6% de desconto sobre o preço de tabela.", "b.pdf"),
        )
    )
    assert not rules.conflicts
    assert len(rules.citations) == 2
    assert rules.quote(Decimal("100.00")).total == Decimal("94.00")


def test_ambiguous_exception_never_becomes_an_invented_minimum() -> None:
    """The supplied under-three-installment exception requires clarification."""
    rules = extract_rules(
        (
            section(
                "Parcelamento em até 3x: sem juros, sem valor mínimo de parcela "
                "(exceto abaixo de R$ 50,00)."
            ),
        )
    )
    assert rules.installment_tiers[0].ambiguous
    with pytest.raises(PolicyUnavailableError, match="ambiguous"):
        rules.quote(Decimal("600.00"), payment_method="credit", installments=3)


def test_actual_manual_extracts_payment_values_without_fallbacks() -> None:
    """Check layout text from the original PDF, including repeated PIX rules."""
    path = next((Path(__file__).parents[1] / "data").glob("*.pdf"))
    sections = tuple(
        PolicySection(
            f"Page {number}",
            page.extract_text(extraction_mode="layout"),
            (Citation(path.name, (number,), f"Page {number}"),),
        )
        for number, page in enumerate(PdfReader(path).pages, start=1)
    )
    rules = extract_rules(sections)
    assert not rules.conflicts
    assert rules.pix_rate == Decimal("0.05")
    assert rules.non_cumulative is True
    assert rules.mixed_payment_minimum == Decimal("2000.00")
    assert set(rules.single_payment_methods) == {"debit", "boleto"}
    assert rules.quote(Decimal("100.01"), payment_method="debit").total == Decimal("100.01")
    assert rules.quote(Decimal("100.01"), payment_method="boleto").total == Decimal("100.01")
    assert {(tier.minimum_count, tier.maximum_count) for tier in rules.installment_tiers} == {
        (1, 3),
        (4, 6),
        (7, 12),
    }
    assert rules.quote(
        Decimal("1200.00"), payment_method="credit", installments=12
    ).installment_amount == Decimal("100.00")


def test_missing_rules_never_use_fixture_defaults() -> None:
    """A new document without payment clauses cannot inherit old percentages."""
    rules = extract_rules((section("The store opens on weekdays."),))
    assert rules.pix_rate is None
    with pytest.raises(PolicyUnavailableError, match="No unambiguous"):
        rules.quote(Decimal("100.00"))


@pytest.mark.parametrize("method", ["debit", "debit_card", "boleto", "bank_slip"])
def test_single_payment_methods_require_explicit_source_declaration(method: str) -> None:
    """Accept debit or boleto only when the source declares an unqualified single payment."""
    rules = extract_rules(
        (
            section(
                "Cartão de Débito Pagamento à vista Todas as bandeiras aceitas. "
                "Boleto Bancário Pagamento à vista Compensação em até 3 dias úteis."
            ),
        )
    )
    assert rules.quote(Decimal("123.45"), payment_method=method).total == Decimal("123.45")
    with pytest.raises(PolicyUnavailableError, match="one payment"):
        rules.quote(Decimal("123.45"), payment_method=method, installments=2)
    with pytest.raises(PolicyUnavailableError, match="no supported"):
        extract_rules(()).quote(Decimal("123.45"), payment_method=method)


@pytest.mark.parametrize(
    ("method", "text"),
    [
        ("debit", "Cartão de Débito Pagamento à vista Não aceitamos esta forma de pagamento."),
        ("debit", "Cartão de Débito Não aceitamos pagamento à vista."),
        ("debit", "Cartão de Débito Pagamento à vista Somente compras presenciais."),
        ("boleto", "Boleto Bancário Pagamento à vista Não disponível para novas compras."),
        ("boleto", "Boleto Bancário Não aceitamos pagamento à vista."),
        ("boleto", "Boleto Bancário Pagamento à vista Exige cadastro aprovado."),
    ],
)
def test_single_payment_denials_and_unknown_restrictions_never_enable_quotes(
    method: str, text: str
) -> None:
    """An affirmative substring cannot erase a refusal or an unsupported condition."""
    rules = extract_rules((section(text),))
    assert method not in rules.single_payment_methods
    assert rules.conflicts
    with pytest.raises(PolicyUnavailableError, match="Conflicting"):
        rules.quote(Decimal("100.00"), payment_method=method)


def test_repeated_method_row_cannot_hide_a_later_denial() -> None:
    """Inspect every occurrence and retain evidence for conflicting method conditions."""
    rules = extract_rules(
        (
            section(
                "Cartão de Débito Pagamento à vista Todas as bandeiras aceitas. "
                "Cartão de Débito Pagamento à vista Não aceitamos esta forma de pagamento."
            ),
        )
    )
    assert rules.conflicts
    assert rules.citations
    with pytest.raises(PolicyUnavailableError):
        rules.quote(Decimal("100.00"), payment_method="debit")


def test_supported_settlement_days_are_source_values() -> None:
    """Retain affirmative boleto conditions when the documented settlement period changes."""
    rules = extract_rules(
        (section("Boleto Bancário Pagamento à vista Compensação em até 5 dias úteis."),)
    )
    assert not rules.conflicts
    assert rules.quote(Decimal("100.00"), payment_method="boleto").total == Decimal("100.00")
