"""Check rule evidence, contact masking and the deterministic store tools on the real manual."""

import re
from dataclasses import replace
from datetime import date, datetime
from pathlib import Path
from typing import cast
from zoneinfo import ZoneInfo

import pytest
from agent_fixtures import FixedEmbedder, manual_snapshot, small_snapshot

from emporio_agent.agent import ConversationAccess, StoreTools, conversation_token
from emporio_agent.documents import PolicyIndex, PolicySection
from emporio_agent.policy_evidence import (
    MASK,
    RULE_EVIDENCE,
    PolicyEvidence,
    metro_shipping_cents,
    opening_status,
    return_window,
)
from emporio_agent.runtime import Snapshot

PHONE = re.compile(r"\(?\d{2}\)?\s*\d{4,5}-\d{4}")
EMAIL = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")


@pytest.fixture(scope="module")
def manual() -> Snapshot:
    """Parse the delivered sources once for every check in this module."""
    return manual_snapshot()


def sections(snapshot: Snapshot) -> tuple[PolicySection, ...]:
    """Narrow the optional policy index of a snapshot that always has a PDF."""
    assert snapshot.policies is not None
    return snapshot.policies.sections


def tools(snapshot: Snapshot, tmp_path: Path) -> StoreTools:
    """Bind real tools to a private conversation without starting a model."""
    access = ConversationAccess(tmp_path / "sessions.sqlite3", conversation_token(), "generation")
    return StoreTools(snapshot, access)


def test_every_rule_finds_its_quotes_in_the_delivered_manual(manual: Snapshot) -> None:
    """Each deterministic rule is backed verbatim by its numbered manual section."""
    evidence = PolicyEvidence(sections(manual))
    assert evidence.whatsapp == "(67) 3321-4500"
    assert evidence.email == "contato@emporiodamusica.com.br"
    for rule in RULE_EVIDENCE:
        assert evidence.has(rule), rule
    assert {c.section for c in evidence.citations("defect")} == {
        "4.2 Trocas por Defeito",
        "8.1 Garantia Legal",
    }


def test_a_missing_quote_disables_only_that_rule(manual: Snapshot, tmp_path: Path) -> None:
    """A changed manual turns off the rule it no longer supports and keeps the rest."""
    edited = tuple(
        replace(section, text=section.text.replace("acima de R$ 500,00", "acima de R$ 600,00"))
        for section in sections(manual)
    )
    evidence = PolicyEvidence(edited)
    assert not evidence.has("metro_shipping")
    assert evidence.has("hours", "regret", "defect", "complaints")
    assert manual.policies is not None
    policies = PolicyIndex(
        manual.policies.chunks, edited, FixedEmbedder(), manual.policies.fingerprint
    )
    quote = tools(replace(manual, policies=policies), tmp_path).calculate_quote(89)
    assert "metro_region_shipping" not in quote
    assert "not confirmed" in str(quote["shipping_note"])


def test_an_ambiguous_whatsapp_channel_offers_none(manual: Snapshot) -> None:
    """Two WhatsApp channels in section 7 are ambiguous, so every number is masked."""
    edited = tuple(
        replace(section, text=section.text + " O canal de WhatsApp (67) 0000-0000 também.")
        if section.title.startswith("7. ")
        else section
        for section in sections(manual)
    )
    evidence = PolicyEvidence(edited)
    assert evidence.whatsapp is None
    assert MASK in evidence.mask(" ".join(section.text for section in edited))
    assert not PHONE.search(evidence.mask(" ".join(section.text for section in edited)))


def test_tools_fail_closed_without_a_confirming_manual(tmp_path: Path) -> None:
    """A manual without these clauses gives no hours, deadlines or protocol."""
    store = tools(small_snapshot(), tmp_path)
    assert "does not confirm" in str(store.store_open_now()["error"])
    assert "does not confirm" in str(
        store.check_return_window("regret", "online", "2026-11-01", "2026-11-02")["error"]
    )
    assert store.sources == []


def test_policy_search_masks_internal_contacts(manual: Snapshot, tmp_path: Path) -> None:
    """Registration contacts in the manual never reach the model; public ones remain."""
    facts = PolicyEvidence(sections(manual))
    raw = " ".join(section.text for section in sections(manual))
    assert {match.group(0) for match in PHONE.finditer(raw)} - {facts.whatsapp}
    assert {match.group(0) for match in EMAIL.finditer(raw)} - {facts.email}
    assert manual.policies is not None
    for chunk in manual.policies.chunks:
        masked = facts.mask(chunk.text)
        assert {match.group(0) for match in PHONE.finditer(masked)} <= {facts.whatsapp}
        assert {match.group(0) for match in EMAIL.finditer(masked)} <= {facts.email}
    store = tools(manual, tmp_path)
    passages = str(store.search_policies("Dados da Empresa CNPJ Telefone E-mail"))
    assert MASK in passages
    assert "CNPJ" in passages
    assert {match.group(0) for match in PHONE.finditer(passages)} <= {facts.whatsapp}


def test_mask_hides_every_contact_when_the_manual_confirms_none() -> None:
    """Without a confirmed public channel, all phones and emails are withheld."""
    facts = PolicyEvidence(())
    text = "Ligue (11) 2345-6789 ou 11 98765-4321, escreva para a@b.com.br. CEP 79202-333."
    assert facts.mask(text) == f"Ligue {MASK} ou {MASK}, escreva para {MASK}. CEP 79202-333."


CAMPO_GRANDE = ZoneInfo("America/Campo_Grande")


@pytest.mark.parametrize(
    ("local", "is_open", "today", "next_opening"),
    [
        ("2026-11-02 08:59", False, "09:00-18:00", "Monday 2026-11-02 09:00"),
        ("2026-11-02 09:00", True, "09:00-18:00", "Tuesday 2026-11-03 09:00"),
        ("2026-11-06 17:59", True, "09:00-18:00", "Saturday 2026-11-07 09:00"),
        ("2026-11-06 18:00", False, "09:00-18:00", "Saturday 2026-11-07 09:00"),
        ("2026-11-07 12:59", True, "09:00-13:00", "Monday 2026-11-09 09:00"),
        ("2026-11-07 13:00", False, "09:00-13:00", "Monday 2026-11-09 09:00"),
        ("2026-11-08 10:00", False, "closed", "Monday 2026-11-09 09:00"),
    ],
)
def test_opening_status_follows_regular_hours_at_the_boundaries(
    local: str, is_open: bool, today: str, next_opening: str
) -> None:
    """Opening is inclusive and closing exclusive; Sundays are closed."""
    status = opening_status(datetime.fromisoformat(local).replace(tzinfo=CAMPO_GRANDE))
    assert status["open_by_regular_hours"] is is_open
    assert status["today"] == today
    assert status["next_opening"] == next_opening
    assert ("closes_at" in status) is is_open
    assert "Holidays" in str(status["warning"])


def test_open_now_tool_uses_store_time_and_never_claims_holiday_knowledge(
    manual: Snapshot, tmp_path: Path
) -> None:
    """A national holiday on a weekday still reports regular hours with the warning."""
    store = tools(manual, tmp_path)
    store.now = lambda: datetime(2026, 11, 20, 10, 0, tzinfo=CAMPO_GRANDE)
    status = store.store_open_now()
    assert status["open_by_regular_hours"] is True
    assert status["whatsapp"] == "(67) 3321-4500"
    assert "not known" in str(status["warning"])
    assert any(source["section"] == "2. Horário de Funcionamento" for source in store.sources)
    assert "error" in tools(small_snapshot(), tmp_path / "other").store_open_now()


def test_store_tools_default_clock_uses_the_configured_timezone(
    manual: Snapshot, tmp_path: Path
) -> None:
    """The real clock is timezone-aware in the configured store zone."""
    access = ConversationAccess(tmp_path / "s.sqlite3", conversation_token(), "generation")
    assert StoreTools(manual, access, "America/Campo_Grande").now().tzinfo == CAMPO_GRANDE


TODAY = date(2026, 11, 20)


@pytest.mark.parametrize(
    ("reason", "channel", "purchased", "received", "field", "deadline", "inside"),
    [
        ("regret", "online", "2026-11-01", "2026-11-13", None, "2026-11-20", True),
        ("regret", "online", "2026-11-01", "2026-11-12", None, "2026-11-19", False),
        ("preference", "online", "2026-11-01", "2026-11-13", None, "2026-11-20", True),
        ("preference", "store", "2026-11-13", None, None, "2026-11-20", True),
        ("defect", "online", "2026-10-21", "2026-10-25", "store_exchange", "2026-11-20", True),
        ("defect", "online", "2026-10-20", "2026-10-25", "store_exchange", "2026-11-19", False),
        (
            "defect",
            "store",
            "2026-08-22",
            None,
            "legal_warranty_intermediation",
            "2026-11-20",
            True,
        ),
        (
            "defect",
            "online",
            "2026-08-10",
            "2026-08-21",
            "legal_warranty_intermediation",
            "2026-11-19",
            False,
        ),
    ],
)
def test_return_windows_count_calendar_days_inclusively(
    reason: str,
    channel: str,
    purchased: str,
    received: str | None,
    field: str | None,
    deadline: str,
    inside: bool,
) -> None:
    """Regret and preference count from receipt, defects from purchase (30) and receipt (90)."""
    result = return_window(reason, channel, purchased, received, TODAY)
    window = cast(dict[str, object], result[field]) if field else result
    assert window["deadline"] == deadline
    assert window["within_window"] is inside


@pytest.mark.parametrize(
    ("reason", "channel", "purchased", "received", "message"),
    [
        ("regret", "online", "2026-11-01", None, "date the product was received"),
        ("preference", "online", "2026-11-01", None, "date the product was received"),
        ("defect", "online", None, "2026-11-01", "purchase date and the receipt date"),
        ("regret", "phone", "2026-11-01", "2026-11-02", "online or in the physical store"),
        ("refund", "online", "2026-11-01", "2026-11-02", "regret, preference or defect"),
        ("regret", "online", "2026-11-01", "2026-11-21", "cannot be in the future"),
        ("regret", "online", "2026-11-05", "2026-11-02", "before the purchase date"),
        ("regret", "online", "2026-11-01", "02/11/2026", "YYYY-MM-DD"),
    ],
)
def test_return_window_asks_for_missing_or_invalid_customer_dates(
    reason: str, channel: str, purchased: str | None, received: str | None, message: str
) -> None:
    """Missing receipt or channel is asked for, never assumed."""
    with pytest.raises(ValueError, match=message):
        return_window(reason, channel, purchased, received, TODAY)


def test_store_purchases_have_no_withdrawal_right() -> None:
    """In-store regret points to the preference exchange instead of a deadline."""
    result = return_window("regret", "store", "2026-11-18", None, TODAY)
    assert result["applicable"] is False
    assert "preference exchange" in str(result["explanation"])
    assert "deadline" not in result


def test_return_tool_cites_the_manual_and_never_approves(manual: Snapshot, tmp_path: Path) -> None:
    """The tool result is informational, sourced and points to WhatsApp."""
    store = tools(manual, tmp_path)
    store.now = lambda: datetime(2026, 11, 20, 10, 0, tzinfo=CAMPO_GRANDE)
    result = store.check_return_window("regret", "online", "2026-11-01", "2026-11-13")
    assert result["within_window"] is True
    assert "not an approval" in str(result["note"])
    assert "(67) 3321-4500" in str(result["note"])
    sections_cited = {str(source["section"]) for source in store.sources}
    assert "4.1 Direito de Arrependimento (Compras Online)" in sections_cited
    assert "4.4 Itens Não Elegíveis para Troca" in sections_cited
    missing = store.check_return_window("regret", "online", "2026-11-01")
    assert "received" in str(missing["error"])
    other = tools(small_snapshot(), tmp_path / "other")
    assert "error" in other.check_return_window("regret", "online", "2026-11-01", "2026-11-13")


@pytest.mark.parametrize(
    ("final_cents", "fee"), [(49999, 3500), (50000, 3500), (50001, 0), (129900, 0)]
)
def test_metro_shipping_is_free_only_above_five_hundred_reais(final_cents: int, fee: int) -> None:
    """Exactly R$ 500,00 still pays the R$ 35,00 fee."""
    assert metro_shipping_cents(final_cents) == fee


@pytest.mark.parametrize(
    ("product_id", "method", "total", "fee", "with_shipping"),
    [
        (89, "credit", "R$ 549,00", "R$ 0,00", "R$ 549,00"),
        (89, "pix", "R$ 521,55", "R$ 0,00", "R$ 521,55"),
        (121, "credit", "R$ 439,20", "R$ 35,00", "R$ 474,20"),
    ],
)
def test_quote_applies_metro_shipping_to_the_final_amount_paid(
    manual: Snapshot,
    tmp_path: Path,
    product_id: int,
    method: str,
    total: str,
    fee: str,
    with_shipping: str,
) -> None:
    """A R$ 549,00 list price promoted to R$ 439,20 pays shipping; PIX stays above R$ 500."""
    store = tools(manual, tmp_path)
    result = store.calculate_quote(product_id, payment_method=method)
    assert result["total"] == total
    assert result["metro_region_shipping"] == fee
    assert result["total_with_metro_shipping"] == with_shipping
    assert "(67) 3321-4500" in str(result["shipping_note"])
    assert any(
        source.get("section") == "5.1 Entregas na Região Metropolitana de Campo Grande"
        for source in store.sources
    )


def test_quote_withholds_shipping_without_a_confirming_manual(tmp_path: Path) -> None:
    """Fixture PDFs have no shipping clause, so no fee is invented."""
    result = tools(small_snapshot(), tmp_path).calculate_quote(101)
    assert "metro_region_shipping" not in result
    assert "not confirmed" in str(result["shipping_note"])
