"""Check owner-approved store facts against the delivered manual and its variations."""

import re
from dataclasses import replace
from pathlib import Path

import pytest
from agent_fixtures import manual_snapshot, small_snapshot

from emporio_agent.agent import ConversationAccess, StoreTools, conversation_token
from emporio_agent.documents import PolicySection
from emporio_agent.runtime import Snapshot
from emporio_agent.store_facts import FACTS, MASK, TOPICS, StoreFacts

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


def test_every_fact_is_confirmed_by_the_delivered_manual(manual: Snapshot) -> None:
    """Each quote exists verbatim in its numbered section, so no fact is invented."""
    facts = StoreFacts(sections(manual))
    assert facts.whatsapp == "(67) 3321-4500"
    assert facts.email == "contato@emporiodamusica.com.br"
    for fact in FACTS:
        assert facts.has(fact.key), fact.key
    assert {topic for topic in TOPICS if facts.topic(topic)} == set(TOPICS)


def test_facts_only_expose_the_public_whatsapp_and_cover_email(manual: Snapshot) -> None:
    """Company registration contacts never appear in any statement."""
    facts = StoreFacts(sections(manual))
    for topic in TOPICS:
        for fact in facts.topic(topic):
            text = f"{fact['statement']} {fact['clarification']}"
            assert {match.group(0) for match in PHONE.finditer(text)} <= {facts.whatsapp}
            assert {match.group(0) for match in EMAIL.finditer(text)} <= {facts.email}


def test_a_missing_quote_withholds_only_that_fact(manual: Snapshot) -> None:
    """A changed manual disables the facts it no longer supports and keeps the rest."""
    edited = tuple(
        replace(section, text=section.text.replace("acima de R$ 500,00", "acima de R$ 600,00"))
        for section in sections(manual)
    )
    facts = StoreFacts(edited)
    assert not facts.has("metro_shipping")
    assert facts.has("other_cities_shipping", "hours", "installments")


def test_a_changed_whatsapp_number_withholds_every_contact_dependent_fact(
    manual: Snapshot,
) -> None:
    """Two WhatsApp channels in section 7 are ambiguous, so no channel is offered."""
    edited = tuple(
        replace(section, text=section.text + " O canal de WhatsApp (67) 0000-0000 também.")
        if section.title.startswith("7. ")
        else section
        for section in sections(manual)
    )
    facts = StoreFacts(edited)
    assert facts.whatsapp is None
    assert not facts.has("whatsapp", "email", "hours", "defect")
    assert facts.has("address", "installments", "metro_shipping")


def test_tool_returns_sourced_facts_and_records_citations(manual: Snapshot, tmp_path: Path) -> None:
    """The agent tool reports manual pages for the UI source list."""
    store = tools(manual, tmp_path)
    result = store.store_facts("returns")
    assert "facts" in result
    assert "7 calendar days after receipt" in str(result)
    assert {"4.1 Direito de Arrependimento (Compras Online)", "4.3 Trocas por Preferência"} <= {
        str(source["section"]) for source in store.sources
    }


def test_tool_fails_closed_without_a_confirming_manual(tmp_path: Path) -> None:
    """A different PDF confirms nothing, so the agent must refer the customer to staff."""
    store = tools(small_snapshot(), tmp_path)
    assert "does not confirm" in str(store.store_facts("contact")["error"])
    assert "Unknown topic" in str(store.store_facts("taxes")["error"])
    assert store.sources == []


def test_policy_search_masks_internal_contacts(manual: Snapshot, tmp_path: Path) -> None:
    """Registration contacts in the manual never reach the model; public ones remain."""
    facts = StoreFacts(sections(manual))
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
    facts = StoreFacts(())
    text = "Ligue (11) 2345-6789 ou 11 98765-4321, escreva para a@b.com.br. CEP 79202-333."
    assert facts.mask(text) == f"Ligue {MASK} ou {MASK}, escreva para {MASK}. CEP 79202-333."
