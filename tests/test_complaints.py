"""Check verified-order complaint records, protocols and personal-data redaction."""

import re
import sqlite3
from contextlib import closing
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest
from agent_fixtures import manual_snapshot, small_snapshot

from emporio_agent.access import ConversationAccess, conversation_token
from emporio_agent.complaints import REDACTED, new_protocol, record_complaint
from emporio_agent.runtime import Snapshot
from emporio_agent.tools import StoreTools

NOW = datetime(2026, 11, 2, 23, 30, tzinfo=ZoneInfo("America/Campo_Grande"))
PROTOCOL = re.compile(r"REC-20261102-[A-HJ-NP-Z2-9]{6}")


@pytest.fixture(scope="module")
def snapshot() -> Snapshot:
    """Combine fixture orders with the delivered manual's complaint process."""
    manual = manual_snapshot()
    return replace(small_snapshot(), policies=manual.policies, rulings=manual.rulings)


def tools(snapshot: Snapshot, tmp_path: Path, token: str | None = None) -> StoreTools:
    """Bind tools to one conversation with a fixed store-local clock."""
    access = ConversationAccess(
        tmp_path / "sessions.sqlite3", token or conversation_token(), "generation"
    )
    store = StoreTools(snapshot, access)
    store.now = lambda: NOW
    return store


def rows(tmp_path: Path) -> list[tuple[object, ...]]:
    """Read stored complaints the way staff would."""
    with closing(sqlite3.connect(tmp_path / "sessions.sqlite3")) as connection:
        return connection.execute(
            "SELECT protocol, created_at, order_id, summary, status, email FROM complaints"
        ).fetchall()


def test_protocol_is_dated_and_avoids_confusable_characters() -> None:
    """Protocols use the store-local date and an unambiguous random suffix."""
    protocols = {new_protocol(NOW) for _ in range(200)}
    assert all(PROTOCOL.fullmatch(protocol) for protocol in protocols)
    assert len(protocols) > 190


def test_verified_order_complaint_is_recorded_with_a_protocol(
    snapshot: Snapshot, tmp_path: Path
) -> None:
    """The tool stores the complaint and returns the WhatsApp message with the protocol."""
    store = tools(snapshot, tmp_path)
    assert store.verify_order(501, "alex@example.test")["verified"] is True
    result = store.register_complaint("A caixa chegou amassada e o violão está riscado.")
    protocol = str(result["protocol"])
    assert PROTOCOL.fullmatch(protocol)
    assert result["order_id"] == 501
    assert result["whatsapp"] == "(67) 3321-4500"
    assert protocol in str(result["whatsapp_message"])
    assert "24 business hours" in str(result["response_time"])
    assert rows(tmp_path) == [
        (
            protocol,
            "2026-11-02T23:30:00-04:00",
            501,
            "A caixa chegou amassada e o violão está riscado.",
            "open",
            "alex@example.test",
        )
    ]
    assert any(source.get("section") == "7.3 Situações Especiais" for source in store.sources)


def test_complaints_require_a_verified_order(snapshot: Snapshot, tmp_path: Path) -> None:
    """Without verification, or after a failed one, nothing is stored."""
    store = tools(snapshot, tmp_path)
    assert "verify_order first" in str(
        store.register_complaint("Atendimento demorado demais.")["error"]
    )
    assert store.verify_order(501, "bea@example.test")["verified"] is False
    assert "error" in store.register_complaint("Atendimento demorado demais.")
    assert not _table_exists(tmp_path)


def _table_exists(tmp_path: Path) -> bool:
    with closing(sqlite3.connect(tmp_path / "sessions.sqlite3")) as connection:
        return bool(
            connection.execute(
                "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'complaints'"
            ).fetchone()
        )


def test_personal_data_is_redacted_before_storage(tmp_path: Path) -> None:
    """Emails, phones and CPFs typed by the customer never reach the complaint table."""
    database = tmp_path / "sessions.sqlite3"
    record_complaint(
        database,
        "conversation",
        501,
        "Me liguem no (67) 99876-5432 ou 67 3333-4444, email ana@exemplo.com, CPF 123.456.789-09.",
        NOW,
        limit=3,
    )
    (summary,) = rows(tmp_path)[0][3:4]
    assert summary == (
        f"Me liguem no {REDACTED} ou {REDACTED}, email ana@exemplo.com, CPF {REDACTED}."
    )


def test_summary_length_and_conversation_limit(tmp_path: Path) -> None:
    """Empty summaries are refused and one conversation cannot flood the table."""
    database = tmp_path / "sessions.sqlite3"
    with pytest.raises(ValueError, match="10 to 1000"):
        record_complaint(database, "conversation", 501, "   ruim  ", NOW, limit=3)
    for _ in range(3):
        record_complaint(database, "conversation", 501, "Produto com defeito.", NOW, limit=3)
    with pytest.raises(ValueError, match="maximum number"):
        record_complaint(database, "conversation", 501, "Produto com defeito.", NOW, limit=3)
    record_complaint(database, "another", 502, "Produto com defeito.", NOW, limit=3)
    assert len(rows(tmp_path)) == 4


def test_complaints_fail_closed_without_a_confirming_manual(tmp_path: Path) -> None:
    """A manual without the complaint process gives no protocol."""
    store = tools(small_snapshot(), tmp_path)
    assert store.verify_order(501, "alex@example.test")["verified"] is True
    assert "does not confirm" in str(store.register_complaint("Produto com defeito.")["error"])


def test_existing_complaint_table_gains_the_email_column(tmp_path: Path) -> None:
    """A table created before the email column is migrated without losing rows."""
    database = tmp_path / "sessions.sqlite3"
    with closing(sqlite3.connect(database)) as connection, connection:
        connection.execute(
            "CREATE TABLE complaints (protocol TEXT PRIMARY KEY, created_at TEXT NOT NULL, "
            "order_id INTEGER NOT NULL, conversation TEXT NOT NULL, summary TEXT NOT NULL, "
            "status TEXT NOT NULL DEFAULT 'open') STRICT"
        )
        connection.execute(
            "INSERT INTO complaints (protocol, created_at, order_id, conversation, summary) "
            "VALUES ('REC-OLD', 'x', 1, 'c', 'Antiga reclamação')"
        )
    record_complaint(database, "c2", 501, "Produto com defeito.", NOW, limit=3, email="a@b.com")
    assert [row[5] for row in rows(tmp_path)] == [None, "a@b.com"]
