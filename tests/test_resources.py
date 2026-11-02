"""Verify deterministic SQLite connection closure and preserved transaction behavior."""

import sqlite3
from contextlib import closing
from dataclasses import dataclass, field
from pathlib import Path
from typing import cast

import pytest
from agent_fixtures import small_snapshot
from test_embeddings import VoyageStub, make_embedder

from emporio_agent.agent import ConversationAccess, conversation_token
from emporio_agent.sql import QueryEngine


class TrackedConnection(sqlite3.Connection):
    """Retain real database connections so garbage collection cannot hide a leak."""

    was_closed = False

    def close(self) -> None:
        """Close the actual SQLite handle before recording ownership completion."""
        super().close()
        self.was_closed = True


@dataclass
class ResourceTracker:
    """Record handles and optionally reject one real SQLite action."""

    connections: list[TrackedConnection] = field(default_factory=list[TrackedConnection])
    denied_action: int | None = None
    denied_table: str = ""


@pytest.fixture
def tracker(monkeypatch: pytest.MonkeyPatch) -> ResourceTracker:
    """Wrap SQLite's factory without replacing database execution or transaction semantics."""
    result = ResourceTracker()
    real_connect = sqlite3.connect

    def connect(database: str | Path, *, timeout: float = 5.0) -> TrackedConnection:
        connection = real_connect(database, timeout=timeout, factory=TrackedConnection)
        result.connections.append(connection)

        def authorize(
            action: int,
            first: str | None,
            _second: str | None,
            _database: str | None,
            _source: str | None,
        ) -> int:
            if action == result.denied_action and first == result.denied_table:
                return sqlite3.SQLITE_DENY
            return sqlite3.SQLITE_OK

        connection.set_authorizer(authorize)
        return connection

    monkeypatch.setattr(sqlite3, "connect", connect)
    return result


def _assert_closed(tracker: ResourceTracker) -> None:
    assert tracker.connections
    for connection in tracker.connections:
        assert connection.was_closed
        with pytest.raises(sqlite3.ProgrammingError, match="closed database"):
            connection.execute("SELECT 1")


def _scalar(path: Path, sql: str) -> object:
    with closing(sqlite3.connect(path)) as connection:
        row = cast(tuple[object, ...] | None, connection.execute(sql).fetchone())
        assert row is not None
        return row[0]


def test_conversation_connections_close_and_verification_commits(
    tmp_path: Path, tracker: ResourceTracker
) -> None:
    """Constructor, write, and read contexts release handles while retaining successful state."""
    path = tmp_path / "sessions.sqlite3"
    token = conversation_token()
    access = ConversationAccess(path, token, "generation")
    _assert_closed(tracker)
    assert access.verify(QueryEngine(small_snapshot().catalog), 501, "alex@example.test")
    _assert_closed(tracker)
    assert access.verified_order() == 501
    _assert_closed(tracker)
    assert ConversationAccess(path, token, "generation").verified_order() == 501
    _assert_closed(tracker)


def test_rejected_contact_and_generation_early_return_close_handles(
    tmp_path: Path, tracker: ResourceTracker
) -> None:
    """Normal verification failures revoke access and early returns still close."""
    path = tmp_path / "sessions.sqlite3"
    token = conversation_token()
    access = ConversationAccess(path, token, "first")
    engine = QueryEngine(small_snapshot().catalog)
    assert not access.verify(engine, 501, "wrong@example.test")
    assert _scalar(path, "SELECT order_id FROM store_access") is None
    replacement = ConversationAccess(path, token, "replacement")
    assert not access.verify(engine, 501, "alex@example.test")
    assert replacement.verified_order() is None
    _assert_closed(tracker)


def test_conversation_constructor_failure_rolls_back_and_closes(
    tmp_path: Path, tracker: ResourceTracker
) -> None:
    """A rejected update rolls back the earlier INSERT and releases the failed connection."""
    path = tmp_path / "sessions.sqlite3"
    tracker.denied_action = sqlite3.SQLITE_UPDATE
    tracker.denied_table = "store_access"
    with pytest.raises(sqlite3.DatabaseError, match="authorized"):
        ConversationAccess(path, conversation_token(), "generation")
    _assert_closed(tracker)
    tracker.denied_action = None
    assert _scalar(path, "SELECT COUNT(*) FROM sqlite_master WHERE name = 'store_access'") == 0
    _assert_closed(tracker)


def test_verification_exception_rolls_back_and_closes(
    tmp_path: Path, tracker: ResourceTracker, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Exceptions after BEGIN cannot leave a write transaction or handle open."""
    path = tmp_path / "sessions.sqlite3"
    access = ConversationAccess(path, conversation_token(), "generation")
    engine = QueryEngine(small_snapshot().catalog)
    assert access.verify(engine, 501, "alex@example.test")

    def fail_verification(_order_id: int | str, _email: str) -> int | None:
        raise ValueError("Fixture identification failure.")

    monkeypatch.setattr(engine, "verify_order", fail_verification)
    with pytest.raises(ValueError, match="identification failure"):
        access.verify(engine, 501, "alex@example.test")
    _assert_closed(tracker)
    assert _scalar(path, "SELECT order_id FROM store_access") == 501
    _assert_closed(tracker)


def test_embedding_constructor_hits_and_misses_close_connections(
    tmp_path: Path, tracker: ResourceTracker
) -> None:
    """Cache writes survive closure and a cache hit requires no additional provider request."""
    path = tmp_path / "vectors.sqlite3"
    stub = VoyageStub()
    with stub.client:
        embedder = make_embedder(path, stub)
        _assert_closed(tracker)
        assert embedder.embed_documents(["policy"]) == [(0.6, 0.8)]
        _assert_closed(tracker)
        stub.failure = "timeout"
        restored = make_embedder(path, stub)
        assert restored.embed_documents(["policy"]) == [(0.6, 0.8)]
        assert len(stub.requests) == 1
        assert _scalar(path, "SELECT COUNT(*) FROM vectors") == 1
        _assert_closed(tracker)


def test_embedding_provider_failure_closes_the_read_connection(
    tmp_path: Path, tracker: ResourceTracker
) -> None:
    """An exception during a cache miss releases the active connection immediately."""
    stub = VoyageStub()
    with stub.client:
        embedder = make_embedder(tmp_path / "vectors.sqlite3", stub)
        stub.failure = "timeout"
        with pytest.raises(ValueError, match="timed out"):
            embedder.embed_documents(["policy"])
    _assert_closed(tracker)


def test_invalid_embedding_batch_leaves_no_rows_and_closes(
    tmp_path: Path, tracker: ResourceTracker
) -> None:
    """An invalid later vector cannot publish a valid first row or keep the database locked."""
    path = tmp_path / "vectors.sqlite3"
    stub = VoyageStub()
    with stub.client:
        stub.invalid_second = True
        embedder = make_embedder(path, stub)
        with pytest.raises(ValueError, match="zero or invalid vector"):
            embedder.embed_documents(["first policy", "second policy"])
        _assert_closed(tracker)
        assert _scalar(path, "SELECT COUNT(*) FROM vectors") == 0
        stub.invalid_second = False
        assert embedder.embed_documents(["first policy", "second policy"]) == [(0.6, 0.8)] * 2
        assert _scalar(path, "SELECT COUNT(*) FROM vectors") == 2
        _assert_closed(tracker)


def test_embedding_constructor_failure_closes_connection(
    tmp_path: Path, tracker: ResourceTracker
) -> None:
    """A schema-creation failure does not leak the connection opened by the constructor."""
    tracker.denied_action = sqlite3.SQLITE_CREATE_TABLE
    tracker.denied_table = "vectors"
    stub = VoyageStub()
    with stub.client:
        with pytest.raises(sqlite3.DatabaseError, match="authorized"):
            make_embedder(tmp_path / "vectors.sqlite3", stub)
    _assert_closed(tracker)
