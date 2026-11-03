"""Conversation identifiers and server-owned order authorization."""

import hashlib
import re
import secrets
import sqlite3
from contextlib import closing
from pathlib import Path

from emporio_agent.sql import QueryEngine


def conversation_token(candidate: str | None = None) -> str:
    """Reuse only well-formed bearer tokens; otherwise create an unpredictable conversation."""
    if candidate and re.fullmatch(r"[A-Za-z0-9_-]{43}", candidate):
        return candidate
    return secrets.token_urlsafe(32)


class ConversationAccess:
    """Persist server-owned order authorization separately from model-authored text."""

    def __init__(self, database: Path, token: str, generation: str) -> None:
        """Initialize private state for an unguessable conversation capability."""
        database.parent.mkdir(parents=True, exist_ok=True)
        self.database = database
        self.identity = hashlib.sha256(token.encode()).hexdigest()
        self.generation = generation
        with closing(sqlite3.connect(database, timeout=10)) as connection, connection:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                "CREATE TABLE IF NOT EXISTS store_access ("
                "identity TEXT PRIMARY KEY, order_id INTEGER, "
                "generation TEXT NOT NULL DEFAULT '') STRICT"
            )
            columns = {row[1] for row in connection.execute("PRAGMA table_info(store_access)")}
            if "failures" in columns:
                connection.execute("ALTER TABLE store_access DROP COLUMN failures")
            if "generation" not in columns:
                connection.execute(
                    "ALTER TABLE store_access ADD COLUMN generation TEXT NOT NULL DEFAULT ''"
                )
            connection.execute(
                "INSERT OR IGNORE INTO store_access VALUES (?, NULL, ?)",
                (self.identity, generation),
            )
            connection.execute(
                "UPDATE store_access SET order_id = NULL, generation = ? "
                "WHERE identity = ? AND generation != ?",
                (generation, self.identity, generation),
            )

    def verified_order(self) -> int | None:
        """Read an order scope that can only have been established by trusted code."""
        with closing(sqlite3.connect(self.database, timeout=10)) as connection, connection:
            row = connection.execute(
                "SELECT order_id FROM store_access WHERE identity = ? AND generation = ?",
                (self.identity, self.generation),
            ).fetchone()
        return int(row[0]) if row is not None and row[0] is not None else None

    def verify(self, engine: QueryEngine, order_id: int, email: str) -> bool:
        """Check the order/contact pair and never preserve an old scope after failure."""
        with closing(sqlite3.connect(self.database, timeout=10)) as connection, connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT generation FROM store_access WHERE identity = ?", (self.identity,)
            ).fetchone()
            if row[0] != self.generation:
                return False
            wanted = engine.verify_order(order_id, email)
            connection.execute(
                "UPDATE store_access SET order_id = ? WHERE identity = ?",
                (wanted, self.identity),
            )
        return wanted is not None
