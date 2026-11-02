"""Record customer complaints for staff follow-up, scoped to a verified order."""

import re
import secrets
import sqlite3
from contextlib import closing
from datetime import datetime
from pathlib import Path

PROTOCOL_ALPHABET = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"
MAX_PER_CONVERSATION = 3
# Owner ruling: emails may be kept; phones and CPFs typed by the customer are removed.
_PERSONAL = re.compile(r"\(?\b\d{2}\)?\s*9?\d{4}-?\d{4}\b|\b\d{3}\.?\d{3}\.?\d{3}-?\d{2}\b")
REDACTED = "[dado pessoal omitido]"


def new_protocol(now: datetime) -> str:
    """Build a dated protocol whose suffix is unguessable and easy to read aloud."""
    suffix = "".join(secrets.choice(PROTOCOL_ALPHABET) for _ in range(6))
    return f"REC-{now:%Y%m%d}-{suffix}"


def record_complaint(
    database: Path,
    conversation: str,
    order_id: int,
    summary: str,
    now: datetime,
    *,
    email: str | None = None,
) -> str:
    """Store a redacted summary with the verified order's email and return its protocol.

    Raises:
        ValueError: The summary is empty or too long, or the conversation reached its limit.
    """
    text = _PERSONAL.sub(REDACTED, " ".join(summary.split()))
    if not 10 <= len(text) <= 1000:
        raise ValueError("Summarize the complaint in 10 to 1000 characters.")
    with closing(sqlite3.connect(database, timeout=10)) as connection, connection:
        connection.execute("BEGIN IMMEDIATE")
        connection.execute(
            "CREATE TABLE IF NOT EXISTS complaints ("
            "protocol TEXT PRIMARY KEY, created_at TEXT NOT NULL, order_id INTEGER NOT NULL, "
            "conversation TEXT NOT NULL, summary TEXT NOT NULL, "
            "status TEXT NOT NULL DEFAULT 'open', email TEXT) STRICT"
        )
        columns = {row[1] for row in connection.execute("PRAGMA table_info(complaints)")}
        if "email" not in columns:
            connection.execute("ALTER TABLE complaints ADD COLUMN email TEXT")
        (count,) = connection.execute(
            "SELECT COUNT(*) FROM complaints WHERE conversation = ?", (conversation,)
        ).fetchone()
        if count >= MAX_PER_CONVERSATION:
            raise ValueError(
                "This conversation already registered the maximum number of complaints."
            )
        for _ in range(5):
            protocol = new_protocol(now)
            try:
                connection.execute(
                    "INSERT INTO complaints (protocol, created_at, order_id, conversation, "
                    "summary, email) VALUES (?, ?, ?, ?, ?, ?)",
                    (
                        protocol,
                        now.isoformat(timespec="seconds"),
                        order_id,
                        conversation,
                        text,
                        email,
                    ),
                )
            except sqlite3.IntegrityError:
                continue
            return protocol
    raise ValueError("A complaint protocol could not be allocated. Please retry.")
