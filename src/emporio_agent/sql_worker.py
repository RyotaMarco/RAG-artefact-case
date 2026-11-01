"""Execute already-validated SELECT statements in a credential-free scoped process."""

import json
import sqlite3
import sys
import time
from typing import cast

from emporio_agent.data import Row, register_decimal_collation, sql_type
from emporio_agent.money import check_cents

_ALLOWED_FUNCTIONS = {
    "count",
    "sum",
    "min",
    "max",
    "lower",
    "upper",
    "length",
    "coalesce",
    "like",
    "exact_sum",
    "exact_avg",
}


class ExactSum:
    """Aggregate cents with Python integers and reject unrepresentable SQLite results."""

    def __init__(self) -> None:
        """Start an empty exact accumulator."""
        self.total = 0
        self.count = 0

    def step(self, value: int | None) -> None:
        """Accumulate nonnull integer cents without binary floating point."""
        if value is not None:
            self.total += check_cents(value)
            self.count += 1

    def finalize(self) -> int | None:
        """Return an exact signed-64-bit total or SQL NULL for no inputs."""
        return check_cents(self.total) if self.count else None


class ExactAverage(ExactSum):
    """Average cents exactly and round once to cents using HALF_UP."""

    def finalize(self) -> int | None:
        """Use an integer quotient and remainder, avoiding a repeating decimal intermediate."""
        if not self.count:
            return None
        quotient, remainder = divmod(self.total, self.count)
        return check_cents(quotient + (remainder * 2 >= self.count))


def _load_tables(connection: sqlite3.Connection, tables: dict[str, dict[str, object]]) -> None:
    """Materialize only the trusted public and verified-order scope as strict SQLite tables."""
    register_decimal_collation(connection)
    for name, table in tables.items():
        columns = cast(list[str], table["columns"])
        definitions = ", ".join(f'"{column}" {sql_type(column)}' for column in columns)
        connection.execute(f'CREATE TABLE "{name}" ({definitions}) STRICT')
        rows = cast(list[Row], table["rows"])
        connection.executemany(
            f'INSERT INTO "{name}" VALUES ({", ".join("?" for _ in columns)})',
            [tuple(row[column] for column in columns) for row in rows],
        )
    connection.commit()


def _restrict_connection(
    connection: sqlite3.Connection, tables: dict[str, dict[str, object]]
) -> None:
    """Apply the read-only authorizer, execution deadline and SQLite resource limits."""
    # SQLite accepts NULL here; its bundled typing stub incorrectly requires an int.
    connection.create_aggregate("exact_sum", 1, ExactSum)  # pyright: ignore[reportArgumentType]
    connection.create_aggregate("exact_avg", 1, ExactAverage)  # pyright: ignore[reportArgumentType]
    connection.execute("PRAGMA query_only=ON")
    connection.execute("PRAGMA trusted_schema=OFF")
    connection.setlimit(sqlite3.SQLITE_LIMIT_LENGTH, 65_536)
    connection.setlimit(sqlite3.SQLITE_LIMIT_SQL_LENGTH, 20_000)
    connection.setlimit(sqlite3.SQLITE_LIMIT_COLUMN, 64)
    connection.setlimit(sqlite3.SQLITE_LIMIT_COMPOUND_SELECT, 1)
    connection.setlimit(sqlite3.SQLITE_LIMIT_EXPR_DEPTH, 40)
    connection.setlimit(sqlite3.SQLITE_LIMIT_ATTACHED, 0)
    deadline = time.monotonic() + 2
    connection.set_progress_handler(lambda: int(time.monotonic() > deadline), 500)

    def authorize(
        action: int,
        first: str | None,
        second: str | None,
        database: str | None,
        _source: str | None,
    ) -> int:
        if action == sqlite3.SQLITE_SELECT:
            return sqlite3.SQLITE_OK
        if action == sqlite3.SQLITE_READ and database in (None, "main") and first in tables:
            columns = cast(list[str], tables[first]["columns"])
            if second == "" or second in columns:
                return sqlite3.SQLITE_OK
        if action == sqlite3.SQLITE_FUNCTION and second in _ALLOWED_FUNCTIONS:
            return sqlite3.SQLITE_OK
        return sqlite3.SQLITE_DENY

    connection.set_authorizer(authorize)


def _read_result(cursor: sqlite3.Cursor) -> dict[str, object]:
    """Validate unique result names and exact values within the row and byte budgets."""
    if cursor.description is None:
        raise ValueError("Only result-producing queries are allowed.")
    columns = [str(column[0]) for column in cursor.description]
    if len(set(columns)) != len(columns):
        raise ValueError("Use unique aliases for result columns.")
    rows = cast(list[tuple[object, ...]], cursor.fetchmany(51))
    if any(value is not None and type(value) not in (int, str) for row in rows for value in row):
        raise ValueError("The query produced a non-exact or unsupported result type.")
    result: dict[str, object] = {
        "columns": columns,
        "rows": rows[:50],
        "truncated": len(rows) > 50,
    }
    if len(json.dumps(result, ensure_ascii=False).encode()) > 32 * 1024:
        raise ValueError("The result is too large; request fewer rows or columns.")
    return result


def _run(payload: dict[str, object]) -> dict[str, object]:
    """Own one scoped SQLite connection from table loading through bounded result reading."""
    tables = cast(dict[str, dict[str, object]], payload["tables"])
    connection = sqlite3.connect(":memory:", cached_statements=0)
    try:
        _load_tables(connection, tables)
        _restrict_connection(connection, tables)
        cursor = connection.execute(cast(str, payload["sql"]))
        return _read_result(cursor)
    finally:
        connection.close()


def main() -> None:
    """Read one trusted scope, execute one query, and write one bounded JSON result."""
    try:
        decoded: object = json.loads(sys.stdin.read())
        if not isinstance(decoded, dict):
            raise ValueError("Invalid worker input.")
        result = _run(cast(dict[str, object], decoded))
    except (ValueError, sqlite3.Error, OverflowError, MemoryError, KeyError) as exc:
        result = {"error": f"Query rejected: {exc}"}
    sys.stdout.write(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
