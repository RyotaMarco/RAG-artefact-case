"""Discover source snapshots and validate the six supplied business schemas."""

import csv
import hashlib
import io
import json
import re
import sqlite3
import unicodedata
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import cast

from emporio_agent.money import MAX_CENTS, canonical_rate, parse_money_cents

type Value = int | str | None
type Row = dict[str, Value]


class DataError(ValueError):
    """A source snapshot could not be validated without changing its meaning."""


@dataclass(frozen=True)
class SourceFile:
    """An immutable supported input file and its portable source identity."""

    relative_path: str
    content: bytes
    digest: str


@dataclass(frozen=True)
class Table:
    """A logical table assembled from one or more validated CSV shards."""

    columns: tuple[str, ...]
    rows: tuple[Row, ...]
    source_paths: tuple[str, ...]


@dataclass(frozen=True)
class Catalog:
    """A complete validated operational snapshot, separate from conversation storage."""

    fingerprint: str
    sources: tuple[SourceFile, ...]
    tables: dict[str, Table]

    def describe(self) -> dict[str, object]:
        """Describe safe SQL schemas and observed values without customer records."""
        result: dict[str, object] = {}
        for name, table in self.tables.items():
            if name == "customers":
                continue
            enums = {
                column: sorted({str(row[column]) for row in table.rows if row[column] is not None})
                for column in ("status", "payment_method", "is_active")
                if column in table.columns
            }
            result[name] = {
                "columns": {
                    column: "INTEGER cents (BRL)" if column.endswith("_cents") else sql_type(column)
                    for column in table.columns
                    if name != "orders" or column not in {"customer_id", "notes"}
                },
                "row_count": len(table.rows),
                "sources": table.source_paths,
                "observed_values": enums,
                "requires_verified_order": name in {"orders", "order_items"},
            }
        return result


_HEADERS: dict[str, tuple[str, ...]] = {
    "categories": ("category_id", "name", "description"),
    "customers": ("customer_id", "name", "phone", "email", "city"),
    "products": (
        "product_id",
        "price_brl",
        "name",
        "category_id",
        "description",
        "stock_quantity",
        "status",
        "specs",
        "created_at",
    ),
    "promotions": ("promotion_id", "product_id", "discount_percent", "description", "is_active"),
    "orders": (
        "order_id",
        "customer_id",
        "order_date",
        "status",
        "total_brl",
        "payment_method",
        "tracking_code",
        "estimated_delivery",
        "notes",
    ),
    "order_items": ("order_id", "quantity", "product_id"),
}
_KEYS: dict[str, tuple[str, ...]] = {
    "categories": ("category_id",),
    "customers": ("customer_id",),
    "products": ("product_id",),
    "promotions": ("promotion_id",),
    "orders": ("order_id",),
    "order_items": ("order_id", "product_id"),
}
_OPTIONAL = {"tracking_code", "estimated_delivery", "notes"}
_DATES = {"created_at", "order_date", "estimated_delivery"}
_MONEY = {"price_brl": "price_cents", "total_brl": "total_cents"}


def normalize_name(value: str) -> str:
    """Normalize case, accents, spaces, and punctuation for deterministic name matching."""
    ascii_text = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode("ascii")
    return re.sub(r"[^a-z0-9]", "", ascii_text.casefold())


def inventory_fingerprint(sources: Sequence[SourceFile]) -> str:
    """Hash content and relative paths so changes, additions, and removals invalidate snapshots."""
    inventory = [(source.relative_path, source.digest) for source in sources]
    encoded = json.dumps(sorted(inventory), ensure_ascii=False, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def discover_sources(directory: Path) -> tuple[SourceFile, ...]:
    """Discover all CSV/PDF files recursively without allowing paths outside the data directory."""
    root = directory.resolve()
    if not root.is_dir():
        raise DataError("The data directory does not exist.")
    sources: list[SourceFile] = []
    for path in sorted(root.rglob("*")):
        if path.suffix.casefold() not in {".csv", ".pdf"} or not path.is_file():
            continue
        if not path.resolve().is_relative_to(root):
            raise DataError("A source resolves outside the data directory.")
        content = path.read_bytes()
        sources.append(
            SourceFile(
                path.relative_to(root).as_posix(), content, hashlib.sha256(content).hexdigest()
            )
        )
    if not sources:
        raise DataError("The data directory contains no supported CSV or PDF files.")
    return tuple(sources)


def sql_type(column: str) -> str:
    """Return the strict SQLite storage type used by normalized business columns."""
    return (
        "INTEGER"
        if column.endswith(("_id", "_cents"))
        or column
        in {
            "stock_quantity",
            "quantity",
            "is_active",
        }
        else "TEXT"
    )


def _integer(value: str, *, zero_allowed: bool = False) -> int:
    if not re.fullmatch(r"\d+", value):
        raise DataError("An integer field contains an invalid value.")
    result = int(value)
    if result > MAX_CENTS or result < (0 if zero_allowed else 1):
        raise DataError("An integer field is outside its supported range.")
    return result


def _normalize(column: str, value: str) -> tuple[str, Value]:
    value = value.strip()
    if not value:
        if column in _OPTIONAL:
            return column, None
        raise DataError(f"Required column {column!r} contains an empty value.")
    if column in _MONEY:
        return _MONEY[column], parse_money_cents(value)
    if column.endswith("_id") or column == "quantity":
        return column, _integer(value)
    if column == "stock_quantity":
        return column, _integer(value, zero_allowed=True)
    if column == "is_active":
        if value.casefold() not in {"0", "1", "false", "true"}:
            raise DataError("is_active must contain a boolean flag.")
        return column, int(value.casefold() in {"1", "true"})
    if column == "discount_percent":
        return column, canonical_rate(value)
    if column in _DATES:
        if date.fromisoformat(value).isoformat() != value:
            raise DataError("Dates must use YYYY-MM-DD format.")
    if column == "specs":
        parsed: object = json.loads(value)
        if not isinstance(parsed, dict):
            raise DataError("Product specs must be a JSON object.")
    return column, value


def _read_csv(source: SourceFile) -> tuple[str, list[Row]]:
    try:
        text = source.content.decode("utf-8-sig")
        reader = csv.reader(io.StringIO(text, newline=""))
        headers = tuple(value.strip() for value in next(reader))
        if len(set(headers)) != len(headers):
            raise DataError("CSV headers must be unique.")
        known_headers = {column for required in _HEADERS.values() for column in required}
        matches = [
            name
            for name, required in _HEADERS.items()
            if set(required) <= set(headers)
            and not ((set(headers) & known_headers) - set(required))
        ]
        if matches:
            specificity = max(len(_HEADERS[name]) for name in matches)
            matches = [name for name in matches if len(_HEADERS[name]) == specificity]
        if len(matches) != 1:
            raise DataError("CSV headers do not identify exactly one supported business schema.")
        name = matches[0]
        rows: list[Row] = []
        for number, values in enumerate(reader, start=2):
            if not values or all(not value.strip() for value in values):
                continue
            if len(values) != len(headers):
                raise DataError(f"CSV row {number} has the wrong number of fields.")
            raw = dict(zip(headers, values, strict=True))
            normalized = dict(_normalize(column, raw[column]) for column in _HEADERS[name])
            if name == "products":
                normalized["search_name"] = normalize_name(cast(str, normalized["name"]))
            rows.append(normalized)
        return name, rows
    except (UnicodeError, StopIteration, csv.Error, ValueError) as exc:
        raise DataError(f"Invalid source {source.relative_path!r}: {exc}") from exc


def build_catalog(sources: Sequence[SourceFile]) -> Catalog:
    """Build an idempotent full snapshot, deduplicating identical shard records only."""
    grouped: dict[str, dict[tuple[Value, ...], Row]] = {}
    provenance: dict[str, list[str]] = {}
    for source in sorted(sources, key=lambda item: item.relative_path):
        if Path(source.relative_path).suffix.casefold() != ".csv":
            continue
        name, rows = _read_csv(source)
        target = grouped.setdefault(name, {})
        provenance.setdefault(name, []).append(source.relative_path)
        for row in rows:
            key = tuple(row[column] for column in _KEYS[name])
            if key in target and target[key] != row:
                raise DataError(f"Conflicting primary key in {name}: {key!r}.")
            target[key] = row
    tables: dict[str, Table] = {}
    for name, records in grouped.items():
        columns = tuple(_MONEY.get(column, column) for column in _HEADERS[name])
        if name == "products":
            columns += ("search_name",)
        tables[name] = Table(columns, tuple(records.values()), tuple(provenance[name]))
    relationships = (
        ("products", "category_id", "categories", "category_id"),
        ("orders", "customer_id", "customers", "customer_id"),
        ("order_items", "order_id", "orders", "order_id"),
        ("order_items", "product_id", "products", "product_id"),
        ("promotions", "product_id", "products", "product_id"),
    )
    for child, column, parent, parent_key in relationships:
        if child not in tables or parent not in tables:
            continue
        known = {row[parent_key] for row in tables[parent].rows}
        if any(row[column] not in known for row in tables[child].rows):
            raise DataError(f"An orphan reference exists in {child}.{column}.")
    return Catalog(inventory_fingerprint(sources), tuple(sources), tables)


def write_catalog(catalog: Catalog, path: Path) -> None:
    """Write a new strict SQLite snapshot; the caller publishes its temporary path atomically."""
    if path.exists() and path.stat().st_size:
        raise DataError("A catalog snapshot cannot overwrite a nonempty database.")
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    try:
        connection.execute("BEGIN IMMEDIATE")
        for name, table in catalog.tables.items():
            definitions = [
                f'"{column}" {sql_type(column)}' + ("" if column in _OPTIONAL else " NOT NULL")
                for column in table.columns
            ]
            definitions.append("PRIMARY KEY (" + ", ".join(_KEYS[name]) + ")")
            connection.execute(f'CREATE TABLE "{name}" ({", ".join(definitions)}) STRICT')
            connection.executemany(
                f'INSERT INTO "{name}" VALUES ({", ".join("?" for _ in table.columns)})',
                [tuple(row[column] for column in table.columns) for row in table.rows],
            )
        connection.execute(
            "CREATE TABLE snapshot_metadata (key TEXT PRIMARY KEY, value TEXT) STRICT"
        )
        connection.execute(
            "INSERT INTO snapshot_metadata VALUES (?, ?)", ("fingerprint", catalog.fingerprint)
        )
        connection.commit()
    except sqlite3.Error as exc:
        connection.rollback()
        raise DataError(f"The catalog snapshot could not be written: {exc}") from exc
    finally:
        connection.close()
