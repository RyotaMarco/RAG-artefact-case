"""Validate changing source inventories, shard identity, and semantic data boundaries."""

import hashlib
import sqlite3
from pathlib import Path

import pytest

from emporio_agent.data import (
    DataError,
    SourceFile,
    build_catalog,
    discover_sources,
    register_decimal_collation,
    write_catalog,
)

_PRODUCT_ROW = {
    "product_id": "1",
    "price_brl": "10.00",
    "name": "Demo Guitar",
    "category_id": "1",
    "description": "An instrument",
    "stock_quantity": "2",
    "status": "active",
    "specs": "{}",
    "created_at": "2026-01-01",
}


def _source(name: str, text: str) -> SourceFile:
    content = text.encode("utf-8")
    return SourceFile(name, content, hashlib.sha256(content).hexdigest())


@pytest.mark.parametrize("count", [40, 90])
def test_arbitrary_shard_counts_and_header_order(count: int) -> None:
    """Classify renamed files by headers without fixed filename or row-count assumptions."""
    sources = tuple(
        _source(
            f"export-{index}.CSV",
            f"description,name,category_id\nDescription {index},Category,{index + 1}\n",
        )
        for index in range(count)
    )
    first = build_catalog(sources)
    second = build_catalog(tuple(reversed(sources)))
    assert len(first.tables["categories"].rows) == count
    assert first.fingerprint == second.fingerprint
    assert first.tables == second.tables


@pytest.mark.parametrize(
    ("role", "row", "extra"),
    [
        ("products", _PRODUCT_ROW, "phone"),
        ("products", _PRODUCT_ROW, "custom_field"),
        (
            "customers",
            {
                "customer_id": "7",
                "name": "Demo Buyer",
                "phone": "000",
                "email": "buyer@example.test",
                "city": "Example City",
            },
            "description",
        ),
        ("categories", {"category_id": "1", "name": "Guitars", "description": "Group"}, "phone"),
    ],
)
def test_complete_schemas_ignore_unselected_extra_columns(
    role: str, row: dict[str, str], extra: str
) -> None:
    """Recognize complete roles while keeping both known and unknown extras unexposed."""
    export = {extra: "Ignored metadata", **row}
    headers = tuple(reversed(export))
    text = ",".join(headers) + "\n" + ",".join(export[column] for column in headers) + "\n"
    catalog = build_catalog((_source("renamed-export.csv", text),))
    assert set(catalog.tables) == {role}
    table = catalog.tables[role]
    assert extra not in table.columns
    assert extra not in table.rows[0]
    assert table.rows[0]["name"] == row["name"]
    assert len(table.columns) == len(row) + (1 if role == "products" else 0)


@pytest.mark.parametrize("missing", tuple(_PRODUCT_ROW))
def test_each_missing_product_column_prevents_category_fallback(missing: str) -> None:
    """Reject incomplete products even when their surviving columns contain a full category."""
    export = {column: value for column, value in _PRODUCT_ROW.items() if column != missing}
    text = ",".join(export) + "\n" + ",".join(export.values()) + "\n"
    with pytest.raises(DataError, match="supported business schema"):
        build_catalog((_source("incomplete-products.csv", text),))


@pytest.mark.parametrize("reverse", [False, True])
def test_independent_complete_schemas_are_ambiguous_regardless_of_size(reverse: bool) -> None:
    """A wider product export must not win over a separate complete customer schema."""
    export = {
        **_PRODUCT_ROW,
        "customer_id": "7",
        "phone": "000",
        "email": "buyer@example.test",
        "city": "Example City",
    }
    headers = tuple(reversed(export)) if reverse else tuple(export)
    text = ",".join(headers) + "\n" + ",".join(export[column] for column in headers) + "\n"
    with pytest.raises(DataError, match="supported business schema"):
        build_catalog((_source("mixed-business-roles.csv", text),))


def test_ignored_shard_metadata_does_not_change_record_identity_or_hide_conflicts() -> None:
    """Discard extras consistently while retaining business-key deduplication and conflicts."""
    headers = ",".join((*_PRODUCT_ROW, "phone")) + "\n"
    values = ",".join(_PRODUCT_ROW.values())
    first = _source("first.csv", headers + values + ",111\n")
    second = _source("second.csv", headers + values + ",222\n")
    catalog = build_catalog((first, second))
    assert len(catalog.tables["products"].rows) == 1
    assert "phone" not in catalog.tables["products"].columns
    changed = _source("changed.csv", headers + values.replace("10.00", "11.00") + ",222\n")
    with pytest.raises(DataError, match="Conflicting primary key"):
        build_catalog((first, changed))


def test_added_changed_removed_files_replace_snapshot() -> None:
    """An updated inventory changes identity and never appends removed source rows."""
    first = _source("first.csv", "category_id,name,description\n1,Guitars,Original\n")
    second = _source("second.csv", "category_id,name,description\n2,Keys,Original\n")
    initial = build_catalog((first, second))
    updated = _source("first.csv", "category_id,name,description\n1,Guitars,Changed\n")
    replacement = build_catalog((updated,))
    assert initial.fingerprint != replacement.fingerprint
    assert replacement.tables["categories"].rows == (
        {"category_id": 1, "name": "Guitars", "description": "Changed"},
    )


def test_identical_shards_dedupe_but_conflicting_keys_fail() -> None:
    """Overlapping exports must never silently overwrite conflicting records."""
    text = "category_id,name,description\n1,Guitars,Description\n"
    assert (
        len(
            build_catalog((_source("a.csv", text), _source("b.csv", text)))
            .tables["categories"]
            .rows
        )
        == 1
    )
    with pytest.raises(DataError, match="Conflicting primary key"):
        build_catalog((_source("a.csv", text), _source("b.csv", text.replace("Guitars", "Keys"))))


def test_unknown_or_malformed_csv_is_not_guessed() -> None:
    """Reject unsupported domains and malformed rows with an actionable source name."""
    with pytest.raises(DataError, match="unknown.csv"):
        build_catalog((_source("unknown.csv", "something,else\na,b\n"),))
    with pytest.raises(DataError, match="wrong number"):
        build_catalog((_source("bad.csv", "category_id,name,description\n1,Name\n"),))
    with pytest.raises(DataError, match="supported business schema"):
        build_catalog(
            (
                _source(
                    "incomplete-products.csv",
                    "product_id,name,category_id,description,stock_quantity,status,"
                    "specs,created_at\n1,Name,1,Description,1,active,{},2026-01-01\n",
                ),
            )
        )


def test_foreign_keys_are_checked_when_related_tables_exist() -> None:
    """Allow partial capabilities but reject an inconsistent present relationship."""
    items = _source("items.csv", "order_id,quantity,product_id\n1,2,4\n")
    assert "order_items" in build_catalog((items,)).tables
    orders = _source(
        "orders.csv",
        "order_id,customer_id,order_date,status,total_brl,"
        "payment_method,tracking_code,estimated_delivery,notes\n"
        "2,1,2026-01-01,pending,1.00,pix,,,\n",
    )
    with pytest.raises(DataError, match="orphan"):
        build_catalog((items, orders))


def test_discovery_routes_extensions_and_fingerprints_contents(tmp_path: Path) -> None:
    """Discover nested PDF/CSV inputs while ignoring unrelated filesystem artifacts."""
    (tmp_path / "nested").mkdir()
    (tmp_path / "nested" / "catalog.CSV").write_text(
        "category_id,name,description\n1,Guitars,Description\n", encoding="utf-8"
    )
    (tmp_path / "POLICY.PDF").write_bytes(b"%PDF-test")
    (tmp_path / "notes.txt").write_text("Ignored", encoding="utf-8")
    sources = discover_sources(tmp_path)
    assert {source.relative_path for source in sources} == {"nested/catalog.CSV", "POLICY.PDF"}
    assert len(build_catalog(sources).tables) == 1


def test_original_fixture_preserves_missing_fields_and_historical_totals() -> None:
    """Treat original-data regression facts as fixtures, never ingestion assumptions."""
    catalog = build_catalog(discover_sources(Path(__file__).parents[1] / "data"))
    orders = {row["order_id"]: row for row in catalog.tables["orders"].rows}
    assert orders[3]["total_cents"] == 345000
    assert orders[20]["total_cents"] == 140000
    assert orders[9]["tracking_code"] is None
    product = next(row for row in catalog.tables["products"].rows if row["product_id"] == 95)
    assert product["price_cents"] == 219900
    assert product["search_name"] == "takaminegd20dreadnoughtnatural"


def test_persisted_snapshot_keeps_cents_integer_and_does_not_overwrite(tmp_path: Path) -> None:
    """Persist strict normalized tables and a fingerprint without destroying existing files."""
    catalog = build_catalog(discover_sources(Path(__file__).parents[1] / "data"))
    path = tmp_path / "snapshot.sqlite"
    write_catalog(catalog, path)
    connection = sqlite3.connect(path)
    try:
        assert connection.execute(
            "SELECT typeof(price_cents) FROM products LIMIT 1"
        ).fetchone() == ("integer",)
        assert connection.execute(
            "SELECT value FROM snapshot_metadata WHERE key = 'fingerprint'"
        ).fetchone() == (catalog.fingerprint,)
    finally:
        connection.close()
    with pytest.raises(DataError, match="overwrite"):
        write_catalog(catalog, path)


def test_persisted_percentages_retain_text_and_compare_numerically(tmp_path: Path) -> None:
    """Keep exact percentage strings while fixing comparisons on reopened SQLite snapshots."""
    fractional = "7.000000000000000000000000000000000000001"
    source = _source(
        "offers.csv",
        "promotion_id,product_id,discount_percent,description,is_active\n"
        f"1,1,8,Offer,true\n2,1,40,Offer,true\n3,1,{fractional},Offer,true\n",
    )
    path = tmp_path / "snapshot.sqlite"
    write_catalog(build_catalog((source,)), path)
    connection = sqlite3.connect(path)
    try:
        register_decimal_collation(connection)
        assert connection.execute(
            "SELECT discount_percent, typeof(discount_percent) FROM promotions "
            "ORDER BY discount_percent"
        ).fetchall() == [(fractional, "text"), ("8", "text"), ("40", "text")]
        assert connection.execute("SELECT MAX(discount_percent) FROM promotions").fetchone() == (
            "40",
        )
    finally:
        connection.close()
