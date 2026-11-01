"""Exercise the scoped SQL grammar, exact aggregates, and result/resource boundaries."""

import hashlib
from pathlib import Path

import pytest

from emporio_agent.data import SourceFile, build_catalog, discover_sources
from emporio_agent.money import MAX_CENTS
from emporio_agent.sql import QueryEngine, QueryError


def _source(name: str, text: str) -> SourceFile:
    content = text.encode()
    return SourceFile(name, content, hashlib.sha256(content).hexdigest())


def _engine() -> QueryEngine:
    return QueryEngine(build_catalog(discover_sources(Path(__file__).parents[1] / "data")))


def _join_engine() -> QueryEngine:
    sources = (
        _source(
            "groups.csv",
            "category_id,name,description\n1,Guitars,Instruments\n2,Other,Empty group\n",
        ),
        _source(
            "instruments.csv",
            "product_id,price_brl,name,category_id,description,stock_quantity,status,specs,created_at\n"
            "10,10.01,First,1,Description,1,active,{},2026-01-01\n"
            "11,20.02,Second,1,Description,1,active,{},2026-01-01\n",
        ),
        _source(
            "offers.csv",
            "promotion_id,product_id,discount_percent,description,is_active\n101,10,5,Offer,true\n",
        ),
        _source(
            "purchases.csv",
            "order_id,customer_id,order_date,status,total_brl,payment_method,tracking_code,estimated_delivery,notes\n"
            "501,5,2026-01-01,shipped,10.01,pix,,,Private first\n"
            "502,6,2026-01-01,pending,20.02,pix,,,Private second\n",
        ),
        _source("lines.csv", "order_id,quantity,product_id\n501,1,10\n502,1,11\n"),
    )
    return QueryEngine(build_catalog(sources))


def test_known_keys_support_inner_left_and_multiple_joins() -> None:
    """Traverse public foreign keys in either direction without inventing relationships."""
    engine = _join_engine()
    result = engine.query(
        "SELECT p.product_id, c.name, m.discount_percent FROM products p "
        "INNER JOIN categories c ON c.category_id = p.category_id "
        "LEFT JOIN promotions m ON (p.product_id = m.product_id) ORDER BY p.product_id"
    )
    assert result.rows == ((10, "Guitars", "5"), (11, "Guitars", None))
    assert engine.query(
        "SELECT c.category_id, p.product_id FROM categories c "
        "LEFT OUTER JOIN products p ON c.category_id = p.category_id "
        "ORDER BY c.category_id, p.product_id"
    ).rows == ((1, 10), (1, 11), (2, None))


def test_verified_order_joins_keep_private_rows_scoped() -> None:
    """Known joins retain the server-selected order subset across public product lookups."""
    result = _join_engine().query(
        "SELECT o.order_id, p.name, c.name AS category FROM orders o "
        "JOIN order_items i ON i.order_id = o.order_id "
        "JOIN products p ON p.product_id = i.product_id "
        "JOIN categories c ON p.category_id = c.category_id",
        verified_order_id=501,
    )
    assert result.rows == ((501, "First", "Guitars"),)


@pytest.mark.parametrize(
    "join",
    [
        ", categories c WHERE p.category_id = c.category_id",
        "CROSS JOIN categories c",
        "CROSS JOIN categories c ON p.category_id = c.category_id",
        "JOIN categories c",
        "JOIN categories c ON 1 = 1",
        "JOIN categories c ON p.product_id = c.category_id",
        "JOIN categories c ON p.category_id = p.category_id",
        "JOIN categories c ON p.category_id = c.category_id OR 1 = 1",
        "JOIN categories c ON p.category_id = c.category_id AND p.status = 'active'",
        "JOIN products q ON p.product_id = q.product_id",
        "NATURAL JOIN categories c",
        "JOIN categories c USING (category_id)",
        "RIGHT JOIN categories c ON p.category_id = c.category_id",
        "FULL JOIN categories c ON p.category_id = c.category_id",
        "JOIN categories c ON p.product_id = m.product_id "
        "JOIN promotions m ON p.product_id = m.product_id",
        "JOIN order_items i ON p.product_id = i.product_id "
        "JOIN promotions m ON i.product_id = m.product_id",
    ],
)
def test_cartesian_nonkey_and_unsupported_joins_are_rejected(join: str) -> None:
    """Reject fan-out caused by missing keys, guessed keys, or disconnected new aliases."""
    with pytest.raises(QueryError):
        _join_engine().query(f"SELECT p.product_id FROM products p {join}", verified_order_id=501)


def test_all_null_left_join_money_aggregates_remain_sql_null() -> None:
    """Unmatched rows distinguish unknown money from an actual zero sum or average."""
    result = _join_engine().query(
        "SELECT c.category_id, SUM(p.price_cents) AS total, AVG(p.price_cents) AS mean "
        "FROM categories c LEFT JOIN products p ON c.category_id = p.category_id "
        "GROUP BY c.category_id ORDER BY c.category_id"
    )
    assert result.rows == ((1, 3003, 1502), (2, None, None))
    assert result.as_dict()["rows"] == [[1, "BRL 30.03", "BRL 15.02"], [2, None, None]]


def test_catalog_query_has_exact_money_and_provenance() -> None:
    """Return ready-to-use BRL display values without asking an LLM to divide cents."""
    result = _engine().query("SELECT name, price_cents FROM products WHERE product_id = 95")
    assert result.rows[0][1] == 219900
    assert result.as_dict()["rows"] == [["Takamine GD20 Dreadnought Natural", "BRL 2,199.00"]]
    assert result.money_indexes == (1,)
    assert result.source_paths


def test_normalized_name_and_availability_queries() -> None:
    """Support exact normalized matching and combine source status with stock quantity."""
    result = _engine().query(
        "SELECT product_id FROM products WHERE search_name LIKE '%takaminegd20%' "
        "AND status = 'active' AND stock_quantity > 0"
    )
    assert result.rows == ((95,),)


@pytest.mark.parametrize(
    "query",
    [
        "DELETE FROM products",
        "SELECT * FROM customers",
        "SELECT * FROM orders",
        "SELECT * FROM sqlite_master",
        "SELECT * FROM pragma_table_info('products')",
        "SELECT name FROM products; SELECT name FROM products;",
        "ATTACH DATABASE ':memory:' AS x",
        "PRAGMA table_info(products)",
        "SELECT load_extension('anything') FROM products",
        "SELECT randomblob(10) FROM products",
        "SELECT price_cents * 2 FROM products",
        "SELECT price_cents + 1 FROM products",
        "SELECT CAST(price_cents AS REAL) FROM products",
        "SELECT CAST(CAST(price_cents AS REAL) AS INTEGER) FROM products",
        "SELECT price_cents / 100 FROM products",
        "SELECT price_cents || '' FROM products",
        "SELECT SUM(price_cents + 0) FROM products",
        "SELECT AVG(stock_quantity) FROM products",
        "SELECT COALESCE(price_cents, 0) FROM products",
        "SELECT product_id FROM products WHERE price_cents < '100000'",
        "SELECT product_id FROM products WHERE price_cents < 100000.0",
        "SELECT product_id FROM products WHERE price_cents LIKE '%'",
        "SELECT product_id FROM products WHERE price_cents < 9223372036854775808",
        "SELECT SUM(price_cents) FROM products HAVING SUM(price_cents) < '100000'",
        "SELECT product_id FROM products WHERE price_cents IS TRUE",
        "SELECT name FROM products UNION SELECT email FROM customers",
        "SELECT name FROM (SELECT name FROM products)",
        "WITH p AS (SELECT * FROM products) SELECT name FROM p",
    ],
)
def test_unsafe_or_coercive_sql_is_rejected(query: str) -> None:
    """Deny unauthorized data, unsafe functions, and every arithmetic/coercion escape."""
    with pytest.raises(QueryError):
        _engine().query(query)


def test_verified_order_is_a_data_scope_not_a_model_filter() -> None:
    """A malicious WHERE clause cannot escape the server-selected order subset."""
    engine = _engine()
    assert engine.verify_order(1, "invalid@example.com") is None
    orders = engine.catalog.tables["orders"].rows
    customer_id = next(row["customer_id"] for row in orders if row["order_id"] == 1)
    email = next(
        row["email"]
        for row in engine.catalog.tables["customers"].rows
        if row["customer_id"] == customer_id
    )
    assert isinstance(email, str)
    verified = engine.verify_order("1", email.upper())
    assert verified == 1
    result = engine.query(
        "SELECT order_id FROM orders WHERE order_id = 2 OR 1 = 1", verified_order_id=verified
    )
    assert result.rows == ((1,),)
    with pytest.raises(QueryError):
        engine.query("SELECT email FROM customers", verified_order_id=verified)


def test_row_limit_and_count_star() -> None:
    """Bound output independently of the model and support zero-column aggregate reads."""
    engine = _engine()
    assert engine.query("SELECT COUNT(*) AS count FROM products").rows == ((65,),)
    result = engine.query("SELECT product_id FROM products ORDER BY product_id")
    assert len(result.rows) == 50
    assert result.truncated


def test_aggregates_round_exactly_and_reject_overflow() -> None:
    """SUM stays exact and AVG rounds once using HALF_UP, including very large integers."""
    headers = (
        "product_id,price_brl,name,category_id,description,stock_quantity,status,specs,created_at\n"
    )
    source = _source(
        "prices.csv",
        headers
        + "1,1.00,First,1,Description,1,active,{},2026-01-01\n"
        + "2,1.01,Second,1,Description,1,active,{},2026-01-01\n",
    )
    engine = QueryEngine(build_catalog((source,)))
    assert engine.query(
        "SELECT SUM(price_cents) AS total, AVG(price_cents) AS mean FROM products"
    ).rows == ((201, 101),)
    huge = _source(
        "huge.csv",
        headers
        + "1,92233720368547758.07,First,1,Description,1,active,{},2026-01-01\n"
        + "2,0.01,Second,1,Description,1,active,{},2026-01-01\n",
    )
    huge_engine = QueryEngine(build_catalog((huge,)))
    assert huge_engine.query("SELECT price_cents FROM products WHERE product_id = 1").rows == (
        (MAX_CENTS,),
    )
    with pytest.raises(QueryError):
        huge_engine.query("SELECT SUM(price_cents) FROM products")


def test_hard_timeout_terminates_worker(monkeypatch: pytest.MonkeyPatch) -> None:
    """A parent deadline ends the subprocess instead of leaving a query running."""
    monkeypatch.setattr("emporio_agent.sql.QUERY_TIMEOUT_SECONDS", 0.001)
    with pytest.raises(QueryError, match="time limit"):
        _engine().query("SELECT product_id FROM products")


def test_oversized_result_is_rejected() -> None:
    """A single huge text field cannot exceed the model-facing result budget."""
    source = _source("large.csv", "category_id,name,description\n1,Guitars," + "x" * 40_000 + "\n")
    with pytest.raises(QueryError, match="too large"):
        QueryEngine(build_catalog((source,))).query("SELECT description FROM categories")


def test_formatted_result_budget_includes_all_source_provenance() -> None:
    """Many shards cannot bypass the result cap through added source metadata."""
    sources = tuple(
        _source(
            f"{'x' * 380}/{index}.csv",
            f"category_id,name,description\n{index + 1},Guitars,Description\n",
        )
        for index in range(90)
    )
    result = QueryEngine(build_catalog(sources)).query("SELECT COUNT(*) FROM categories")
    assert result.rows == ((90,),)
    with pytest.raises(QueryError, match="formatted result is too large"):
        result.as_dict()


def test_missing_private_tables_disable_verification_without_breaking_public_data() -> None:
    """Partial source sets retain available capabilities and never fabricate an order scope."""
    source = _source("categories.csv", "category_id,name,description\n1,Guitars,Description\n")
    engine = QueryEngine(build_catalog((source,)))
    assert engine.verify_order(1, "someone@example.com") is None
    assert engine.query("SELECT COUNT(*) FROM categories").rows == ((1,),)
    with pytest.raises(QueryError, match="unavailable table"):
        engine.query("SELECT order_id FROM orders", verified_order_id=1)
