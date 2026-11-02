"""Validate a closed SQL grammar and execute queries in a scoped subprocess."""

import json
import os
import re
import subprocess
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import sqlglot
from sqlglot import exp
from sqlglot.errors import SqlglotError

from emporio_agent.data import Catalog, Table, Value
from emporio_agent.money import MAX_CENTS, format_brl

MAX_ROWS = 50
MAX_RESULT_BYTES = 32 * 1024
QUERY_TIMEOUT_SECONDS = 3
_PUBLIC = {"products", "categories", "promotions"}
_PRIVATE = {"orders", "order_items"}
_JOIN_KEYS = frozenset(
    frozenset((left, right))
    for left, right in (
        (("products", "category_id"), ("categories", "category_id")),
        (("promotions", "product_id"), ("products", "product_id")),
        (("order_items", "order_id"), ("orders", "order_id")),
        (("order_items", "product_id"), ("products", "product_id")),
    )
)
_ALLOWED_NODES: frozenset[type[exp.Expr]] = frozenset(
    {
        exp.Select,
        exp.From,
        exp.Table,
        exp.Identifier,
        exp.Column,
        exp.Where,
        exp.EQ,
        exp.NEQ,
        exp.GT,
        exp.GTE,
        exp.LT,
        exp.LTE,
        exp.And,
        exp.Or,
        exp.Not,
        exp.Paren,
        exp.In,
        exp.Between,
        exp.Like,
        exp.Order,
        exp.Ordered,
        exp.Limit,
        exp.Offset,
        exp.Join,
        exp.TableAlias,
        exp.Alias,
        exp.Distinct,
        exp.Group,
        exp.Having,
        exp.Count,
        exp.Sum,
        exp.Avg,
        exp.Min,
        exp.Max,
        exp.Lower,
        exp.Upper,
        exp.Length,
        exp.Coalesce,
        exp.Null,
        exp.Boolean,
        exp.Literal,
        exp.Star,
        exp.Is,
    }
)
_AGGREGATES = (exp.Sum, exp.Avg, exp.Min, exp.Max)
_COMPARISONS = (exp.EQ, exp.NEQ, exp.GT, exp.GTE, exp.LT, exp.LTE)


class QueryError(ValueError):
    """A query was rejected or failed without publishing partial results."""


@dataclass(frozen=True)
class QueryResult:
    """A bounded result with exact monetary units and source provenance."""

    columns: tuple[str, ...]
    rows: tuple[tuple[Value, ...], ...]
    truncated: bool
    money_indexes: tuple[int, ...] = ()
    source_paths: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, object]:
        """Return model-facing values with money already formatted in BRL."""
        rendered: list[list[Value]] = []
        for row in self.rows:
            rendered.append(
                [
                    format_brl(value)
                    if index in self.money_indexes and isinstance(value, int)
                    else value
                    for index, value in enumerate(row)
                ]
            )
        result: dict[str, object] = {
            "columns": self.columns,
            "rows": rendered,
            "column_units": {self.columns[index]: "BRL" for index in self.money_indexes},
            "truncated": self.truncated,
            "sources": self.source_paths,
            "rounding": "Monetary AVG is rounded to cents using HALF_UP.",
        }
        if len(json.dumps(result, ensure_ascii=False).encode()) > MAX_RESULT_BYTES:
            raise QueryError("The formatted result is too large; narrow the query.")
        return result


def _direct(expression: exp.Expr) -> exp.Expr:
    while isinstance(expression, (exp.Alias, exp.Paren)):
        child = expression.this
        if not isinstance(child, exp.Expr):
            raise QueryError("Invalid SQL expression.")
        expression = child
    return expression


def _resolve_column(column: exp.Column, aliases: Mapping[str, Table]) -> tuple[str, str]:
    if column.catalog or column.db:
        raise QueryError("Database-qualified columns are not available.")
    name = column.name.casefold()
    if column.table:
        alias = column.table.casefold()
        table = aliases.get(alias)
        if table is None or name not in table.columns:
            raise QueryError("The SQL query references an unavailable column.")
        return alias, name
    candidates = [alias for alias, table in aliases.items() if name in table.columns]
    if len(candidates) != 1:
        raise QueryError("Use qualified names for ambiguous or unavailable columns.")
    return candidates[0], name


def _is_money(expression: exp.Expr, aliases: Mapping[str, Table]) -> bool:
    expression = _direct(expression)
    if isinstance(expression, _AGGREGATES):
        expression = _direct(expression.this)
    return isinstance(expression, exp.Column) and _resolve_column(expression, aliases)[1].endswith(
        "_cents"
    )


def _integer_literal(expression: exp.Expr) -> bool:
    return (
        isinstance(expression, exp.Literal)
        and not expression.is_string
        and len(expression.name) <= 19
        and expression.is_int
        and 0 <= int(expression.name) <= MAX_CENTS
    )


def _typed_money_operand(expression: exp.Expr, aliases: Mapping[str, Table]) -> bool:
    return _is_money(expression, aliases) or _integer_literal(expression)


def _validate_joins(
    query: exp.Select, aliases: Mapping[str, Table], names: Mapping[str, str]
) -> None:
    """Require every joined table to connect through a declared business foreign key."""
    source = query.args.get("from_")
    if not isinstance(source, exp.From) or not isinstance(source.this, exp.Table):
        raise QueryError("A query must start from one available business table.")
    if len(set(names.values())) != len(names):
        raise QueryError("Self-joins and repeated business tables are not permitted.")
    prior = {source.this.alias_or_name.casefold()}
    joins = cast(list[exp.Join], query.args.get("joins") or [])
    for join in joins:
        shape = (join.args.get("side", ""), join.args.get("kind", ""))
        if (
            shape not in {("", ""), ("", "INNER"), ("LEFT", ""), ("LEFT", "OUTER")}
            or not isinstance(join.this, exp.Table)
            or any(
                value
                for key, value in join.args.items()
                if key not in {"this", "on", "side", "kind"}
            )
        ):
            raise QueryError("Only INNER or LEFT joins with explicit key equality are permitted.")
        condition = join.args.get("on")
        if not isinstance(condition, exp.Expr) or not isinstance(_direct(condition), exp.EQ):
            raise QueryError("Each join requires one equality on a known business key.")
        equality = cast(exp.EQ, _direct(condition))
        left, right = _direct(equality.left), _direct(equality.right)
        if not isinstance(left, exp.Column) or not isinstance(right, exp.Column):
            raise QueryError("Join keys must be direct business columns.")
        left_alias, left_name = _resolve_column(left, aliases)
        right_alias, right_name = _resolve_column(right, aliases)
        new_alias = join.this.alias_or_name.casefold()
        connected = {left_alias, right_alias}
        keys = frozenset(((names[left_alias], left_name), (names[right_alias], right_name)))
        if (
            len(connected) != 2
            or new_alias not in connected
            or not connected.difference({new_alias}).issubset(prior)
            or keys not in _JOIN_KEYS
        ):
            raise QueryError(
                "Each join must connect its new table to a prior table by a known key."
            )
        prior.add(new_alias)


def _parse_select(sql: str) -> exp.Select:
    """Parse one bounded SELECT before consulting the available table scope."""
    if len(sql) > 16_384:
        raise QueryError("The SQL query is too long.")
    try:
        parsed = sqlglot.parse(sql, read="sqlite")
    except SqlglotError as exc:
        raise QueryError("The SQL query could not be parsed.") from exc
    if len(parsed) != 1 or not isinstance(parsed[0], exp.Select):
        raise QueryError("Only one plain SELECT query is permitted.")
    return parsed[0]


def _query_tables(
    query: exp.Select, tables: Mapping[str, Table]
) -> tuple[dict[str, Table], dict[str, str]]:
    """Resolve SQL table aliases only against the server-selected business tables."""
    aliases: dict[str, Table] = {}
    names: dict[str, str] = {}
    for table_node in query.find_all(exp.Table):
        name = table_node.name.casefold()
        alias = table_node.alias_or_name.casefold()
        if table_node.db or table_node.catalog or name not in tables or alias in aliases:
            raise QueryError("The SQL query references an unavailable table or alias.")
        aliases[alias] = tables[name]
        names[alias] = name
    if not aliases:
        raise QueryError("A query must read an available business table.")
    return aliases, names


def _expand_projections(query: exp.Select, aliases: Mapping[str, Table]) -> None:
    """Expand stars into explicit scoped columns so money positions remain identifiable."""
    expanded: list[exp.Expr] = []
    for projection in query.expressions:
        if isinstance(projection, exp.Star) or (
            isinstance(projection, exp.Column) and projection.is_star
        ):
            selected = aliases.items()
            if isinstance(projection, exp.Column):
                alias = projection.table.casefold()
                if alias not in aliases:
                    raise QueryError("Unknown table in star projection.")
                selected = {alias: aliases[alias]}.items()
            for alias, table in selected:
                expanded.extend(exp.column(column, table=alias) for column in table.columns)
        else:
            expanded.append(projection)
    query.set("expressions", expanded)


def _validate_money_filter(node: exp.Expr, aliases: Mapping[str, Table]) -> None:
    """Require integer-cent operands in monetary predicates and permit only NULL IS checks."""
    if isinstance(node, _COMPARISONS):
        left, right = node.left, node.right
        if _is_money(left, aliases) or _is_money(right, aliases):
            if not (_typed_money_operand(left, aliases) and _typed_money_operand(right, aliases)):
                raise QueryError("Compare money only with integer cents or another money column.")
    if isinstance(node, exp.Between) and _is_money(node.this, aliases):
        low = cast(exp.Expr, node.args["low"])
        high = cast(exp.Expr, node.args["high"])
        if not (_typed_money_operand(low, aliases) and _typed_money_operand(high, aliases)):
            raise QueryError("Monetary bounds must use integer cents.")
    if isinstance(node, exp.In) and _is_money(node.this, aliases):
        if not all(_typed_money_operand(value, aliases) for value in node.expressions):
            raise QueryError("Monetary lists must use integer cents.")
    if isinstance(node, exp.Is) and _is_money(node.this, aliases):
        if not isinstance(node.expression, exp.Null):
            raise QueryError("IS comparisons on money support only NULL checks.")


def _validate_numeric_coercion(node: exp.Expr, aliases: Mapping[str, Table]) -> None:
    """Preserve exact monetary and percentage types across functions and text patterns."""
    if isinstance(node, (exp.Lower, exp.Upper, exp.Length, exp.Coalesce, exp.Like)):
        if any(
            _is_money(column, aliases) or _resolve_column(column, aliases)[1] == "discount_percent"
            for column in node.find_all(exp.Column)
        ):
            raise QueryError("Monetary and percentage values cannot be coerced by SQL functions.")


def _validate_nodes(query: exp.Select, aliases: Mapping[str, Table]) -> None:
    """Enforce the AST allowlist, column scope, numeric literals and exact money rules."""
    for node in query.walk():
        if type(node) not in _ALLOWED_NODES:
            raise QueryError(
                "Only simple SELECT, joins, filters, and supported aggregates are allowed."
            )
        if isinstance(node, exp.Literal) and not node.is_string:
            if not _integer_literal(node):
                raise QueryError("Numeric SQL literals must be nonnegative signed-64-bit integers.")
        if isinstance(node, exp.Column) and not node.is_star:
            _resolve_column(node, aliases)
        _validate_numeric_coercion(node, aliases)
        _validate_money_filter(node, aliases)
        if isinstance(node, _AGGREGATES):
            argument = _direct(node.this)
            if not isinstance(argument, exp.Column):
                raise QueryError("Aggregates require one direct column.")
            if isinstance(node, exp.Avg) and not _is_money(argument, aliases):
                raise QueryError("AVG is supported only for exact monetary columns.")


def _is_percentage(expression: exp.Expr, aliases: Mapping[str, Table]) -> bool:
    """Identify a percentage column or its exact MIN/MAX result before trusted rewriting."""
    expression = _direct(expression)
    if isinstance(expression, (exp.Min, exp.Max)):
        expression = _direct(expression.this)
    return (
        isinstance(expression, exp.Column)
        and _resolve_column(expression, aliases)[1] == "discount_percent"
    )


def _decimal_expression(expression: exp.Expr) -> exp.Collate:
    """Restore TEXT affinity and exact numeric collation without converting through a float."""
    return exp.Collate(
        this=exp.Cast(this=expression.copy(), to=exp.DataType.build("TEXT")),
        expression=exp.Var(this="exact_decimal"),
    )


def _normalize_percentages(query: exp.Select, aliases: Mapping[str, Table]) -> None:
    """Restore exact comparison rules lost by aggregate results and IN right-hand operands."""
    for predicate in list(query.find_all(exp.In)):
        if not _is_percentage(predicate.this, aliases) and any(
            _is_percentage(value, aliases) for value in predicate.expressions
        ):
            predicate.set("this", _decimal_expression(predicate.this))
    for aggregate in list(query.find_all(exp.Min, exp.Max)):
        if _is_percentage(aggregate, aliases):
            aggregate.replace(_decimal_expression(aggregate))


def _prepare_execution(query: exp.Select, aliases: Mapping[str, Table]) -> exp.Select:
    """Adapt validated monetary aggregates to exact worker functions and cap output rows."""
    _normalize_percentages(query, aliases)
    for aggregate in list(query.find_all(exp.Sum, exp.Avg)):
        if _is_money(_direct(aggregate.this), aliases):
            replacement = exp.Anonymous(
                this="exact_sum" if isinstance(aggregate, exp.Sum) else "exact_avg",
                expressions=[aggregate.this.copy()],
            )
            aggregate.replace(replacement)
    limit = cast(exp.Limit | None, query.args.get("limit"))
    requested = MAX_ROWS + 1
    if limit is not None:
        value = limit.expression
        if not isinstance(value, exp.Literal) or not value.is_int:
            raise QueryError("LIMIT requires a nonnegative integer literal.")
        requested = min(int(value.this), requested)
    return query.limit(requested)


def _validate_query(sql: str, tables: Mapping[str, Table]) -> tuple[str, tuple[int, ...], set[str]]:
    """Validate the complete query before applying exact aggregates and result bounds."""
    query = _parse_select(sql)
    aliases, names = _query_tables(query, tables)
    _validate_joins(query, aliases, names)
    _expand_projections(query, aliases)
    _validate_nodes(query, aliases)
    money_indexes = tuple(
        index
        for index, projection in enumerate(query.expressions)
        if _is_money(projection, aliases)
    )
    query = _prepare_execution(query, aliases)
    return query.sql(dialect="sqlite"), money_indexes, set(names.values())


class QueryEngine:
    """Execute validated SQL over public data and a server-verified order scope."""

    def __init__(self, catalog: Catalog) -> None:
        """Keep the immutable catalog; create no shared database connection."""
        self.catalog = catalog

    def verify_order(self, order_id: int | str, email: str) -> int | None:
        """Match an order ID and its customer's email without exposing other customers."""
        raw_id = str(order_id)
        if not re.fullmatch(r"\d{1,19}", raw_id):
            return None
        wanted = int(raw_id)
        orders = self.catalog.tables.get("orders")
        customers = self.catalog.tables.get("customers")
        if orders is None or customers is None:
            return None
        customer_ids = {
            row["customer_id"]
            for row in customers.rows
            if isinstance(row["email"], str)
            and row["email"].strip().casefold() == email.strip().casefold()
        }
        return (
            wanted
            if any(
                row["order_id"] == wanted and row["customer_id"] in customer_ids
                for row in orders.rows
            )
            else None
        )

    def customer_first_name(self, verified_order_id: int) -> str | None:
        """Return only the first name of a verified order's customer, for greetings."""
        orders = self.catalog.tables.get("orders")
        customers = self.catalog.tables.get("customers")
        if orders is None or customers is None or "name" not in customers.columns:
            return None
        owner = next(
            (row["customer_id"] for row in orders.rows if row["order_id"] == verified_order_id),
            None,
        )
        name = next(
            (
                row["name"]
                for row in customers.rows
                if owner is not None and row["customer_id"] == owner
            ),
            None,
        )
        return name.split()[0] if isinstance(name, str) and name.split() else None

    def _scope(self, verified_order_id: int | None) -> dict[str, Table]:
        scoped = {name: table for name, table in self.catalog.tables.items() if name in _PUBLIC}
        if verified_order_id is not None:
            for name in _PRIVATE:
                table = self.catalog.tables.get(name)
                if table is not None:
                    columns = tuple(
                        column for column in table.columns if column not in {"customer_id", "notes"}
                    )
                    scoped[name] = Table(
                        columns,
                        tuple(
                            {column: row[column] for column in columns}
                            for row in table.rows
                            if row["order_id"] == verified_order_id
                        ),
                        table.source_paths,
                    )
        return scoped

    def query(self, sql: str, *, verified_order_id: int | None = None) -> QueryResult:
        """Validate and execute one bounded query with no API credentials in its worker."""
        tables = self._scope(verified_order_id)
        validated, money_indexes, used = _validate_query(sql, tables)
        payload = {
            "sql": validated,
            "tables": {
                name: {"columns": table.columns, "rows": table.rows}
                for name, table in tables.items()
                if name in used
            },
        }
        environment = {
            key: value
            for key, value in os.environ.items()
            if key.upper()
            in {
                "SYSTEMROOT",
                "WINDIR",
                "TEMP",
                "TMP",
                "PATH",
                "LANG",
                "LC_ALL",
            }
        }
        environment["PYTHONPATH"] = str(Path(__file__).resolve().parents[1])
        environment["PYTHONIOENCODING"] = "utf-8"
        try:
            process = subprocess.run(
                [sys.executable, "-m", "emporio_agent.sql_worker"],
                input=json.dumps(payload, ensure_ascii=False),
                capture_output=True,
                text=True,
                encoding="utf-8",
                env=environment,
                timeout=QUERY_TIMEOUT_SECONDS,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise QueryError("The query exceeded its three-second time limit.") from exc
        if process.returncode != 0 or len(process.stdout.encode()) > MAX_RESULT_BYTES:
            raise QueryError("The restricted query worker failed or returned too much data.")
        decoded: object = json.loads(process.stdout)
        if not isinstance(decoded, dict):
            raise QueryError("The query worker returned an invalid result.")
        response = cast(dict[str, object], decoded)
        if "error" in response:
            raise QueryError(str(response["error"]))
        columns = cast(list[str], response["columns"])
        raw_rows = cast(list[list[Value]], response["rows"])
        sources = tuple(sorted({path for name in used for path in tables[name].source_paths}))
        return QueryResult(
            tuple(columns),
            tuple(tuple(row) for row in raw_rows),
            bool(response["truncated"]),
            money_indexes,
            sources,
        )
