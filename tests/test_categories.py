"""Check that the model sees exact category names and that they filter the real catalog."""

from typing import cast

from agent_fixtures import manual_snapshot

from emporio_agent.sql import QueryEngine

SNAPSHOT = manual_snapshot()


def test_schema_lists_exact_category_names() -> None:
    """The model receives "Violões" as written, not a spelling it has to guess."""
    schema = cast(dict[str, dict[str, dict[str, list[str]]]], SNAPSHOT.catalog.describe())
    names = schema["categories"]["observed_values"]["name"]
    assert "Violões" in names and "Ukuleles" in names and len(names) == 9


def test_exact_category_name_finds_every_available_acoustic_guitar() -> None:
    """The exact name returns the 12 violões up to R$ 1.000; an accent-free guess returns none."""
    engine = QueryEngine(SNAPSHOT.catalog)
    query = (
        "SELECT p.name FROM products p JOIN categories c ON p.category_id = c.category_id "
        "WHERE c.name = '{}' AND p.status = 'active' AND p.stock_quantity > 0 "
        "AND p.price_cents <= 100000"
    )
    exact = cast(
        list[object],
        engine.query(query.format("Violões"), verified_order_id=None).as_dict()["rows"],
    )
    guessed = cast(
        list[object], engine.query(query.format("Violao"), verified_order_id=None).as_dict()["rows"]
    )
    assert len(exact) == 12
    assert guessed == []
