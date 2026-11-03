"""Explicitly opt-in paid API smoke test; normal CI never runs it."""

import os
from dataclasses import replace

import pytest

from emporio_agent.access import conversation_token
from emporio_agent.agent import build_agent, close_agent, run_chat
from emporio_agent.config import Settings
from emporio_agent.runtime import StoreRuntime

pytestmark = pytest.mark.live


def test_real_embeddings_and_claude_tools() -> None:
    """Check multilingual policy retrieval and one grounded answer with approved API keys.

    Trial quotas may require a later retry, which resumes completed embeddings from the
    persistent live-smoke cache. Live runs still send billable Claude requests.
    """
    if os.environ.get("RUN_LIVE_TESTS") != "1":
        pytest.skip("Set RUN_LIVE_TESTS=1 explicitly to allow paid provider calls.")
    settings = Settings.from_env()
    settings = replace(settings, runtime_dir=settings.runtime_dir / "live-smoke")
    if not settings.anthropic_key or not settings.voyage_key:
        pytest.skip(
            "Both ANTHROPIC_API_KEY and VOYAGE_API_KEY are required for the live smoke test."
        )
    snapshot = StoreRuntime(settings).refresh()
    assert snapshot.policies is not None, "The live fixture must supply policy PDFs."
    for question in ("formas de pagamento e parcelamento", "payment methods and installments"):
        hits = snapshot.policies.search(question)
        assert hits and any(hit.citations for hit in hits)
    product = next(
        row
        for row in snapshot.catalog.tables["products"].rows
        if row["status"] == "active"
        and isinstance(row["stock_quantity"], int)
        and row["stock_quantity"] > 0
    )
    agent, tools = build_agent(settings, snapshot, conversation_token())
    try:
        answer = run_chat(
            agent,
            f"What is the current price and stock of product ID {product['product_id']}? "
            "Please consult your data tool and cite its source.",
        )
        assert answer.strip() and "R$" in answer
        assert any(str(source.get("file", "")).lower().endswith(".csv") for source in tools.sources)
    finally:
        close_agent(agent)
