"""Run the real store UI with native provider HTTP fixtures, exclusively for browser tests."""

import json
import os
from pathlib import Path
from typing import cast

import streamlit as st

from app import render
from emporio_agent.config import Settings
from emporio_agent.runtime import StoreRuntime
from tests.agent_fixtures import (
    CharacterTokenizer,
    ClaudeStub,
    FixedEmbedder,
    JsonObject,
    latest_user_text,
    text_response,
    tool_response,
    tool_results,
)


def _rows(results: list[JsonObject]) -> list[list[object]]:
    """Read only genuine query rows returned through the provider's tool-result payload."""
    for result in reversed(results):
        rows = result.get("rows")
        if isinstance(rows, list):
            return [
                cast(list[object], row) for row in cast(list[object], rows) if isinstance(row, list)
            ]
    return []


def respond(body: JsonObject) -> JsonObject:
    """Choose native tool requests while deriving factual answers from actual tool outputs.

    Parameters:
        body: The complete native Anthropic request assembled by Agno.

    Returns:
        A native tool-use or final-text response without contacting an external service.
    """
    question = latest_user_text(body).casefold()
    results = tool_results(body)
    rows = _rows(results)
    if "force a request failure" in question:
        raise RuntimeError("Deliberate test-only provider failure")
    if "my favourite instrument is" in question:
        return text_response("I will remember Bluejay Acoustic Guitar.")
    if "remember my favourite" in question:
        remembered = "My favourite instrument is Bluejay Acoustic Guitar." in json.dumps(body)
        return text_response(
            "Your favourite instrument is Bluejay Acoustic Guitar."
            if remembered
            else "I do not have a favourite instrument in this conversation."
        )
    if "return policy" in question or "shipping policy" in question:
        if results:
            passages = results[-1].get("passages")
            if isinstance(passages, list):
                text = "\n".join(
                    str(cast(JsonObject, passage).get("text", ""))
                    for passage in cast(list[object], passages)
                    if isinstance(passage, dict)
                )
                return text_response(text)
            return text_response("Policy retrieval is unavailable.")
        return tool_response("search_policies", {"query": question})
    if "show order 7 for alex@example.test" in question:
        if rows:
            number, status, tracking = rows[0]
            return text_response(f"Order {number} is {status}. Tracking: {tracking}.")
        if any(result.get("verified") is True for result in results):
            return tool_response(
                "query_data",
                {"sql": "SELECT order_id, status, tracking_code FROM orders"},
                call_id="toolu_order_query",
            )
        return tool_response("verify_order", {"order_id": 7, "email": "alex@example.test"})
    if "my order status" in question:
        if rows:
            number, status, tracking = rows[0]
            return text_response(f"Order {number} is {status}. Tracking: {tracking}.")
        if not results:
            return tool_response(
                "query_data", {"sql": "SELECT order_id, status, tracking_code FROM orders"}
            )
        return text_response("Please provide your order number and email.")
    if "price" in question:
        if rows:
            name, price = rows[0]
            return text_response(f"{name} costs {price}.")
        return tool_response(
            "query_data", {"sql": "SELECT name, price_cents FROM products WHERE product_id = 701"}
        )
    return text_response("How can I help with the store?")


@st.cache_resource(show_spinner=False)
def runtime(settings: Settings) -> StoreRuntime:
    """Exercise real discovery and snapshots with deterministic embedding and tokenizer inputs."""
    if not settings.voyage_key:
        return StoreRuntime(settings)
    return StoreRuntime(settings, embedder=FixedEmbedder(), tokenizer=CharacterTokenizer())


scenario = os.environ.get("EMPORIO_E2E_SCENARIO", "normal")
settings = Settings(
    data_dir=Path(os.environ["EMPORIO_E2E_DATA_DIR"]),
    runtime_dir=Path(os.environ["EMPORIO_E2E_RUNTIME_DIR"]),
    anthropic_key="" if scenario == "missing-key" else "fixture-anthropic-key",
    voyage_key="" if scenario == "missing-voyage" else "fixture-voyage-key",
)
render(settings, runtime_factory=runtime, transport_factory=lambda: ClaudeStub(respond).transport)
