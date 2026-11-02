"""Exercise the native Agno/Anthropic loop with real tools and durable sessions."""

import json
import sqlite3
from collections.abc import Generator, Sequence
from contextlib import closing, contextmanager
from dataclasses import replace
from datetime import datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import cast

import httpx2
import pytest
from agent_fixtures import (
    ClaudeStub,
    FixedEmbedder,
    JsonObject,
    latest_user_text,
    manual_snapshot,
    small_snapshot,
    text_response,
    tool_response,
    tool_results,
)
from agno.agent import Agent
from agno.models.anthropic import Claude
from agno.models.message import Message

from emporio_agent.agent import (
    ConversationAccess,
    StoreTools,
    build_agent,
    chat_messages,
    close_agent,
    conversation_token,
    run_chat,
)
from emporio_agent.config import Settings
from emporio_agent.documents import PolicyIndex
from emporio_agent.runtime import Snapshot
from emporio_agent.sql import QueryEngine


def test_policy_tool_keeps_safe_provider_recovery_guidance(tmp_path: Path) -> None:
    """A translated provider failure reaches the real agent tool result with useful guidance."""

    class UnavailableEmbedder(FixedEmbedder):
        def embed_query(self, query: str) -> tuple[float, ...]:
            """Represent the safe error raised at the hosted embedding boundary."""
            raise ValueError("The embedding provider request limit was reached. Retry later.")

    snapshot = small_snapshot()
    assert snapshot.policies is not None
    policies = PolicyIndex(
        snapshot.policies.chunks,
        snapshot.policies.sections,
        UnavailableEmbedder(),
        snapshot.policies.fingerprint,
    )
    with _agent(
        _settings(tmp_path),
        replace(snapshot, policies=policies),
        conversation_token(),
        [
            tool_response("search_policies", {"query": "payment"}),
            text_response("Please retry later."),
        ],
    ) as (agent, _, stub):
        run_chat(agent, "What payment methods are available?")
        assert "request limit was reached. Retry later." in str(tool_results(stub.requests[-1]))


def test_valid_order_details_recover_after_repeated_failures_and_resume(tmp_path: Path) -> None:
    """Wrong details never permanently lock a conversation out of valid identification."""
    settings = _settings(tmp_path)
    snapshot = small_snapshot()
    token = conversation_token()
    with _agent(settings, snapshot, token, []) as (_, tools, _):
        for _ in range(4):
            assert tools.verify_order(501, "wrong@example.test")["verified"] is False
    with _agent(settings, snapshot, token, []) as (_, tools, _):
        assert tools.verify_order(501, "alex@example.test")["verified"] is True
        assert tools.query_data("SELECT order_id FROM orders")["rows"] == [[501]]
        assert tools.verify_order(501, "wrong@example.test")["verified"] is False
        assert "error" in tools.query_data("SELECT order_id FROM orders")


def test_old_verification_schema_migrates_without_losing_authorization(tmp_path: Path) -> None:
    """Remove the legacy failure counter while retaining a valid generation-bound scope."""
    path = tmp_path / "sessions.sqlite3"
    token = conversation_token()
    access = ConversationAccess(path, token, "generation")
    engine = QueryEngine(small_snapshot().catalog)
    assert access.verify(engine, 501, "alex@example.test")
    with closing(sqlite3.connect(path)) as connection, connection:
        connection.execute(
            "ALTER TABLE store_access ADD COLUMN failures INTEGER NOT NULL DEFAULT 3"
        )
    resumed = ConversationAccess(path, token, "generation")
    assert resumed.verified_order() == 501
    assert resumed.verify(engine, 501, "alex@example.test")
    with closing(sqlite3.connect(path)) as connection:
        columns = {row[1] for row in connection.execute("PRAGMA table_info(store_access)")}
    assert "failures" not in columns


@pytest.mark.parametrize("second_rate", ["10", "20"])
def test_multiple_active_promotions_require_unambiguous_discount(
    tmp_path: Path, second_rate: str
) -> None:
    """Repeated equal discounts are safe; distinct offers have no implicit priority."""
    snapshot = small_snapshot(price_cents=10000, promotion_percent="10")
    promotions = snapshot.catalog.tables["promotions"]
    other = dict(promotions.rows[0])
    other.update(promotion_id=302, discount_percent=second_rate)
    tables = dict(snapshot.catalog.tables)
    tables["promotions"] = replace(promotions, rows=(*promotions.rows, other))
    snapshot = replace(snapshot, catalog=replace(snapshot.catalog, tables=tables))
    with _agent(_settings(tmp_path), snapshot, conversation_token(), []) as (_, tools, _):
        result = tools.calculate_quote([{"product_id": 101}])
    if second_rate == "20":
        assert "Staff must confirm" in str(result["error"])
        assert "total" not in result
    else:
        assert result["total"] == "R$ 90,00"
        assert result["policy_note"] == "PIX discount does not stack with a promotional price."


def test_provider_failure_is_not_presented_as_a_completed_answer(tmp_path: Path) -> None:
    """Agno error RunOutputs must become a controlled UI error and stay out of history."""

    def fail(request: httpx2.Request) -> httpx2.Response:
        raise httpx2.ConnectError("Fixture network failure", request=request)

    token = conversation_token()
    agent, _ = build_agent(
        _settings(tmp_path), small_snapshot(), token, transport=httpx2.MockTransport(fail)
    )
    try:
        with pytest.raises(ValueError, match="did not complete"):
            run_chat(agent, "Hello")
        assert chat_messages(agent, token) == []
    finally:
        close_agent(agent)


def _settings(tmp_path: Path) -> Settings:
    return Settings(tmp_path / "data", tmp_path / "runtime", anthropic_key="fixture-key")


@contextmanager
def _agent(
    settings: Settings,
    snapshot: Snapshot,
    token: str,
    responses: Sequence[JsonObject],
) -> Generator[tuple[Agent, StoreTools, ClaudeStub]]:
    stub = ClaudeStub(responses)
    agent, tools = build_agent(settings, snapshot, token, transport=stub.transport)
    try:
        yield agent, tools, stub
    finally:
        close_agent(agent)


def _rows(result: JsonObject) -> list[object]:
    value = result.get("rows")
    assert isinstance(value, list)
    return cast(list[object], value)


def _additional_input(agent: Agent) -> list[object]:
    """Narrow an upstream field whose third-party dictionary annotation is untyped."""
    value = cast(list[object] | None, getattr(agent, "additional_input"))
    return value or []


def test_deployment_settings_reach_native_sdk_and_store_context(tmp_path: Path) -> None:
    """Apply configured model limits, HTTP timeout and store time zone to the real agent."""
    settings = replace(
        _settings(tmp_path),
        model="configured-claude-model",
        claude_timeout_seconds=12.5,
        max_output_tokens=777,
        max_tool_calls=4,
        store_timezone="Asia/Tokyo",
    )
    with _agent(settings, small_snapshot(), conversation_token(), [text_response("Hello.")]) as (
        agent,
        _,
        stub,
    ):
        assert agent.tool_call_limit == 4
        model = agent.model
        assert isinstance(model, Claude) and model.client is not None
        client = cast(httpx2.Client, getattr(model.client, "_client"))
        assert client.timeout.read == 12.5
        context = _additional_input(agent)[-1]
        assert isinstance(context, Message) and isinstance(context.content, str)
        payload = cast(dict[str, object], json.loads(context.content))
        timestamp = payload["store_local_time"]
        assert isinstance(timestamp, str)
        assert datetime.fromisoformat(timestamp).utcoffset() == timedelta(hours=9)
        assert run_chat(agent, "Hi") == "Hello."
        assert stub.requests[0]["model"] == "configured-claude-model"
        assert stub.requests[0]["max_tokens"] == 777


@pytest.mark.parametrize("model_id", ["claude-haiku-5-5", "claude-sonnet-5-5"])
def test_native_tool_loop_and_sdk_configuration(tmp_path: Path, model_id: str) -> None:
    """Run real SQL through a native tool round trip and inspect the actual HTTP payload."""
    settings = replace(_settings(tmp_path), model=model_id)
    with _agent(
        settings,
        small_snapshot(),
        conversation_token(),
        [
            tool_response(
                "query_data",
                {"sql": "SELECT name, price_cents FROM products WHERE product_id = 101"},
            ),
            text_response("The Demo Guitar costs BRL 100.01.", thinking=True),
        ],
    ) as (agent, tools, stub):
        assert run_chat(agent, "Quanto custa o violão?") == "The Demo Guitar costs BRL 100.01."
        assert len(stub.requests) == 2
        first = stub.requests[0]
        assert first["model"] == settings.model
        assert first["max_tokens"] == 2048
        assert first["thinking"] == {"type": "adaptive"}
        assert first["output_config"] == {"effort": "low"}
        assert all(name not in first for name in ("temperature", "top_p", "top_k"))
        system = " ".join(str(first["system"]).replace("\\n", " ").split())
        assert "use Brazilian Portuguese" in system
        assert "Always answer in English" not in system
        assert "never claim to be a human attendant" in system
        assert "cache_control" in str(first["system"])
        assert "cache_control" in str(first["tools"])
        assert latest_user_text(first) == "Quanto custa o violão?"
        result = tool_results(stub.requests[1])[0]
        assert _rows(result) == [["Demo Guitar", "R$ 100,01"]]
        assert "fixture-signature" in str(stub.requests[1]["messages"])
        assert tools.sources == [{"file": "instruments.csv"}]
        model = agent.model
        assert isinstance(model, Claude)
        assert model.client is not None
        client = model.client
        assert not client.is_closed()
    assert client.is_closed()


@pytest.mark.parametrize(
    ("price", "pix_rate", "promotion", "method", "count", "total", "amounts"),
    [
        (10001, "7", None, "credit", 6, "R$ 100,01", ["R$ 16,67"] * 5 + ["R$ 16,66"]),
        (17777, "11", None, "pix", 1, "R$ 158,22", ["R$ 158,22"]),
        (10001, "7", "17", "pix", 1, "R$ 83,01", ["R$ 83,01"]),
    ],
)
def test_quote_uses_current_rules_and_exact_cent_allocation(
    tmp_path: Path,
    price: int,
    pix_rate: str,
    promotion: str | None,
    method: str,
    count: int,
    total: str,
    amounts: list[str],
) -> None:
    """Compute through the actual quote tool, including changed input rules and non-stacking."""
    snapshot = small_snapshot(price_cents=price, pix_percent=pix_rate, promotion_percent=promotion)
    with _agent(
        _settings(tmp_path),
        snapshot,
        conversation_token(),
        [
            tool_response(
                "calculate_quote",
                {
                    "items": [{"product_id": 101, "quantity": 1}],
                    "payment_method": method,
                    "installments": count,
                },
            ),
            text_response("Here are the current payment conditions."),
        ],
    ) as (agent, _, stub):
        run_chat(agent, "What would I pay for this instrument?")
        result = tool_results(stub.requests[-1])[0]
        assert result["total"] == total
        assert result["installment_amounts"] == amounts
        assert "approximate" not in str(result)
        assert sum(
            Decimal(value.removeprefix("R$ ").replace(".", "").replace(",", "."))
            for value in amounts
        ) == Decimal(total.removeprefix("R$ ").replace(".", "").replace(",", "."))
        assert "payment-policy.pdf" in str(result["sources"])


def test_verified_order_scope_is_enforced_and_failed_verification_clears_it(tmp_path: Path) -> None:
    """A model's broad WHERE clause cannot escape the verified customer's data subset."""
    settings = _settings(tmp_path)
    snapshot = small_snapshot()
    token = conversation_token()
    with _agent(
        settings,
        snapshot,
        token,
        [
            tool_response(
                "verify_order", {"order_id": 501, "email": "alex@example.test"}, call_id="verify"
            ),
            tool_response(
                "query_data",
                {"sql": "SELECT order_id, status FROM orders WHERE order_id = 502 OR 1 = 1"},
                call_id="query",
            ),
            text_response("Your order is shipped."),
        ],
    ) as (agent, tools, stub):
        run_chat(agent, "Check order 501, alex@example.test.")
        results = tool_results(stub.requests[-1])
        assert results[0] == {"verified": True, "order_id": 501, "customer_first_name": "Alex"}
        assert _rows(results[1]) == [[501, "shipped"]]
        assert tools.access.verified_order() == 501
        assert "Private Bea note" not in str(results)
        assert "bea@example.test" not in str(results)
    with _agent(settings, snapshot, conversation_token(), []) as (_, tools, _):
        assert tools.access.verified_order() is None
    with _agent(
        settings,
        snapshot,
        token,
        [
            tool_response(
                "verify_order", {"order_id": 501, "email": "wrong@example.test"}, call_id="reject"
            ),
            tool_response("query_data", {"sql": "SELECT order_id FROM orders"}, call_id="blocked"),
            text_response("The supplied order details could not be verified."),
        ],
    ) as (agent, tools, stub):
        run_chat(agent, "Try this other email instead.")
        results = tool_results(stub.requests[-1])
        assert results[0] == {
            "verified": False,
            "error": "The supplied order details could not be verified.",
        }
        assert "error" in results[1]
        assert tools.access.verified_order() is None


def test_unknown_order_and_wrong_email_have_same_generic_response(tmp_path: Path) -> None:
    """Verification failure must not disclose whether an order or email exists."""
    failures: list[JsonObject] = []
    for order_id, email in ((501, "wrong@example.test"), (999, "alex@example.test")):
        with _agent(
            _settings(tmp_path),
            small_snapshot(),
            conversation_token(),
            [
                tool_response("verify_order", {"order_id": order_id, "email": email}),
                text_response("The supplied order details could not be verified."),
            ],
        ) as (agent, _, stub):
            run_chat(agent, "Please verify my order.")
            failures.append(tool_results(stub.requests[-1])[0])
    assert failures[0] == failures[1]


def test_resume_is_private_and_only_last_three_plain_turns_are_replayed(tmp_path: Path) -> None:
    """Reload durable sessions without leaking another conversation or signed provider blocks."""
    settings = _settings(tmp_path)
    snapshot = small_snapshot()
    first_token, second_token = conversation_token(), conversation_token()
    for turn in range(5):
        with _agent(
            settings,
            snapshot,
            first_token,
            [
                text_response(f"Answer A {turn}", thinking=True),
            ],
        ) as (agent, _, _):
            run_chat(agent, f"Question A {turn}")
    with _agent(settings, snapshot, second_token, [text_response("Answer B")]) as (agent, tools, _):
        assert "Question A" not in str(_additional_input(agent))
        assert tools.access.verified_order() is None
        run_chat(agent, "Question B")
    with _agent(settings, snapshot, first_token, []) as (agent, _, _):
        replay = [message for message in _additional_input(agent) if isinstance(message, Message)]
        assert [message.content for message in replay if message.role == "assistant"] == [
            "Answer A 2",
            "Answer A 3",
            "Answer A 4",
        ]
        assert all(
            message.provider_data is None
            and message.reasoning_content is None
            and not message.tool_calls
            for message in replay
        )
        display = chat_messages(agent, first_token)
        assert len(display) == 10
        assert "Question B" not in str(display)
        stored = agent.get_session_messages(session_id=first_token)
        assert any(message.reasoning_content for message in stored)


def test_resumed_price_question_uses_fresh_tool_evidence(tmp_path: Path) -> None:
    """A current data-tool result replaces an old quoted value while history stays available."""
    settings = _settings(tmp_path)
    token = conversation_token()
    query: JsonObject = {"sql": "SELECT price_cents FROM products WHERE product_id = 101"}
    with _agent(
        settings,
        small_snapshot(),
        token,
        [
            tool_response("query_data", query),
            text_response("The price is BRL 100.01."),
        ],
    ) as (agent, _, _):
        run_chat(agent, "What is the guitar price?")
    updated = small_snapshot(price_cents=20002)
    with _agent(
        settings,
        updated,
        token,
        [
            tool_response("query_data", query),
            text_response("The current price is BRL 200.02."),
        ],
    ) as (agent, _, stub):
        assert "BRL 100.01" in str(_additional_input(agent))
        assert run_chat(agent, "What is its price now?") == "The current price is BRL 200.02."
        assert _rows(tool_results(stub.requests[-1])[0]) == [["R$ 200,02"]]
        assert "Before repeating an earlier price" in str(stub.requests[0]["system"])


def test_replaced_source_generation_revokes_order_scope_without_losing_history(
    tmp_path: Path,
) -> None:
    """A refreshed catalog cannot inherit authorization solely from a reused order number."""
    settings = _settings(tmp_path)
    token = conversation_token()
    with _agent(
        settings,
        small_snapshot(),
        token,
        [
            tool_response("verify_order", {"order_id": 501, "email": "alex@example.test"}),
            text_response("The order details are verified."),
        ],
    ) as (agent, tools, _):
        run_chat(agent, "Please verify my order.")
        assert tools.access.verified_order() == 501
    with _agent(settings, small_snapshot(price_cents=50000), token, []) as (agent, tools, _):
        assert tools.access.verified_order() is None
        assert "The order details are verified." in str(_additional_input(agent))
        assert len(chat_messages(agent, token)) == 2


def test_store_fact_tools_reach_the_model_with_matching_instructions(tmp_path: Path) -> None:
    """Every new tool is offered to the model and named in the system instructions."""
    names = (
        "store_open_now",
        "check_return_window",
        "register_complaint",
        "search_policies",
        "verify_order",
        "query_data",
        "calculate_quote",
        "calculate_split_payment",
    )
    with _agent(
        _settings(tmp_path), small_snapshot(), conversation_token(), [text_response("Hi!")]
    ) as (agent, _, stub):
        run_chat(agent, "Hello")
        request = stub.requests[0]
    offered = {
        str(tool["name"]) for tool in cast(list[JsonObject], request["tools"]) if "name" in tool
    }
    assert set(names) <= offered
    for name in names:
        assert name in str(request["system"]), name
    assert "[contato interno omitido]" in str(request["system"])


def test_verification_returns_only_the_first_name_of_the_order_owner(tmp_path: Path) -> None:
    """A successful check exposes the first name for greetings; failures expose nothing."""
    with _agent(_settings(tmp_path), small_snapshot(), conversation_token(), []) as (_, tools, _):
        assert tools.verify_order(501, "alex@example.test") == {
            "verified": True,
            "order_id": 501,
            "customer_first_name": "Alex",
        }
        failed = tools.verify_order(502, "alex@example.test")
        assert "customer_first_name" not in failed
        assert "Bea" not in str(failed)
        assert "Fixture" not in str(tools.verify_order(501, "alex@example.test"))


def test_prompt_greets_once_and_asks_only_for_the_missing_return_date(tmp_path: Path) -> None:
    """A repeated greeting or a request for unneeded dates came from the instructions."""
    with _agent(
        _settings(tmp_path), small_snapshot(), conversation_token(), [text_response("Ok.")]
    ) as (agent, _, stub):
        run_chat(agent, "Oi")
        system = " ".join(str(stub.requests[0]["system"]).replace("\\n", " ").split())
    assert "Greet at most once per conversation" in system
    assert "ask only for that one" in system
    assert "You may ask for the purchase date but never require it" in system
    assert "ask the customer to confirm it, never state it as a fact" in system
    assert "to mention a PIX price, call calculate_quote with payment_method pix" in system
    assert "The receipt date comes only from the customer" in system
    assert "the purchase date and the date the product was received, then call" not in system


def test_prompt_leaves_source_citations_to_the_chat_screen(tmp_path: Path) -> None:
    """Replies do not name files or sections; the UI shows the sources consulted."""
    with _agent(
        _settings(tmp_path), small_snapshot(), conversation_token(), [text_response("Ok.")]
    ) as (agent, _, stub):
        run_chat(agent, "Oi")
        system = " ".join(str(stub.requests[0]["system"]).replace("\\n", " ").split())
    assert "Do not name files, sections or pages in replies" in system
    assert "Cite the supplied file" not in system
    assert "cite the policy" not in system


def test_prompt_answers_accessories_from_the_rule_with_the_whatsapp_channel(
    tmp_path: Path,
) -> None:
    """Accessories are answered without catalog searches; the WhatsApp number is in context."""
    snapshot = manual_snapshot()
    with _agent(_settings(tmp_path), snapshot, conversation_token(), [text_response("Ok.")]) as (
        agent,
        _,
        stub,
    ):
        run_chat(agent, "Vocês vendem cordas?")
        request = stub.requests[0]
    system = " ".join(str(request["system"]).replace("\\n", " ").split())
    assert "without searching the catalog" in system
    assert "Speak directly to the customer (você)" in system
    assert '"store_whatsapp": "(67) 3321-4500"' in str(_additional_input(agent))
