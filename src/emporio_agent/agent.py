"""One Agno agent with source-bound tools and private, resumable conversations."""

import json
from datetime import datetime
from zoneinfo import ZoneInfo

import httpx2
from agno.agent import Agent
from agno.db.sqlite import SqliteDb
from agno.models.anthropic import Claude
from agno.models.message import Message
from agno.run.base import RunStatus
from anthropic import Anthropic

from emporio_agent.access import ConversationAccess, conversation_token
from emporio_agent.config import Settings
from emporio_agent.prompt import INSTRUCTIONS
from emporio_agent.runtime import Snapshot
from emporio_agent.tools import StoreTools


def _recent_history(agent: Agent, token: str) -> list[Message]:
    """Reconstruct text-only history, without replaying signed provider/tool blocks."""
    return [
        Message(role=message["role"], content=message["content"], from_history=True)
        for message in chat_messages(agent, token, last_n_runs=3)
    ]


def build_agent(
    settings: Settings,
    snapshot: Snapshot,
    token: str,
    *,
    transport: httpx2.BaseTransport | None = None,
) -> tuple[Agent, StoreTools]:
    """Build an isolated native Agno/Claude loop with injectable HTTP for tests.

    Raises:
        ValueError: The model key or conversation identifier is invalid.
    """
    if not settings.anthropic_key:
        raise ValueError("ANTHROPIC_API_KEY is required to start the chat agent.")
    if conversation_token(token) != token:
        raise ValueError("The conversation identifier is invalid.")
    database = settings.runtime_dir / "sessions.sqlite3"
    access = ConversationAccess(database, token, snapshot.generation_id)
    tools = StoreTools(snapshot, access, settings.store_timezone)
    http_client = httpx2.Client(transport=transport, timeout=settings.claude_timeout_seconds)
    model = Claude(
        id=settings.model,
        api_key=settings.anthropic_key,
        client=Anthropic(api_key=settings.anthropic_key, http_client=http_client, max_retries=1),
        max_tokens=settings.max_output_tokens,
        thinking={"type": "adaptive"},
        output_config={"effort": "low"},
        cache_system_prompt=True,
        cache_tools=True,
        retries=0,
    )
    agent = Agent(
        id="emporio-store-assistant",
        name="Store support",
        model=model,
        db=SqliteDb(db_file=str(database)),
        user_id=access.identity,
        session_id=token,
        add_history_to_context=False,
        tools=[
            tools.store_open_now,
            tools.check_return_window,
            tools.search_policies,
            tools.verify_order,
            tools.register_complaint,
            tools.query_data,
            tools.calculate_quote,
            tools.calculate_split_payment,
        ],
        instructions=INSTRUCTIONS,
        tool_call_limit=settings.max_tool_calls,
        markdown=True,
        telemetry=False,
    )
    context = {
        "store_local_time": datetime.now(ZoneInfo(settings.store_timezone)).isoformat(),
        "capabilities": snapshot.capabilities(),
        "schema": snapshot.catalog.describe(),
        "verified_order_id": access.verified_order(),
        "source_generation": snapshot.generation_id,
        "source_note": "These are file snapshots, not live carrier or inventory integrations.",
        "store_whatsapp": snapshot.evidence.whatsapp,
    }
    agent.additional_input = [
        *_recent_history(agent, token),
        Message(role="user", content=json.dumps(context, ensure_ascii=False), from_history=True),
    ]
    return agent, tools


def chat_messages(
    agent: Agent, token: str, *, last_n_runs: int | None = None
) -> list[dict[str, str]]:
    """Read completed text without provider reasoning or tool results.

    Parameters:
        agent: The current conversation's agent and session storage.
        token: The conversation identifier.
        last_n_runs: Optional completed-turn limit; the UI reads all turns by default.

    Returns:
        User and final assistant text, shared by display and model-history reconstruction.
    """
    if agent.get_session(session_id=token) is None:
        return []
    messages: list[dict[str, str]] = []
    for message in agent.get_session_messages(session_id=token, last_n_runs=last_n_runs):
        if (
            message.role in {"user", "assistant"}
            and isinstance(message.content, str)
            and not message.from_history
            and not message.tool_calls
        ):
            messages.append({"role": message.role, "content": message.content})
    return messages


def run_chat(agent: Agent, question: str) -> str:
    """Run the actual agent synchronously and require a final text response."""
    if not question.strip() or len(question) > 8000:
        raise ValueError("Please enter a question containing between 1 and 8000 characters.")
    # Agno's overload includes unparameterized containers; this call has a concrete string input.
    result = agent.run(question, stream=False)  # pyright: ignore[reportUnknownMemberType]
    if result.status != RunStatus.completed:
        raise ValueError("The model request did not complete. Please try again.")
    content: object = result.content
    if not isinstance(content, str) or not content.strip():
        raise ValueError("The model returned no final text. Please try again.")
    return content


def close_agent(agent: Agent) -> None:
    """Release per-render HTTP and session handles without closing shared source indexes."""
    model = agent.model
    if isinstance(model, Claude) and model.client is not None:
        model.client.close()
    if isinstance(agent.db, SqliteDb):
        agent.db.close()
