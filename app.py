"""Accessible native Streamlit chat; production never substitutes fake API responses."""

import logging
from collections.abc import Callable

import httpx2
import streamlit as st
from agno.agent import Agent

from emporio_agent.agent import (
    StoreTools,
    build_agent,
    chat_messages,
    close_agent,
    conversation_token,
    run_chat,
)
from emporio_agent.config import Settings
from emporio_agent.runtime import Snapshot, StoreRuntime

logger = logging.getLogger(__name__)


@st.cache_resource(show_spinner=False)
def _runtime(settings: Settings) -> StoreRuntime:
    """Share immutable indexes and a refresh lock, never user agents or identities."""
    return StoreRuntime(settings)


def _render_sources(sources: list[dict[str, object]]) -> None:
    """Present unique references supplied by the current tools."""
    with st.expander("Sources consulted"):
        seen: set[str] = set()
        for source in sources:
            label = str(source["file"])
            if "section" in source:
                label += f" · {source['section']} · pages {source['pages']}"
            if label not in seen:
                st.write(label)
                seen.add(label)


def _render_chat(agent: Agent, tools: StoreTools, token: str) -> None:
    """Display completed messages and handle one customer question."""
    history = chat_messages(agent, token)
    if not history:
        with st.chat_message("assistant"):
            st.write("Hello! How can I help you find your next instrument?")
    for message in history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    question = st.chat_input("Ask the store a question", max_chars=8000)
    if not question:
        return
    with st.chat_message("user"):
        st.write(question)
    try:
        with st.status("Preparing your answer…", expanded=False) as status:
            answer = run_chat(agent, question)
            status.update(label="Answer ready", state="complete")
        with st.chat_message("assistant"):
            st.markdown(answer)
            if tools.sources:
                _render_sources(tools.sources)
    except Exception as error:
        logger.warning("Agent request failed: %s", type(error).__name__)
        st.error("The request could not be completed. Please try again.")


def _render_page() -> str:
    """Initialize the page and conversation controls, returning its private identifier."""
    st.set_page_config(page_title="Empório da Música | Store support", page_icon="🎵")
    st.title("Empório da Música")
    st.caption(
        "Your music starts here. Ask about instruments, store policies or an existing order."
    )
    candidate = st.query_params.get("conversation")
    token = conversation_token(candidate)
    if candidate != token:
        st.query_params["conversation"] = token
    with st.sidebar:
        st.subheader("Your conversation")
        if st.button("New chat", use_container_width=True):
            st.query_params["conversation"] = conversation_token()
            st.rerun()
        st.caption("Bookmark this page to return. Keep the conversation link private.")
        st.caption("Order lookup requires an order number and the email used for that order.")
        st.caption("Information comes from the supplied store files. No purchases are made here.")
    return token


def _prepare_store(
    settings: Settings, runtime_factory: Callable[[Settings], StoreRuntime]
) -> Snapshot | None:
    """Load a valid source snapshot and show its availability or a preparation error."""
    try:
        with st.status("Checking store information…", expanded=False) as status:
            runtime = runtime_factory(settings)
            snapshot = runtime.refresh()
            status.update(label="Store information is ready", state="complete")
    except Exception as error:
        logger.warning("Source refresh failed: %s", type(error).__name__)
        if isinstance(error, ValueError):
            st.error(str(error))
        else:
            st.error("Store sources could not be loaded. Please check configuration and try again.")
        if st.button("Retry loading sources"):
            st.rerun()
        return None
    with st.sidebar.expander("Available information"):
        for name, available in snapshot.capabilities().items():
            st.write(f"{name.capitalize()}: {'available' if available else 'not supplied'}")
        st.caption(f"{len(snapshot.catalog.sources)} source files in the current snapshot.")
    return snapshot


def _serve_conversation(
    settings: Settings,
    snapshot: Snapshot,
    token: str,
    transport_factory: Callable[[], httpx2.BaseTransport] | None,
) -> None:
    """Own one agent's initialization, chat handling and resource cleanup."""
    try:
        agent, tools = build_agent(
            settings,
            snapshot,
            token,
            transport=transport_factory() if transport_factory is not None else None,
        )
    except Exception as error:
        logger.warning("Conversation initialization failed: %s", type(error).__name__)
        st.error("The conversation could not be opened. Please try again.")
        return
    try:
        _render_chat(agent, tools, token)
    finally:
        close_agent(agent)


def render(
    settings: Settings | None = None,
    *,
    runtime_factory: Callable[[Settings], StoreRuntime] = _runtime,
    transport_factory: Callable[[], httpx2.BaseTransport] | None = None,
) -> None:
    """Compose the page, store preparation and conversation lifecycle.

    Parameters:
        settings: Optional settings for isolated app tests.
        runtime_factory: Shared source runtime, independent of conversation state.
        transport_factory: Test-only HTTP transport; no production UI switch exists.
    """
    token = _render_page()
    try:
        active = settings or Settings.from_env()
    except ValueError as error:
        st.error(str(error))
        st.chat_input("Ask the store a question", disabled=True)
        return
    if not active.anthropic_key:
        st.info("Chat is not configured yet. Set ANTHROPIC_API_KEY in the server environment.")
        st.chat_input("Ask the store a question", disabled=True)
        return
    snapshot = _prepare_store(active, runtime_factory)
    if snapshot is None:
        st.chat_input("Ask the store a question", disabled=True)
        return
    _serve_conversation(active, snapshot, token, transport_factory)


if __name__ == "__main__":
    render()
