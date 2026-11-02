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
_CAPABILITY_LABELS = {
    "catalog": "Catálogo",
    "promotions": "Promoções",
    "orders": "Pedidos",
    "policies": "Políticas",
}


@st.cache_resource(show_spinner=False)
def _runtime(settings: Settings) -> StoreRuntime:
    """Share immutable indexes and a refresh lock, never user agents or identities."""
    return StoreRuntime(settings)


def _render_sources(sources: list[dict[str, object]]) -> None:
    """Present unique references supplied by the current tools."""
    with st.expander("Fontes consultadas"):
        seen: set[str] = set()
        for source in sources:
            label = str(source["file"])
            if "section" in source:
                label += f" · {source['section']} · páginas {source['pages']}"
            if label not in seen:
                st.write(label)
                seen.add(label)


def _render_chat(agent: Agent, tools: StoreTools, token: str) -> None:
    """Display completed messages and handle one customer question."""
    history = chat_messages(agent, token)
    if not history:
        with st.chat_message("assistant"):
            st.write(
                "Oi! Sou o assistente virtual da Empório da Música. Qual é o seu nome? Em que posso te ajudar hoje?"
            )
    for message in history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    question = st.chat_input("Digite sua mensagem para a loja", max_chars=8000)
    if not question:
        return
    with st.chat_message("user"):
        st.write(question)
    try:
        with st.status("Preparando sua resposta…", expanded=False) as status:
            answer = run_chat(agent, question)
            status.update(label="Resposta pronta", state="complete")
        with st.chat_message("assistant"):
            st.markdown(answer)
            if tools.sources:
                _render_sources(tools.sources)
    except Exception as error:
        logger.warning("Agent request failed: %s", type(error).__name__)
        st.error("Não foi possível responder agora. Tente novamente.")


def _render_page() -> str:
    """Initialize the page and conversation controls, returning its private identifier."""
    st.set_page_config(page_title="Empório da Música | Atendimento", page_icon="🎵")
    st.title("Empório da Música")
    st.caption(
        "Sua música começa aqui. Pergunte sobre instrumentos, políticas da loja ou um pedido."
    )
    candidate = st.query_params.get("conversation")
    token = conversation_token(candidate)
    if candidate != token:
        st.query_params["conversation"] = token
    with st.sidebar:
        st.subheader("Sua conversa")
        if st.button("Nova conversa", use_container_width=True):
            st.query_params["conversation"] = conversation_token()
            st.rerun()
        st.caption(
            "Salve esta página nos favoritos para voltar. Não compartilhe o link da conversa."
        )
        st.caption("Para consultar um pedido, informe o número dele e o e-mail usado na compra.")
        st.caption("As informações vêm dos arquivos da loja. Nenhuma compra é feita aqui.")
    return token


def _prepare_store(
    settings: Settings, runtime_factory: Callable[[Settings], StoreRuntime]
) -> Snapshot | None:
    """Load a valid source snapshot and show its availability or a preparation error."""
    try:
        with st.status("Carregando as informações da loja…", expanded=False) as status:
            runtime = runtime_factory(settings)
            snapshot = runtime.refresh()
            status.update(label="Informações da loja prontas", state="complete")
    except Exception as error:
        logger.warning("Source refresh failed: %s", type(error).__name__)
        if isinstance(error, ValueError):
            st.error(str(error))
        else:
            st.error("Não foi possível carregar as informações da loja. Verifique a configuração.")
        if st.button("Tentar carregar novamente"):
            st.rerun()
        return None
    with st.sidebar.expander("Informações disponíveis"):
        for name, available in snapshot.capabilities().items():
            status = "disponível" if available else "não fornecido"
            st.write(f"{_CAPABILITY_LABELS.get(name, name)}: {status}")
        st.caption(f"{len(snapshot.catalog.sources)} arquivos de origem carregados.")
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
        st.error("Não foi possível abrir a conversa. Tente novamente.")
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
        st.chat_input("Digite sua mensagem para a loja", disabled=True)
        return
    if not active.anthropic_key:
        st.info(
            "O chat ainda não está configurado. Defina ANTHROPIC_API_KEY no ambiente do servidor."
        )
        st.chat_input("Digite sua mensagem para a loja", disabled=True)
        return
    snapshot = _prepare_store(active, runtime_factory)
    if snapshot is None:
        st.chat_input("Digite sua mensagem para a loja", disabled=True)
        return
    _serve_conversation(active, snapshot, token, transport_factory)


if __name__ == "__main__":
    render()
