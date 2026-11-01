"""Verify hosted embedding caching, provider failures and response validation."""

from collections.abc import Callable
from pathlib import Path
from typing import cast

import httpx2
import huggingface_hub.utils._http as hf_http
import pytest
from huggingface_hub import InferenceClient, set_client_factory
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace

from emporio_agent.embeddings import EmbeddingConfig, HFEmbedder, ModelTokenizer


@pytest.mark.parametrize(
    "status, expected", [(401, "HF_TOKEN"), (402, "credits"), (429, "limit"), (None, "timed out")]
)
def test_provider_errors_are_actionable_without_exposing_response_secrets(
    tmp_path: Path, status: int | None, expected: str
) -> None:
    """Translate actual HF SDK failures while all native HTTP stays inside a mock transport."""
    attempts: list[httpx2.Request] = []

    def respond(request: httpx2.Request) -> httpx2.Response:
        if request.method == "GET" and request.url.path == "/api/agent-harnesses":
            return httpx2.Response(503)
        if request.method == "GET" and request.url.path.startswith("/api/models/"):
            return httpx2.Response(
                200,
                json={
                    "id": "Qwen/Qwen3-Embedding-0.6B",
                    "inferenceProviderMapping": {
                        "deepinfra": {
                            "providerId": "Qwen/Qwen3-Embedding-0.6B",
                            "status": "live",
                            "task": "feature-extraction",
                        }
                    },
                },
            )
        assert request.method == "POST"
        assert request.url.path == "/deepinfra/v1/openai/embeddings"
        attempts.append(request)
        if status is None:
            raise httpx2.ReadTimeout("secret-provider-body", request=request)
        return httpx2.Response(status, json={"error": "secret-provider-body"})

    previous_factory = cast(Callable[[], httpx2.Client], getattr(hf_http, "_GLOBAL_CLIENT_FACTORY"))
    transport = httpx2.MockTransport(respond)
    native_client = httpx2.Client(transport=transport)
    set_client_factory(lambda: native_client)
    try:
        embedder = HFEmbedder(
            tmp_path / "vectors.sqlite3",
            client=InferenceClient(provider="deepinfra", token="hf_fixture", timeout=30),
        )
        with pytest.raises(ValueError, match=expected) as captured:
            embedder.embed_query("payment methods")
        assert "secret-provider-body" not in str(captured.value)
        assert len(attempts) == 1
    finally:
        set_client_factory(previous_factory)
    assert native_client.is_closed


class FakeClient:
    """Return a scripted vector response and record paid-attempt inputs."""

    def __init__(self, vector: list[float] | None = None) -> None:
        self.calls: list[list[str]] = []
        self.models: list[str] = []
        self.vector = vector if vector is not None else [3.0, 4.0]
        self.fail = False

    def feature_extraction(
        self, text: str | list[str], *, model: str, normalize: bool, truncate: bool
    ) -> object:
        """Return raw vectors as a provider could."""
        assert model
        assert normalize
        assert not truncate
        batch = [text] if isinstance(text, str) else text
        self.calls.append(batch)
        self.models.append(model)
        if self.fail:
            raise TimeoutError("Embedding service timed out")
        return [self.vector for _ in batch]


def test_cache_is_persistent_and_queries_use_instruction(tmp_path: Path) -> None:
    """Identical passages must not incur another network call after a restart."""
    client = FakeClient()
    path = tmp_path / "vectors.sqlite3"
    config = EmbeddingConfig(dimensions=2)
    embedder = HFEmbedder(path, client=client, config=config)
    assert embedder.embed_documents(["policy", "policy"]) == [(0.6, 0.8), (0.6, 0.8)]
    assert len(client.calls) == 1
    reopened = HFEmbedder(path, client=client, config=config)
    assert reopened.embed_documents(["policy"]) == [(0.6, 0.8)]
    assert len(client.calls) == 1
    reopened.embed_query("PIX discount")
    assert client.calls[-1][0].startswith("Instruct:")
    assert "Query: PIX discount" in client.calls[-1][0]
    assert len(client.calls) == 2


@pytest.mark.parametrize("query_prefix, document_prefix", [("query: {text}", "passage:"), ("", "")])
def test_configured_prefixes_and_model_reach_provider_without_template_expansion(
    tmp_path: Path, query_prefix: str, document_prefix: str
) -> None:
    """Allow model-specific input conventions, including encoders requiring no prefixes."""
    client = FakeClient()
    config = EmbeddingConfig(
        model="example/encoder",
        provider="example-provider",
        dimensions=2,
        query_prefix=query_prefix,
        document_prefix=document_prefix,
    )
    embedder = HFEmbedder(tmp_path / "vectors.sqlite3", client=client, config=config)
    embedder.embed_documents(["Document text"])
    embedder.embed_query("Question text")
    assert client.calls == [
        [f"{document_prefix} Document text" if document_prefix else "Document text"],
        [f"{query_prefix} Question text" if query_prefix else "Question text"],
    ]
    assert client.models == ["example/encoder", "example/encoder"]


@pytest.mark.parametrize(
    "changed",
    [
        EmbeddingConfig(model="example/encoder", dimensions=2),
        EmbeddingConfig(provider="example-provider", dimensions=2),
        EmbeddingConfig(dimensions=3),
        EmbeddingConfig(query_prefix="query:", dimensions=2),
        EmbeddingConfig(document_prefix="passage:", dimensions=2),
    ],
)
def test_embedding_configuration_changes_cannot_reuse_old_cached_vectors(
    tmp_path: Path, changed: EmbeddingConfig
) -> None:
    """Version every input convention and vector-space setting in persistent cache keys."""
    path = tmp_path / "vectors.sqlite3"
    original = EmbeddingConfig(dimensions=2)
    HFEmbedder(path, client=FakeClient(), config=original).embed_documents(["Same text"])
    client = FakeClient([1.0] * changed.dimensions)
    replacement = HFEmbedder(path, client=client, config=changed)
    replacement.embed_documents(["Same text"])
    assert replacement.fingerprint != original.fingerprint
    assert len(client.calls) == 1
    replacement.embed_documents(["Same text"])
    assert len(client.calls) == 1


def test_nonempty_prefix_cannot_hide_an_empty_embedding_input(tmp_path: Path) -> None:
    """Reject empty customer or document text before adding configured model prefixes."""
    client = FakeClient()
    embedder = HFEmbedder(
        tmp_path / "vectors.sqlite3",
        client=client,
        config=EmbeddingConfig(dimensions=2, query_prefix="query:", document_prefix="passage:"),
    )
    with pytest.raises(ValueError, match="empty"):
        embedder.embed_documents([" "])
    with pytest.raises(ValueError, match="empty"):
        embedder.embed_query(" ")
    assert not client.calls


def test_provider_failure_is_not_cached_or_silently_retried(tmp_path: Path) -> None:
    """A failed request must return its error and remain retryable by the caller."""
    client = FakeClient()
    client.fail = True
    embedder = HFEmbedder(
        tmp_path / "cache.sqlite3",
        client=client,
        config=EmbeddingConfig(dimensions=2),
    )
    with pytest.raises(TimeoutError, match="timed out"):
        embedder.embed_documents(["policy"])
    assert len(client.calls) == 1
    client.fail = False
    assert embedder.embed_documents(["policy"]) == [(0.6, 0.8)]
    assert len(client.calls) == 2


@pytest.mark.parametrize("vector", [[0.0, 0.0], [float("nan"), 1.0], [1.0]])
def test_invalid_vectors_are_not_accepted(tmp_path: Path, vector: list[float]) -> None:
    """Reject shape, zero-vector and nonfinite failures at the provider boundary."""
    embedder = HFEmbedder(
        tmp_path / "cache.sqlite3",
        client=FakeClient(vector),
        config=EmbeddingConfig(dimensions=2),
    )
    with pytest.raises(ValueError):
        embedder.embed_documents(["policy"])


def test_tokenizer_splits_without_truncating() -> None:
    """Disable source tokenizer truncation and preserve the final input words."""
    native = Tokenizer(WordLevel({"[UNK]": 0, "policy": 1, "tail": 2}, unk_token="[UNK]"))
    native.pre_tokenizer = Whitespace()
    native.enable_truncation(max_length=8)  # pyright: ignore[reportUnknownMemberType]
    tokenizer = ModelTokenizer(native)
    text = "policy " * 1800 + "tail"
    assert tokenizer.count(text) == 1801
    spans = tokenizer.split(text, 600, 80)
    assert all(tokenizer.count(span) <= 600 for span in spans)
    assert spans[-1].endswith("tail")
