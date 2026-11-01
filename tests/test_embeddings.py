"""Verify hosted embedding caching, provider failures and response validation."""

from pathlib import Path

import httpx
import pytest
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace

from emporio_agent.embeddings import EmbeddingConfig, HFEmbedder, ModelTokenizer


@pytest.mark.parametrize("status, expected", [(401, "HF_TOKEN"), (402, "credits"), (429, "limit")])
def test_provider_errors_are_actionable_without_exposing_response_secrets(
    tmp_path: Path, status: int, expected: str
) -> None:
    """Report a provider failure category without displaying its raw response body."""

    class RejectingClient(FakeClient):
        def feature_extraction(
            self, text: str | list[str], *, model: str, normalize: bool, truncate: bool
        ) -> object:
            """Fail only the controlled external provider boundary."""
            response = httpx.Response(status, request=httpx.Request("POST", "https://example.test"))
            raise httpx.HTTPStatusError(
                "secret-provider-body", request=response.request, response=response
            )

    embedder = HFEmbedder(tmp_path / "vectors.sqlite3", client=RejectingClient())
    with pytest.raises(ValueError, match=expected) as captured:
        embedder.embed_query("payment methods")
    assert "secret-provider-body" not in str(captured.value)


class FakeClient:
    """Return a scripted vector response and record paid-attempt inputs."""

    def __init__(self, vector: list[float] | None = None) -> None:
        self.calls: list[list[str]] = []
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
