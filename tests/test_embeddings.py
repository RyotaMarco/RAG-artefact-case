"""Verify native Voyage HTTP, role-aware caches and bounded free-tier usage."""

import json
import re
import sqlite3
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from email.utils import format_datetime
from pathlib import Path
from threading import Barrier
from typing import cast

import httpx2
import pytest
from agent_fixtures import CharacterTokenizer
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace

from emporio_agent.embeddings import EmbeddingConfig, ModelTokenizer, VoyageEmbedder

type JsonObject = dict[str, object]
SERVER_PREFIXES = {
    "document": "Represent the document for retrieval: ",
    "query": "Represent the query for retrieving supporting documents: ",
}


@dataclass
class FakeClock:
    """Advance monotonic time without sleeping or consuming provider quota."""

    value: float = 0

    def __call__(self) -> float:
        """Return the controlled monotonic timestamp."""
        return self.value

    def advance(self, seconds: float) -> None:
        """Expire a rate-limit window without waiting in real time."""
        self.value += seconds


def inputs(body: JsonObject) -> list[str]:
    """Read a typed batch from the real HTTP payload."""
    values = body["input"]
    assert isinstance(values, list)
    assert all(isinstance(value, str) for value in cast(list[object], values))
    return cast(list[str], values)


def token_cost(body: JsonObject) -> int:
    """Include the server's documented role prompt in character-token fixture accounting."""
    prefix = SERVER_PREFIXES[str(body["input_type"])]
    return sum(len(prefix + text) for text in inputs(body))


def embedding_response(body: JsonObject) -> JsonObject:
    """Return explicit input indexes and matching token usage."""
    dimensions = body["output_dimension"]
    assert isinstance(dimensions, int)
    vector = [3.0, 4.0] + [0.0] * (dimensions - 2)
    return {
        "object": "list",
        "model": body["model"],
        "data": [
            {"index": index, "object": "embedding", "embedding": vector}
            for index, _ in enumerate(inputs(body))
        ],
        "usage": {"total_tokens": token_cost(body)},
    }


class VoyageStub:
    """Intercept HTTP only, retaining real cache, validation and quota behavior."""

    def __init__(self) -> None:
        """Begin with valid vectors and no provider attempts or injected failures."""
        self.requests: list[JsonObject] = []
        self.urls: list[str] = []
        self.status = 200
        self.headers: dict[str, str] = {}
        self.failure: str | None = None
        self.invalid_second = False
        self.response: Callable[[JsonObject], JsonObject] | None = None
        self.client = httpx2.Client(
            transport=httpx2.MockTransport(self._handle),
            headers={"Authorization": "Bearer fixture-key"},
            timeout=30,
        )

    def _handle(self, request: httpx2.Request) -> httpx2.Response:
        assert request.method == "POST" and request.url.path.endswith("/embeddings")
        assert request.headers["Authorization"] == "Bearer fixture-key"
        body = cast(JsonObject, json.loads(request.content))
        self.requests.append(body)
        self.urls.append(str(request.url))
        if self.failure == "timeout":
            raise httpx2.ReadTimeout("secret-provider-body", request=request)
        if self.failure == "network":
            raise httpx2.ConnectError("secret-provider-body", request=request)
        if self.status != 200:
            return httpx2.Response(
                self.status, json={"detail": "secret-provider-body"}, headers=self.headers
            )
        response = self.response(body) if self.response is not None else embedding_response(body)
        if self.invalid_second:
            data = cast(list[JsonObject], response["data"])
            data[1] = {"index": 1, "embedding": [0.0, 0.0]}
        # Raw JSON lets deliberately invalid NaN/Infinity values reach response validation.
        return httpx2.Response(
            200, content=json.dumps(response), headers={"Content-Type": "application/json"}
        )


def make_embedder(
    path: Path,
    stub: VoyageStub,
    *,
    clock: FakeClock | None = None,
    config: EmbeddingConfig | None = None,
    max_batch_tokens: int = 8000,
    requests_per_minute: int = 3,
    tokens_per_minute: int = 10000,
) -> VoyageEmbedder:
    """Use deterministic time and a tokenizer requiring no download."""
    return VoyageEmbedder(
        path,
        client=stub.client,
        tokenizer=CharacterTokenizer(),
        config=config or EmbeddingConfig(dimensions=2),
        clock=clock or FakeClock(),
        max_batch_tokens=max_batch_tokens,
        requests_per_minute=requests_per_minute,
        tokens_per_minute=tokens_per_minute,
    )


def cache_count(path: Path) -> int:
    """Inspect committed rows without relying on adapter internals."""
    with closing(sqlite3.connect(path)) as connection:
        row = connection.execute("SELECT COUNT(*) FROM vectors").fetchone()
        assert row is not None
        return int(row[0])


def test_native_payload_and_role_cache_survive_restart(tmp_path: Path) -> None:
    """Cache query/document roles separately without sending local instruction prefixes."""
    path, stub = tmp_path / "vectors.sqlite3", VoyageStub()
    with stub.client:
        embedder = make_embedder(path, stub)
        assert embedder.embed_documents(["Same text", "Same text"]) == [(0.6, 0.8)] * 2
        assert embedder.embed_query("Same text") == (0.6, 0.8)
        assert len(stub.requests) == 2
        assert stub.urls == ["https://ai.mongodb.com/v1/embeddings"] * 2
        assert stub.requests[0] == {
            "input": ["Same text"],
            "model": "voyage-4-lite",
            "input_type": "document",
            "output_dimension": 2,
            "output_dtype": "float",
            "truncation": False,
        }
        assert stub.requests[1] == {**stub.requests[0], "input_type": "query"}
        restored = make_embedder(path, stub, requests_per_minute=1)
        stub.failure = "network"
        assert restored.embed_documents(["Same text"]) == [(0.6, 0.8)]
        assert restored.embed_query("Same text") == (0.6, 0.8)
        assert len(stub.requests) == 2 and cache_count(path) == 2


@pytest.mark.parametrize(
    "changed",
    [
        EmbeddingConfig(model="voyage-4", dimensions=2),
        EmbeddingConfig(base_url="https://api.voyageai.com/v1", dimensions=2),
        EmbeddingConfig(dimensions=3),
    ],
)
def test_changed_embedding_identity_does_not_reuse_incompatible_vectors(
    tmp_path: Path, changed: EmbeddingConfig
) -> None:
    """Version endpoint, model and dimensions independently in persistent cache keys."""
    path, stub = tmp_path / "vectors.sqlite3", VoyageStub()
    with stub.client:
        original = make_embedder(path, stub)
        original.embed_documents(["Same text"])
        replacement = make_embedder(path, stub, config=changed)
        assert replacement.fingerprint != original.fingerprint
        assert len(replacement.embed_documents(["Same text"])[0]) == changed.dimensions
        assert len(stub.requests) == 2
        assert stub.urls[-1] == changed.base_url + "/embeddings"


def test_response_indexes_restore_order_and_cached_associations(tmp_path: Path) -> None:
    """Out-of-order responses cannot attach one passage's vector to another passage."""
    stub = VoyageStub()

    def reversed_response(body: JsonObject) -> JsonObject:
        return {
            **embedding_response(body),
            "data": [
                {"index": 1, "embedding": [0.0, 5.0]},
                {"index": 0, "embedding": [3.0, 4.0]},
            ],
        }

    stub.response = reversed_response
    with stub.client:
        embedder = make_embedder(tmp_path / "vectors.sqlite3", stub)
        assert embedder.embed_documents(["First", "Second"]) == [(0.6, 0.8), (0.0, 1.0)]
        assert embedder.embed_documents(["Second", "First"]) == [(0.0, 1.0), (0.6, 0.8)]
        assert len(stub.requests) == 1


@pytest.mark.parametrize("indexes", [[0, 0], [0], [0, 2], [-1, 1], [True, 1]])
def test_invalid_response_indexes_leave_no_cached_rows(
    tmp_path: Path, indexes: list[int | bool]
) -> None:
    """Reject incomplete, duplicate, out-of-range and noninteger input indexes."""
    stub = VoyageStub()
    stub.response = lambda body: {
        **embedding_response(body),
        "data": [{"index": index, "embedding": [3.0, 4.0]} for index in indexes],
    }
    path = tmp_path / "vectors.sqlite3"
    with stub.client:
        with pytest.raises(ValueError):
            make_embedder(path, stub).embed_documents(["First", "Second"])
    assert cache_count(path) == 0


@pytest.mark.parametrize("vector", [[0.0, 0.0], [float("nan"), 1.0], [float("inf"), 1.0], [1.0]])
def test_invalid_vectors_do_not_enter_cache(tmp_path: Path, vector: list[float]) -> None:
    """Reject dimension, finite-value and nonzero-norm errors at the HTTP boundary."""
    stub = VoyageStub()
    stub.response = lambda body: {
        **embedding_response(body),
        "data": [{"index": 0, "embedding": vector}],
    }
    path = tmp_path / "vectors.sqlite3"
    with stub.client:
        with pytest.raises(ValueError):
            make_embedder(path, stub).embed_documents(["Policy"])
    assert cache_count(path) == 0


def test_batches_obey_token_budget_and_item_cap(tmp_path: Path) -> None:
    """Enforce token-aware batch boundaries and the existing 32-input maximum."""
    stub = VoyageStub()
    with stub.client:
        limited = make_embedder(
            tmp_path / "small.sqlite3", stub, max_batch_tokens=100, requests_per_minute=100
        )
        assert len(limited.embed_documents([str(index) + "x" * 30 for index in range(5)])) == 5
        assert len(stub.requests) == 5
        assert all(token_cost(body) <= 100 for body in stub.requests)
        stub.requests.clear()
        capped = make_embedder(tmp_path / "many.sqlite3", stub, requests_per_minute=100)
        assert len(capped.embed_documents([f"Policy {index}" for index in range(33)])) == 33
        assert [len(inputs(body)) for body in stub.requests] == [32, 1]


def test_token_quota_keeps_completed_batches_and_resumes_missing_vectors(tmp_path: Path) -> None:
    """A request budget above TPM still splits safely and preserves progress for retry."""
    path, clock, stub = tmp_path / "vectors.sqlite3", FakeClock(), VoyageStub()
    texts = ["a" * 50, "b" * 50]
    with stub.client:
        embedder = make_embedder(
            path, stub, clock=clock, max_batch_tokens=8000, tokens_per_minute=150
        )
        with pytest.raises(ValueError, match="Retry in"):
            embedder.embed_documents(texts)
        assert len(stub.requests) == 1 and len(inputs(stub.requests[0])) == 1
        assert cache_count(path) == 1
        clock.advance(60.01)
        assert embedder.embed_documents(texts) == [(0.6, 0.8)] * 2
        assert inputs(stub.requests[-1]) == [texts[1]] and cache_count(path) == 2


def test_queries_and_documents_share_request_quota_without_sleeping(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Cache hits cost no quota and a fourth uncached request fails quickly until expiry."""

    def reject_sleep(_seconds: float) -> None:
        raise AssertionError("The limiter must not block a request thread.")

    monkeypatch.setattr("time.sleep", reject_sleep)
    clock, stub = FakeClock(), VoyageStub()
    with stub.client:
        embedder = make_embedder(tmp_path / "vectors.sqlite3", stub, clock=clock)
        embedder.embed_documents(["First"])
        embedder.embed_query("Question")
        embedder.embed_documents(["Second"])
        assert embedder.embed_query("Question") == (0.6, 0.8)
        with pytest.raises(ValueError, match="Retry in"):
            embedder.embed_query("Another question")
        assert len(stub.requests) == 3 and clock.value == 0
        clock.advance(60.01)
        assert embedder.embed_query("Another question") == (0.6, 0.8)


def test_concurrent_queries_cannot_overbook_the_shared_request_window(tmp_path: Path) -> None:
    """Several chat threads must share one atomic request reservation."""
    stub, ready = VoyageStub(), Barrier(4)
    with stub.client:
        embedder = make_embedder(tmp_path / "vectors.sqlite3", stub)

        def search(index: int) -> bool:
            ready.wait(timeout=5)
            try:
                embedder.embed_query(f"Question {index}")
            except ValueError as error:
                assert "Retry in" in str(error)
                return False
            return True

        with ThreadPoolExecutor(max_workers=4) as pool:
            results = list(pool.map(search, range(4)))
    assert sum(results) == 3 and len(stub.requests) == 3


def test_reported_token_usage_can_raise_the_next_request_reservation(tmp_path: Path) -> None:
    """Provider usage above the local estimate must count against the shared token quota."""
    stub = VoyageStub()
    stub.response = lambda body: {**embedding_response(body), "usage": {"total_tokens": 1000}}
    with stub.client:
        embedder = make_embedder(tmp_path / "vectors.sqlite3", stub, tokens_per_minute=1500)
        embedder.embed_query("First")
        with pytest.raises(ValueError, match="Retry in"):
            embedder.embed_query("x" * 550)
    assert len(stub.requests) == 1


@pytest.mark.parametrize(
    "usage", [None, {"total_tokens": True}, {"total_tokens": -1}, {"total_tokens": "12"}]
)
def test_invalid_usage_does_not_publish_an_unaccounted_batch(tmp_path: Path, usage: object) -> None:
    """A malformed token meter must leave the batch uncached for a controlled retry."""
    path, stub = tmp_path / "vectors.sqlite3", VoyageStub()
    stub.response = lambda body: {**embedding_response(body), "usage": usage}
    with stub.client:
        with pytest.raises(ValueError, match="usage"):
            make_embedder(path, stub).embed_documents(["Policy"])
    assert cache_count(path) == 0


def test_oversized_input_requires_shortening_not_an_endless_cooldown(tmp_path: Path) -> None:
    """An input unable to fit any batch must not suggest waiting for a quota reset."""
    stub = VoyageStub()
    with stub.client:
        with pytest.raises(ValueError) as captured:
            make_embedder(tmp_path / "vectors.sqlite3", stub, max_batch_tokens=100).embed_documents(
                ["x" * 100]
            )
    assert "Retry in" not in str(captured.value) and not stub.requests


@pytest.mark.parametrize("status", [400, 401, 402, 403, 500])
def test_http_errors_hide_provider_bodies_without_silent_retry(tmp_path: Path, status: int) -> None:
    """Leave failed embeddings uncached and never publish raw provider error content."""
    path, stub = tmp_path / "vectors.sqlite3", VoyageStub()
    stub.status = status
    with stub.client:
        with pytest.raises(ValueError) as captured:
            make_embedder(path, stub).embed_query("Payment methods")
    assert "secret-provider-body" not in str(captured.value)
    assert len(stub.requests) == 1 and cache_count(path) == 0


@pytest.mark.parametrize("failure", ["timeout", "network"])
def test_transport_failures_remain_retryable(tmp_path: Path, failure: str) -> None:
    """Native transport exceptions become safe errors without poisoning persistent caches."""
    path, stub = tmp_path / "vectors.sqlite3", VoyageStub()
    stub.failure = failure
    with stub.client:
        embedder = make_embedder(path, stub)
        with pytest.raises(ValueError) as captured:
            embedder.embed_documents(["Policy"])
        assert "secret-provider-body" not in str(captured.value)
        assert len(stub.requests) == 1 and cache_count(path) == 0
        stub.failure = None
        assert embedder.embed_documents(["Policy"]) == [(0.6, 0.8)]
        assert len(stub.requests) == 2


@pytest.mark.parametrize("header, delay", [("12", 12), (None, 60), ("not-a-duration", 60)])
def test_http_429_sets_shared_cooldown_without_repeating_requests(
    tmp_path: Path, header: str | None, delay: int
) -> None:
    """Honor numeric Retry-After and a finite fallback when it is missing or malformed."""
    clock, stub = FakeClock(), VoyageStub()
    stub.status = 429
    stub.headers = {"Retry-After": header} if header is not None else {}
    with stub.client:
        embedder = make_embedder(tmp_path / "vectors.sqlite3", stub, clock=clock)
        with pytest.raises(ValueError, match="Retry in"):
            embedder.embed_documents(["Policy"])
        stub.status = 200
        clock.advance(delay - 1)
        with pytest.raises(ValueError, match="Retry in"):
            embedder.embed_query("Question")
        assert len(stub.requests) == 1
        clock.advance(1.01)
        assert embedder.embed_documents(["Policy"]) == [(0.6, 0.8)]


def test_http_date_retry_after_becomes_monotonic_cooldown(tmp_path: Path) -> None:
    """An absolute header cannot cause an immediate repeat or an unbounded sleep."""
    clock, stub = FakeClock(), VoyageStub()
    stub.status = 429
    stub.headers = {
        "Retry-After": format_datetime(datetime.now(UTC) + timedelta(seconds=90), usegmt=True)
    }
    with stub.client:
        embedder = make_embedder(tmp_path / "vectors.sqlite3", stub, clock=clock)
        with pytest.raises(ValueError, match="Retry in") as captured:
            embedder.embed_documents(["Policy"])
        delay = re.search(r"Retry in (\d+)", str(captured.value))
        assert delay is not None and 75 <= int(delay.group(1)) <= 90
        stub.status = 200
        clock.advance(20)
        with pytest.raises(ValueError, match="Retry in"):
            embedder.embed_documents(["Policy"])
        assert len(stub.requests) == 1
        clock.advance(75)
        assert embedder.embed_documents(["Policy"]) == [(0.6, 0.8)]


def test_invalid_later_batch_retains_prior_committed_progress(tmp_path: Path) -> None:
    """Rollback only a failed batch and request only its missing passages on retry."""
    path, stub = tmp_path / "vectors.sqlite3", VoyageStub()

    def fail_second_batch(body: JsonObject) -> JsonObject:
        result = embedding_response(body)
        if len(stub.requests) == 2:
            result["data"] = [{"index": 0, "embedding": [0.0, 0.0]}]
        return result

    stub.response = fail_second_batch
    texts = ["a" * 30, "b" * 30]
    with stub.client:
        embedder = make_embedder(path, stub, max_batch_tokens=100, requests_per_minute=100)
        with pytest.raises(ValueError):
            embedder.embed_documents(texts)
        assert cache_count(path) == 1
        stub.response = None
        assert embedder.embed_documents(texts) == [(0.6, 0.8)] * 2
        assert len(stub.requests) == 3 and inputs(stub.requests[-1]) == [texts[1]]


def test_empty_inputs_do_not_contact_provider(tmp_path: Path) -> None:
    """Reject blank text and allow empty document batches without an HTTP call."""
    stub = VoyageStub()
    with stub.client:
        embedder = make_embedder(tmp_path / "vectors.sqlite3", stub)
        assert embedder.embed_documents([]) == []
        with pytest.raises(ValueError, match="empty"):
            embedder.embed_documents([" "])
        with pytest.raises(ValueError, match="empty"):
            embedder.embed_query(" ")
    assert not stub.requests


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
