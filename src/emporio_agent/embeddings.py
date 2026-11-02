"""Hosted embeddings with a persistent, validated vector cache."""

import json
import math
import sqlite3
from collections import deque
from collections.abc import Callable, Iterator, Sequence
from contextlib import closing
from dataclasses import dataclass
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from hashlib import sha256
from pathlib import Path
from threading import RLock
from time import monotonic
from typing import Literal, Protocol, cast

import httpx2
import numpy as np
from tokenizers import Tokenizer

DEFAULT_EMBEDDING_MODEL = "voyage-4-lite"
DEFAULT_TOKENIZER_MODEL = "voyageai/voyage-4-lite"
DEFAULT_TOKENIZER_REVISION = "0335ddf7698395712e3220733b4079006951cfef"
_RETRIEVAL_PREFIXES = {
    "query": "Represent the query for retrieving supporting documents: ",
    "document": "Represent the document for retrieval: ",
}
type InputType = Literal["query", "document"]
type JsonObject = dict[str, object]


class TextTokenizer(Protocol):
    """Count and split text without silently truncating it."""

    @property
    def fingerprint(self) -> str:
        """Identify tokenization settings used by the chunker."""
        ...

    def count(self, text: str) -> int:
        """Return the complete token count."""
        ...

    def split(self, text: str, max_tokens: int, overlap: int) -> list[str]:
        """Split text into bounded, overlapping spans."""
        ...


class ModelTokenizer:
    """Use the configured tokenizer without loading an embedding model locally."""

    def __init__(self, tokenizer: Tokenizer) -> None:
        self._tokenizer = tokenizer
        self._tokenizer.no_truncation()

    @classmethod
    def from_file(cls, path: Path) -> "ModelTokenizer":
        """Load a previously downloaded tokenizer file."""
        return cls(Tokenizer.from_file(str(path)))

    def count(self, text: str) -> int:
        """Return the count including special tokens."""
        return len(self._tokenizer.encode(text).ids)

    @property
    def fingerprint(self) -> str:
        """Include vocabulary and special-token settings in index invalidation."""
        return sha256(self._tokenizer.to_str().encode()).hexdigest()

    def split(self, text: str, max_tokens: int, overlap: int) -> list[str]:
        """Preserve original characters while splitting on token offsets."""
        if max_tokens <= overlap or overlap < 0:
            raise ValueError("Token limit must exceed the nonnegative overlap.")
        encoding = self._tokenizer.encode(text, add_special_tokens=False)
        offsets = [(start, end) for start, end in encoding.offsets if end > start]
        if not offsets:
            return []
        # Reserve space for special tokens and re-tokenization at span boundaries.
        width = max_tokens - self.count("") - 8
        if width <= overlap:
            raise ValueError("Token limit is too small for special tokens and overlap.")
        result: list[str] = []
        start = 0
        while start < len(offsets):
            end = min(start + width, len(offsets))
            candidate = text[offsets[start][0] : offsets[end - 1][1]].strip()
            while self.count(candidate) > max_tokens and end > start + 1:
                end -= 1
                candidate = text[offsets[start][0] : offsets[end - 1][1]].strip()
            if not candidate or self.count(candidate) > max_tokens:
                raise ValueError("A text span could not fit the embedding token limit.")
            result.append(candidate)
            if end == len(offsets):
                break
            start = max(start + 1, end - overlap)
        return result


class Embedder(Protocol):
    """Embed documents and queries in the same versioned vector space."""

    @property
    def fingerprint(self) -> str:
        """Identify the complete embedding configuration."""
        ...

    def embed_documents(self, texts: Sequence[str]) -> list[tuple[float, ...]]:
        """Return one normalized vector per document."""
        ...

    def embed_query(self, query: str) -> tuple[float, ...]:
        """Return a normalized retrieval-query vector."""
        ...


@dataclass(frozen=True)
class EmbeddingConfig:
    """Identify an explicit Voyage endpoint and vector space without credentials."""

    model: str = DEFAULT_EMBEDDING_MODEL
    base_url: str = "https://ai.mongodb.com/v1"
    dimensions: int = 1024

    @property
    def fingerprint(self) -> str:
        """Keep endpoint, model, dimensions and retrieval preprocessing in the cache identity."""
        payload = (
            self.model,
            self.base_url.rstrip("/"),
            self.dimensions,
            "voyage-input-type-float-l2-v1",
        )
        return sha256(json.dumps(payload).encode()).hexdigest()


def _retry_after(value: str | None) -> int:
    """Read a provider retry delay without accepting invalid or past dates as a cooldown."""
    if value is None:
        return 60
    try:
        delay = float(value)
    except ValueError:
        try:
            delay = (parsedate_to_datetime(value) - datetime.now(UTC)).total_seconds()
        except (TypeError, ValueError, OverflowError):
            return 60
    return max(1, math.ceil(delay)) if math.isfinite(delay) else 60


class VoyageEmbedder:
    """Cache hosted Voyage vectors and bound requests to a shared trial-rate window."""

    def __init__(
        self,
        cache_path: Path,
        *,
        client: httpx2.Client,
        tokenizer: TextTokenizer,
        config: EmbeddingConfig | None = None,
        max_batch_tokens: int = 8000,
        requests_per_minute: int = 3,
        tokens_per_minute: int = 10000,
        clock: Callable[[], float] = monotonic,
    ) -> None:
        """Retain one HTTP client, cache and request window for all policy searches.

        Parameters:
            cache_path: Persistent vector cache, separate from the source directory.
            client: Authenticated HTTP client with an explicit timeout and no redirects.
            tokenizer: Compatible local tokenizer for token-aware batching.
            config: Explicit model endpoint and output dimensions.
            max_batch_tokens: Maximum estimated input tokens in one batch.
            requests_per_minute: Shared request allowance in the most recent minute.
            tokens_per_minute: Shared estimated token allowance in the most recent minute.
            clock: Monotonic clock, replaceable in deterministic rate-limit tests.
        """
        self.config = config or EmbeddingConfig()
        self._client = client
        self._tokenizer = tokenizer
        self._batch_tokens = min(max_batch_tokens, tokens_per_minute)
        self._rpm = requests_per_minute
        self._tpm = tokens_per_minute
        self._clock = clock
        self._requests: deque[tuple[float, int]] = deque()
        self._cooldown_until = 0.0
        self._path = cache_path
        self._lock = RLock()
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with closing(sqlite3.connect(self._path)) as connection, connection:
            connection.execute(
                "CREATE TABLE IF NOT EXISTS vectors "
                "(cache_key TEXT PRIMARY KEY, vector_json TEXT NOT NULL)"
            )

    @property
    def fingerprint(self) -> str:
        """Expose the vector-space identity to document indexes."""
        return self.config.fingerprint

    def embed_documents(self, texts: Sequence[str]) -> list[tuple[float, ...]]:
        """Embed passages using the provider's document retrieval mode."""
        return self._embed(list(texts), "document")

    def embed_query(self, query: str) -> tuple[float, ...]:
        """Embed a query in its own retrieval mode and cache namespace."""
        return self._embed([query], "query")[0]

    def _vector(self, raw: object) -> tuple[float, ...]:
        array = np.asarray(raw, dtype=np.float64)
        if array.shape != (self.config.dimensions,) or not np.isfinite(array).all():
            raise ValueError("Embedding response has invalid dimensions or nonfinite values.")
        norm = float(np.linalg.norm(array))
        if not np.isfinite(norm) or norm == 0:
            raise ValueError("Embedding response contains a zero or invalid vector.")
        return tuple(float(value) for value in array / norm)

    def _reserve(self, tokens: int) -> None:
        """Reserve a request or return an actionable wait time without sleeping in the UI."""
        now = self._clock()
        while self._requests and self._requests[0][0] <= now - 60:
            self._requests.popleft()
        remaining = len(self._requests)
        used = sum(value for _, value in self._requests)
        delay = max(0.0, self._cooldown_until - now)
        for timestamp, cost in self._requests:
            if remaining < self._rpm and used + tokens <= self._tpm:
                break
            delay = max(delay, timestamp + 60 - now)
            remaining -= 1
            used -= cost
        if delay > 0:
            raise ValueError(
                f"Embedding service rate limit reached. Retry in {math.ceil(delay)} seconds. "
                "Completed embeddings remain cached."
            )
        self._requests.append((now, tokens))

    def _batches(
        self, keys: list[str], texts: dict[str, str], input_type: InputType
    ) -> Iterator[tuple[list[str], int]]:
        """Pack bounded batches including the retrieval prompt added by the server."""
        batch: list[str] = []
        tokens = 0
        for key in keys:
            count = self._tokenizer.count(_RETRIEVAL_PREFIXES[input_type] + texts[key])
            if count > self._batch_tokens:
                raise ValueError(
                    "An embedding input exceeds the token budget. Shorten the question or reduce "
                    "CHUNK_MAX_TOKENS; waiting cannot make this input fit."
                )
            if batch and (tokens + count > self._batch_tokens or len(batch) == 32):
                yield batch, tokens
                batch, tokens = [], 0
            batch.append(key)
            tokens += count
        if batch:
            yield batch, tokens

    def _read_vectors(self, payload: JsonObject, count: int) -> list[tuple[float, ...]]:
        """Validate indexed responses before assigning any vector to its input text."""
        data = payload.get("data")
        if not isinstance(data, list):
            raise ValueError("Embedding response does not match the requested batch.")
        values = cast(list[object], data)
        if len(values) != count:
            raise ValueError("Embedding response does not match the requested batch.")
        vectors: dict[int, tuple[float, ...]] = {}
        for value in values:
            if not isinstance(value, dict):
                raise ValueError("Embedding response contains an invalid result.")
            row = cast(JsonObject, value)
            index = row.get("index")
            if type(index) is not int or not 0 <= index < count or index in vectors:
                raise ValueError("Embedding response contains invalid or repeated indexes.")
            vectors[index] = self._vector(row.get("embedding"))
        return [vectors[index] for index in range(count)]

    def _response_vectors(self, response: httpx2.Response, count: int) -> list[tuple[float, ...]]:
        """Validate vectors and reconcile the request reservation with reported token usage."""
        try:
            raw: object = response.json()
        except ValueError as error:
            raise ValueError("Embedding response is not valid JSON.") from error
        if not isinstance(raw, dict):
            raise ValueError("Embedding response must be a JSON object.")
        payload = cast(JsonObject, raw)
        vectors = self._read_vectors(payload, count)
        usage = payload.get("usage")
        if not isinstance(usage, dict):
            raise ValueError("Embedding response is missing token usage.")
        total = cast(JsonObject, usage).get("total_tokens")
        if type(total) is not int or total < 0:
            raise ValueError("Embedding response contains invalid token usage.")
        timestamp, estimate = self._requests[-1]
        self._requests[-1] = (timestamp, max(estimate, total))
        return vectors

    def _request(
        self, texts: list[str], input_type: InputType, tokens: int
    ) -> list[tuple[float, ...]]:
        """Call only the configured endpoint and translate failures without exposing secrets."""
        self._reserve(tokens)
        try:
            response = self._client.post(
                self.config.base_url.rstrip("/") + "/embeddings",
                json={
                    "input": texts,
                    "model": self.config.model,
                    "input_type": input_type,
                    "output_dimension": self.config.dimensions,
                    "output_dtype": "float",
                    "truncation": False,
                },
            )
        except httpx2.TimeoutException as error:
            raise ValueError("The embedding provider timed out. Please retry later.") from error
        except httpx2.RequestError as error:
            raise ValueError("The embedding provider could not be reached. Retry later.") from error
        if response.status_code == 429:
            delay = _retry_after(response.headers.get("Retry-After"))
            self._cooldown_until = self._clock() + delay
            raise ValueError(
                f"Embedding provider request limit reached. Retry in {delay} seconds. "
                "Completed embeddings remain cached."
            )
        if response.status_code != 200:
            reason = {
                401: "Check VOYAGE_API_KEY and its endpoint permissions.",
                403: "Check VOYAGE_API_KEY and its endpoint permissions.",
                402: "The embedding provider reports insufficient credits.",
            }.get(response.status_code, "Check the provider configuration and retry later.")
            raise ValueError(
                f"Embedding service rejected the request ({response.status_code}). {reason}"
            )
        return self._response_vectors(response, len(texts))

    def _embed(self, texts: list[str], input_type: InputType) -> list[tuple[float, ...]]:
        """Reuse role-specific vectors and commit each validated batch for resumable ingestion."""
        if not texts:
            return []
        if any(not text.strip() for text in texts):
            raise ValueError("Embedding input cannot be empty.")
        keys = [
            sha256(
                json.dumps((self.fingerprint, input_type, text), ensure_ascii=False).encode()
            ).hexdigest()
            for text in texts
        ]
        with self._lock, closing(sqlite3.connect(self._path, timeout=10)) as connection, connection:
            cached: dict[str, tuple[float, ...]] = {}
            for key in set(keys):
                row = connection.execute(
                    "SELECT vector_json FROM vectors WHERE cache_key = ?", (key,)
                ).fetchone()
                if row is not None:
                    cached[key] = self._vector(json.loads(str(row[0])))
            missing = list(dict.fromkeys(key for key in keys if key not in cached))
            by_key = dict(zip(keys, texts, strict=True))
            for batch_keys, tokens in self._batches(missing, by_key, input_type):
                vectors = self._request([by_key[key] for key in batch_keys], input_type, tokens)
                for key, vector in zip(batch_keys, vectors, strict=True):
                    cached[key] = vector
                    connection.execute(
                        "INSERT OR REPLACE INTO vectors VALUES (?, ?)",
                        (key, json.dumps(vector, allow_nan=False)),
                    )
                connection.commit()
            return [cached[key] for key in keys]
