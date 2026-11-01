"""Hosted embeddings with a persistent, validated vector cache."""

import json
import sqlite3
from collections.abc import Sequence
from contextlib import closing
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from threading import RLock
from typing import Protocol

import numpy as np
from tokenizers import Tokenizer


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
    """Use the baked tokenizer without loading an embedding model locally."""

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


class FeatureExtractionClient(Protocol):
    """The narrow subset of Hugging Face's client used for embeddings."""

    def feature_extraction(
        self, text: str | list[str], *, model: str, normalize: bool, truncate: bool
    ) -> object:
        """Return embedding vectors for the supplied inputs."""
        ...


@dataclass(frozen=True)
class EmbeddingConfig:
    """Keep model, dimensions and query formatting in the cache identity."""

    model: str = "Qwen/Qwen3-Embedding-0.6B"
    provider: str = "deepinfra"
    dimensions: int = 1024
    instruction: str = (
        "Given a question about store policies, retrieve relevant passages that answer it."
    )

    @property
    def fingerprint(self) -> str:
        """Identify a vector space and its preprocessing configuration."""
        payload = (self.model, self.provider, self.dimensions, self.instruction, "l2-v1")
        return sha256(json.dumps(payload).encode()).hexdigest()


class HFEmbedder:
    """Cache hosted vectors using a caller-supplied client with a request timeout."""

    def __init__(
        self,
        cache_path: Path,
        *,
        client: FeatureExtractionClient,
        config: EmbeddingConfig | None = None,
    ) -> None:
        self.config = config or EmbeddingConfig()
        self._client = client
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
        """Embed policy passages without the query-only retrieval instruction."""
        return self._embed(list(texts))

    def embed_query(self, query: str) -> tuple[float, ...]:
        """Apply the Qwen retrieval instruction to the query only."""
        prepared = f"Instruct: {self.config.instruction}\nQuery: {query}"
        return self._embed([prepared])[0]

    def _vector(self, raw: object) -> tuple[float, ...]:
        array = np.asarray(raw, dtype=np.float64)
        if array.shape != (self.config.dimensions,) or not np.isfinite(array).all():
            raise ValueError("Embedding response has invalid dimensions or nonfinite values.")
        norm = float(np.linalg.norm(array))
        if not np.isfinite(norm) or norm == 0:
            raise ValueError("Embedding response contains a zero or invalid vector.")
        return tuple(float(value) for value in array / norm)

    def _embed(self, texts: list[str]) -> list[tuple[float, ...]]:
        if any(not text.strip() for text in texts):
            raise ValueError("Embedding input cannot be empty.")
        if not texts:
            return []
        keys = [sha256((self.fingerprint + text).encode()).hexdigest() for text in texts]
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
            for offset in range(0, len(missing), 32):
                batch_keys = missing[offset : offset + 32]
                batch = [by_key[key] for key in batch_keys]
                response = self._client.feature_extraction(
                    batch, model=self.config.model, normalize=True, truncate=False
                )
                array = np.asarray(response, dtype=np.float64)
                if array.shape != (len(batch), self.config.dimensions):
                    raise ValueError("Embedding response does not match the requested batch.")
                for key, raw in zip(batch_keys, array, strict=True):
                    vector = self._vector(raw)
                    cached[key] = vector
                    connection.execute(
                        "INSERT OR REPLACE INTO vectors VALUES (?, ?)",
                        (key, json.dumps(vector, allow_nan=False)),
                    )
                connection.commit()
            return [cached[key] for key in keys]
