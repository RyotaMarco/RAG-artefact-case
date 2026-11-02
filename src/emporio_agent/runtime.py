"""Publish complete source generations without serving partial or stale updates."""

import hashlib
import json
import logging
import os
import re
import tempfile
from dataclasses import dataclass
from decimal import localcontext
from pathlib import Path
from threading import RLock

import httpx2
from huggingface_hub import (
    hf_hub_download,  # pyright: ignore[reportUnknownVariableType]
)
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from emporio_agent.config import Settings
from emporio_agent.data import (
    Catalog,
    DataError,
    SourceFile,
    build_catalog,
    discover_sources,
    inventory_fingerprint,
    write_catalog,
)
from emporio_agent.documents import PolicyChunk, PolicyIndex, PolicySection, build_index
from emporio_agent.embeddings import (
    DEFAULT_TOKENIZER_MODEL,
    DEFAULT_TOKENIZER_REVISION,
    Embedder,
    ModelTokenizer,
    TextTokenizer,
    VoyageEmbedder,
)
from emporio_agent.policies import FinancialRuleSet, extract_rules

PIPELINE_VERSION = "layout-catalog-money-v2"
logger = logging.getLogger(__name__)


class ConfigurationError(ValueError):
    """A required external capability has not been configured."""


class _IndexCache(BaseModel):
    """Validate persisted policy metadata before reusing an unchanged generation."""

    model_config = ConfigDict(extra="forbid")
    generation_id: str
    index_fingerprint: str
    chunks: tuple[PolicyChunk, ...] = Field(min_length=1)
    sections: tuple[PolicySection, ...] = Field(min_length=1)


@dataclass(frozen=True)
class Snapshot:
    """Bind catalog, citations, rules and retrieval to one immutable inventory."""

    generation_id: str
    catalog: Catalog
    policies: PolicyIndex | None
    rules: FinancialRuleSet

    def capabilities(self) -> dict[str, bool]:
        """Describe which user requests the supplied sources can actually support."""
        tables = self.catalog.tables
        return {
            "catalog": "products" in tables,
            "promotions": "products" in tables and "promotions" in tables,
            "orders": "orders" in tables and "customers" in tables,
            "policies": self.policies is not None,
        }


def load_tokenizer(settings: Settings) -> ModelTokenizer:
    """Download the configured tokenizer, never local model weights or PyTorch."""
    model = settings.tokenizer_model or "voyageai/" + settings.embedding.model
    revision = settings.tokenizer_revision or (
        DEFAULT_TOKENIZER_REVISION if model == DEFAULT_TOKENIZER_MODEL else "main"
    )
    cached = hf_hub_download(
        model,
        filename="tokenizer.json",
        revision=revision,
        cache_dir=str(settings.runtime_dir / "tokenizer"),
        token=False,
    )
    return ModelTokenizer.from_file(Path(cached))


class StoreRuntime:
    """Serialize source refreshes while keeping conversation state outside shared resources."""

    def __init__(
        self,
        settings: Settings,
        *,
        embedder: Embedder | None = None,
        tokenizer: TextTokenizer | None = None,
    ) -> None:
        """Accept narrow embedding/tokenizer injections for API-free integration tests."""
        self.settings = settings
        self._embedder = embedder
        self._tokenizer = tokenizer
        self._current: Snapshot | None = None
        self._lock = RLock()

    def _policy_resources(self) -> None:
        """Initialize the hosted provider and local tokenizer only when PDFs need them."""
        if self._embedder is None and not self.settings.voyage_key:
            raise ConfigurationError("VOYAGE_API_KEY is required to index the supplied PDFs.")
        if self._tokenizer is None:
            self._tokenizer = load_tokenizer(self.settings)
        if self._embedder is None:
            self._embedder = VoyageEmbedder(
                self.settings.runtime_dir / "vectors.sqlite3",
                client=httpx2.Client(
                    headers={"Authorization": "Bearer " + self.settings.voyage_key},
                    timeout=self.settings.embedding_timeout_seconds,
                    follow_redirects=False,
                ),
                tokenizer=self._tokenizer,
                config=self.settings.embedding,
                max_batch_tokens=self.settings.embedding_batch_tokens,
                requests_per_minute=self.settings.embedding_requests_per_minute,
                tokens_per_minute=self.settings.embedding_tokens_per_minute,
            )

    def _policy_index(
        self,
        pdfs: tuple[SourceFile, ...],
        generation: str,
        embedder: Embedder,
        tokenizer: TextTokenizer,
    ) -> PolicyIndex:
        """Restore validated policy metadata or atomically cache a rebuilt index."""
        metadata_path = self.settings.runtime_dir / f"policies-{generation}.json"
        if metadata_path.exists():
            try:
                cached = _IndexCache.model_validate_json(metadata_path.read_bytes())
                if cached.generation_id == generation:
                    return PolicyIndex(
                        cached.chunks, cached.sections, embedder, cached.index_fingerprint
                    )
            except (ValidationError, OSError):
                pass
        index = build_index(pdfs, embedder, tokenizer, chunking=self.settings.chunking)
        cache = _IndexCache(
            generation_id=generation,
            index_fingerprint=index.fingerprint,
            chunks=index.chunks,
            sections=index.sections,
        )
        candidate = metadata_path.with_suffix(".pending")
        candidate.write_text(cache.model_dump_json(), encoding="utf-8")
        os.replace(candidate, metadata_path)
        return index

    def _publish(self, snapshot: Snapshot, source_id: str) -> None:
        """Replace the sole operational database only after the candidate passes validation."""
        runtime_dir = self.settings.runtime_dir
        with tempfile.NamedTemporaryFile(
            dir=runtime_dir, suffix=".pending", delete=False
        ) as candidate:
            candidate_path = Path(candidate.name)
        try:
            write_catalog(snapshot.catalog, candidate_path)
            if inventory_fingerprint(discover_sources(self.settings.data_dir)) != source_id:
                raise DataError("Input files changed during ingestion. Retry after updates finish.")
            os.replace(candidate_path, runtime_dir / "operational.sqlite3")
            self._current = snapshot
        finally:
            candidate_path.unlink(missing_ok=True)

    def _prune_legacy_catalogs(self) -> None:
        """Remove only old generation catalogs directly inside the resolved runtime directory."""
        root = self.settings.runtime_dir.resolve()
        for path in root.glob("operational-*.sqlite3"):
            try:
                if (
                    re.fullmatch(r"operational-[0-9a-f]{64}\.sqlite3", path.name)
                    and not path.is_symlink()
                    and path.resolve().parent == root
                    and path.is_file()
                ):
                    path.unlink()
            except OSError as error:
                logger.warning("Old operational catalog cleanup failed: %s", type(error).__name__)

    def refresh(self) -> Snapshot:
        """Validate a candidate completely before publishing it to new agent runs.

        Raises:
            ValueError: A source, external service or configuration is invalid.
        """
        with self._lock:
            sources = discover_sources(self.settings.data_dir)
            source_id = inventory_fingerprint(sources)
            runtime_dir = self.settings.runtime_dir
            runtime_dir.mkdir(parents=True, exist_ok=True)
            pdfs = tuple(
                source for source in sources if source.relative_path.lower().endswith(".pdf")
            )
            if pdfs:
                self._policy_resources()
            identity = (
                source_id,
                PIPELINE_VERSION,
                self._embedder.fingerprint if pdfs and self._embedder is not None else "no-pdfs",
                self._tokenizer.fingerprint
                if pdfs and self._tokenizer is not None
                else "no-tokenizer",
                self.settings.chunking.fingerprint if pdfs else "no-chunks",
            )
            generation = hashlib.sha256(json.dumps(identity).encode()).hexdigest()
            if self._current is not None and self._current.generation_id == generation:
                return self._current
            catalog = build_catalog(sources)
            index: PolicyIndex | None = None
            if pdfs:
                assert self._embedder is not None and self._tokenizer is not None
                index = self._policy_index(pdfs, generation, self._embedder, self._tokenizer)
            with localcontext() as context:
                context.prec = 120
                rules = extract_rules(index.sections if index is not None else ())
            snapshot = Snapshot(generation, catalog, index, rules)
            self._publish(snapshot, source_id)
            self._prune_legacy_catalogs()
            return snapshot
