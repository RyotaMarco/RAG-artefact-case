"""Exercise atomic source generations with real PDF extraction and SQLite persistence."""

import json
import sqlite3
from collections.abc import Callable, Sequence
from dataclasses import replace
from decimal import Decimal
from io import BytesIO
from pathlib import Path
from typing import cast

import pytest
from pypdf import PdfWriter
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject
from tokenizers import Tokenizer
from tokenizers.models import WordLevel

from emporio_agent.config import Settings
from emporio_agent.data import Catalog, DataError, write_catalog
from emporio_agent.documents import ChunkConfig
from emporio_agent.embeddings import (
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_TOKENIZER_REVISION,
    EmbeddingConfig,
    HFEmbedder,
)
from emporio_agent.policies import PolicyUnavailableError
from emporio_agent.runtime import Snapshot, StoreRuntime, load_tokenizer


class CountingTokenizer:
    """Track actual chunking work without downloading a tokenizer."""

    fingerprint = "runtime-character-tokens-v1"

    def __init__(self) -> None:
        """Begin with no tokenization work and no injected failure."""
        self.calls = 0
        self.fail = False

    def count(self, text: str) -> int:
        """Count characters and optionally detect an unexpected cache rebuild."""
        self.calls += 1
        if self.fail:
            raise AssertionError("A validated metadata cache should not rechunk PDFs.")
        return len(text)

    def split(self, text: str, max_tokens: int, overlap: int) -> list[str]:
        """Keep complete deterministic spans within the requested token window."""
        return [
            text[start : start + max_tokens] for start in range(0, len(text), max_tokens - overlap)
        ]


class RecordingEmbedder:
    """Record policy-only inputs and inject a single candidate-generation failure."""

    fingerprint = "runtime-fake-vectors-v1"

    def __init__(self) -> None:
        """Keep document batches and an optional source-change callback."""
        self.batches: list[tuple[str, ...]] = []
        self.fail = False
        self.on_documents: Callable[[], None] | None = None

    def embed_documents(self, texts: Sequence[str]) -> list[tuple[float, ...]]:
        """Return nonzero vectors while recording exactly what crosses the boundary."""
        self.batches.append(tuple(texts))
        if self.fail:
            raise TimeoutError("The fake embedding provider timed out.")
        callback = self.on_documents
        self.on_documents = None
        if callback is not None:
            callback()
        return [(1.0, 2.0) for _ in texts]

    def embed_query(self, query: str) -> tuple[float, ...]:
        """Return a compatible vector without an external request."""
        assert query
        return (1.0, 2.0)


class RecordingClient:
    """Stand in for the hosted provider while exercising the real SQLite vector cache."""

    def __init__(self) -> None:
        """Record provider attempts independently of embedder process lifetime."""
        self.calls: list[list[str]] = []
        self.fail = False

    def feature_extraction(
        self, text: str | list[str], *, model: str, normalize: bool, truncate: bool
    ) -> object:
        """Provide deterministic vectors and fail if a restart unexpectedly misses cache."""
        assert model and normalize and not truncate
        batch = [text] if isinstance(text, str) else text
        self.calls.append(batch)
        if self.fail:
            raise AssertionError("A persisted vector must not trigger another provider call.")
        return [[3.0, 4.0] for _ in batch]


def _settings(directory: Path) -> Settings:
    data = directory / "data"
    data.mkdir()
    return Settings(data, directory / "runtime")


def _categories(directory: Path) -> None:
    (directory / "categories.csv").write_text(
        "category_id,name,description\n7,Acoustic guitars,CSV_ONLY_SENTINEL\n",
        encoding="utf-8",
    )


def _products(directory: Path, price: str = "1234.56", stock: int = 4) -> None:
    (directory / "inventory-export.csv").write_text(
        "product_id,price_brl,name,category_id,description,stock_quantity,status,specs,created_at\n"
        f"41,{price},Guitar ABC,7,CSV_ONLY_SENTINEL,{stock},active,{{}},2026-01-01\n",
        encoding="utf-8",
    )


def _pdf(path: Path, lines: list[str]) -> None:
    writer = PdfWriter()
    page = writer.add_blank_page(width=612, height=792)
    font = DictionaryObject(
        {
            NameObject("/Type"): NameObject("/Font"),
            NameObject("/Subtype"): NameObject("/Type1"),
            NameObject("/BaseFont"): NameObject("/Helvetica"),
        }
    )
    page[NameObject("/Resources")] = DictionaryObject(
        {NameObject("/Font"): DictionaryObject({NameObject("/F1"): font})}
    )
    commands = ["BT /F1 10 Tf 12 TL 50 750 Td"]
    for line in lines:
        escaped = line.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
        commands.append(f"({escaped}) Tj T*")
    commands.append("ET")
    stream = DecodedStreamObject()
    stream.set_data("\n".join(commands).encode("latin-1"))
    page[NameObject("/Contents")] = stream
    buffer = BytesIO()
    writer.write(buffer)
    path.write_bytes(buffer.getvalue())


def _payments(path: Path, percentage: str = "5") -> None:
    _pdf(path, ["1. Payments", f"PIX {percentage}% de desconto sobre o preco de tabela."])


def _runtime(
    directory: Path,
) -> tuple[Settings, StoreRuntime, RecordingEmbedder, CountingTokenizer]:
    settings = _settings(directory)
    _categories(settings.data_dir)
    _products(settings.data_dir)
    _payments(settings.data_dir / "policies.pdf")
    embedder = RecordingEmbedder()
    tokenizer = CountingTokenizer()
    return (
        settings,
        StoreRuntime(settings, embedder=embedder, tokenizer=tokenizer),
        embedder,
        tokenizer,
    )


def _database(settings: Settings) -> Path:
    return settings.runtime_dir / "operational.sqlite3"


def _scalar(path: Path, sql: str) -> object:
    connection = sqlite3.connect(path)
    try:
        row = cast(tuple[object, ...] | None, connection.execute(sql).fetchone())
        assert row is not None
        return row[0]
    finally:
        connection.close()


def _citations(snapshot: Snapshot) -> set[str]:
    assert snapshot.policies is not None
    return {citation.source for chunk in snapshot.policies.chunks for citation in chunk.citations}


@pytest.mark.parametrize(
    "embedding_model, tokenizer_model, revision, expected_model, expected_revision",
    [
        (DEFAULT_EMBEDDING_MODEL, None, None, DEFAULT_EMBEDDING_MODEL, DEFAULT_TOKENIZER_REVISION),
        ("example/encoder", None, None, "example/encoder", "main"),
        (DEFAULT_EMBEDDING_MODEL, "example/tokenizer", None, "example/tokenizer", "main"),
        (
            "example/encoder",
            DEFAULT_EMBEDDING_MODEL,
            None,
            DEFAULT_EMBEDDING_MODEL,
            DEFAULT_TOKENIZER_REVISION,
        ),
        (
            "example/encoder",
            "example/tokenizer",
            "fixture-revision",
            "example/tokenizer",
            "fixture-revision",
        ),
    ],
)
def test_tokenizer_download_follows_effective_repository_and_revision(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    embedding_model: str,
    tokenizer_model: str | None,
    revision: str | None,
    expected_model: str,
    expected_revision: str,
) -> None:
    """Pin the default Qwen tokenizer without applying its commit to an unrelated model."""
    local_tokenizer = tmp_path / "tokenizer.json"
    Tokenizer(WordLevel({"[UNK]": 0}, unk_token="[UNK]")).save(str(local_tokenizer))
    calls: list[dict[str, object]] = []

    def download(repo_id: str, **kwargs: object) -> str:
        calls.append({"repo_id": repo_id, **kwargs})
        return str(local_tokenizer)

    monkeypatch.setattr("emporio_agent.runtime.hf_hub_download", download)
    settings = replace(
        _settings(tmp_path),
        hf_token="hf_fixture",
        embedding=EmbeddingConfig(model=embedding_model),
        tokenizer_model=tokenizer_model,
        tokenizer_revision=revision,
    )
    assert load_tokenizer(settings).count("policy") == 1
    assert calls == [
        {
            "repo_id": expected_model,
            "filename": "tokenizer.json",
            "revision": expected_revision,
            "cache_dir": str(settings.runtime_dir / "tokenizer"),
            "token": "hf_fixture",
        }
    ]


def test_runtime_embedding_settings_reach_client_and_invalidate_generation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Changing model conventions rebuilds policy vectors with the configured hosted client."""
    settings = _settings(tmp_path)
    _categories(settings.data_dir)
    _payments(settings.data_dir / "payments.pdf")
    settings = replace(
        settings,
        hf_token="hf_fixture",
        embedding_timeout_seconds=12,
        embedding=EmbeddingConfig(
            model="example/encoder",
            provider="example-provider",
            dimensions=2,
            query_prefix="query:",
            document_prefix="passage:",
        ),
    )
    client = RecordingClient()
    arguments: list[dict[str, object]] = []

    def make_client(**kwargs: object) -> RecordingClient:
        arguments.append(kwargs)
        return client

    monkeypatch.setattr("emporio_agent.runtime.InferenceClient", make_client)
    first = StoreRuntime(settings, tokenizer=CountingTokenizer()).refresh()
    assert arguments == [{"provider": "example-provider", "token": "hf_fixture", "timeout": 12}]
    assert client.calls[0][0].startswith("passage: ")
    updated = replace(settings, embedding=replace(settings.embedding, document_prefix="document:"))
    second = StoreRuntime(updated, tokenizer=CountingTokenizer()).refresh()
    assert first.generation_id != second.generation_id
    assert len(client.calls) == 2
    assert client.calls[-1][0].startswith("document: ")


def test_chunk_settings_rebuild_metadata_and_reuse_vectors_for_unchanged_passages(
    tmp_path: Path,
) -> None:
    """A window change rechunks PDFs without paying again for identical resulting text."""
    settings = _settings(tmp_path)
    _categories(settings.data_dir)
    _payments(settings.data_dir / "payments.pdf")
    client = RecordingClient()
    embedder = HFEmbedder(
        settings.runtime_dir / "vectors.sqlite3",
        client=client,
        config=EmbeddingConfig(dimensions=2),
    )
    tokenizer = CountingTokenizer()
    first = StoreRuntime(settings, embedder=embedder, tokenizer=tokenizer).refresh()
    original_calls = tokenizer.calls
    updated = replace(
        settings, chunking=ChunkConfig(target_tokens=500, max_tokens=750, overlap_tokens=60)
    )
    second = StoreRuntime(updated, embedder=embedder, tokenizer=tokenizer).refresh()
    assert second.generation_id != first.generation_id
    assert tokenizer.calls > original_calls
    assert first.policies is not None and second.policies is not None
    assert second.policies.chunks == first.policies.chunks
    assert len(client.calls) == 1


def test_unchanged_refresh_reuses_generation_and_does_not_embed_csv(tmp_path: Path) -> None:
    """Reuse one published generation while preserving exact SQLite money and input separation."""
    settings, runtime, embedder, tokenizer = _runtime(tmp_path)
    first = runtime.refresh()
    token_calls = tokenizer.calls
    published = _database(settings).read_bytes()
    assert runtime.refresh() is first
    assert tokenizer.calls == token_calls
    assert len(embedder.batches) == 1
    assert all("CSV_ONLY_SENTINEL" not in text for batch in embedder.batches for text in batch)
    assert _database(settings).read_bytes() == published
    assert not (settings.runtime_dir / "current.json").exists()
    assert _scalar(_database(settings), "SELECT price_cents FROM products") == 123456
    assert _scalar(_database(settings), "SELECT typeof(price_cents) FROM products") == "integer"
    assert first.rules.quote(Decimal("100.00")).total == Decimal("95.00")


@pytest.mark.parametrize("count", [40, 90])
def test_runtime_accepts_arbitrary_csv_shards_and_portable_schema_metadata(
    tmp_path: Path, count: int
) -> None:
    """File counts and filenames never become fixed prompt or ingestion assumptions."""
    settings = _settings(tmp_path)
    for index in range(count):
        (settings.data_dir / f"batch-{index}.CSV").write_text(
            f"description,name,category_id\nDescription,Category {index},{index + 1}\n",
            encoding="utf-8",
        )
    snapshot = StoreRuntime(settings).refresh()
    assert len(snapshot.catalog.tables["categories"].rows) == count
    assert _scalar(_database(settings), "SELECT COUNT(*) FROM categories") == count
    metadata = snapshot.catalog.describe()
    category_metadata = cast(dict[str, object], metadata["categories"])
    assert "row_count" not in category_metadata
    assert "sources" not in category_metadata
    assert snapshot.capabilities() == {
        "catalog": False,
        "promotions": False,
        "orders": False,
        "policies": False,
    }


def test_csv_updates_and_removals_replace_data_and_capabilities(tmp_path: Path) -> None:
    """Publish source changes without appending stale rows or retaining removed capabilities."""
    settings = _settings(tmp_path)
    _categories(settings.data_dir)
    _products(settings.data_dir)
    runtime = StoreRuntime(settings)
    first = runtime.refresh()
    _products(settings.data_dir, "90071992547409.93", stock=0)
    (settings.data_dir / "offers.csv").write_text(
        "promotion_id,product_id,discount_percent,description,is_active\n"
        "8,41,7.5,Current promotion,1\n",
        encoding="utf-8",
    )
    updated = runtime.refresh()
    assert updated.generation_id != first.generation_id
    assert updated.capabilities()["promotions"] is True
    assert _scalar(_database(settings), "SELECT price_cents FROM products") == 9007199254740993
    assert _scalar(_database(settings), "SELECT COUNT(*) FROM products") == 1
    assert _scalar(_database(settings), "SELECT stock_quantity FROM products") == 0
    assert first.catalog.tables["products"].rows[0]["price_cents"] == 123456
    assert list(settings.runtime_dir.glob("operational*.sqlite3")) == [_database(settings)]
    (settings.data_dir / "offers.csv").unlink()
    (settings.data_dir / "inventory-export.csv").unlink()
    reduced = runtime.refresh()
    assert set(reduced.catalog.tables) == {"categories"}
    assert reduced.capabilities()["catalog"] is False
    assert reduced.capabilities()["promotions"] is False
    assert "products" not in reduced.catalog.describe()
    assert list(settings.runtime_dir.glob("operational*.sqlite3")) == [_database(settings)]
    assert _scalar(_database(settings), "SELECT COUNT(*) FROM categories") == 1


def test_order_capability_requires_both_order_and_customer_sources(tmp_path: Path) -> None:
    """An absent private relationship cannot remain advertised after a source removal."""
    settings = _settings(tmp_path)
    _categories(settings.data_dir)
    (settings.data_dir / "purchases.csv").write_text(
        "order_id,customer_id,order_date,status,total_brl,payment_method,tracking_code,"
        "estimated_delivery,notes\n12,3,2026-01-01,pending,125.00,pix,,,\n",
        encoding="utf-8",
    )
    runtime = StoreRuntime(settings)
    assert runtime.refresh().capabilities()["orders"] is False
    customers = settings.data_dir / "buyers.csv"
    customers.write_text(
        "customer_id,name,phone,email,city\n3,Example Buyer,000,buyer@example.test,Example City\n",
        encoding="utf-8",
    )
    assert runtime.refresh().capabilities()["orders"] is True
    customers.unlink()
    assert runtime.refresh().capabilities()["orders"] is False


def test_invalid_candidate_raises_without_replacing_current_snapshot(tmp_path: Path) -> None:
    """A failed candidate is visible as an error and cannot silently publish partial tables."""
    settings, runtime, _, _ = _runtime(tmp_path)
    first = runtime.refresh()
    published = _database(settings).read_bytes()
    invalid = settings.data_dir / "unexpected.csv"
    invalid.write_text("unsupported,fields\n1,2\n", encoding="utf-8")
    with pytest.raises(DataError, match="unexpected.csv"):
        runtime.refresh()
    assert _database(settings).read_bytes() == published
    assert getattr(runtime, "_current") is first
    assert _scalar(_database(settings), "SELECT COUNT(*) FROM products") == 1
    invalid.unlink()
    assert runtime.refresh() is first


def test_successful_publication_prunes_only_legacy_catalog_files(tmp_path: Path) -> None:
    """Remove known old generation files after success without touching unrelated runtime data."""
    settings = _settings(tmp_path)
    _categories(settings.data_dir)
    runtime = StoreRuntime(settings)
    runtime.refresh()
    legacy = settings.runtime_dir / f"operational-{'a' * 64}.sqlite3"
    legacy.write_bytes(b"legacy catalog")
    unrelated = settings.runtime_dir / "operational-user-backup.sqlite3"
    unrelated.write_bytes(b"keep this file")
    nested = settings.runtime_dir / "archive" / f"operational-{'b' * 64}.sqlite3"
    nested.parent.mkdir()
    nested.write_bytes(b"keep nested files")
    matching_directory = settings.runtime_dir / f"operational-{'c' * 64}.sqlite3"
    matching_directory.mkdir()
    invalid = settings.data_dir / "unexpected.csv"
    invalid.write_text("unsupported,fields\n1,2\n", encoding="utf-8")
    with pytest.raises(DataError):
        runtime.refresh()
    assert legacy.exists()
    invalid.unlink()
    _products(settings.data_dir)
    runtime.refresh()
    assert not legacy.exists()
    assert unrelated.read_bytes() == b"keep this file"
    assert nested.read_bytes() == b"keep nested files"
    assert matching_directory.is_dir()
    assert _database(settings).is_file()


def test_failed_atomic_replace_preserves_database_snapshot_and_allows_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An interrupted publication cannot overwrite the database or advance the active snapshot."""
    settings = _settings(tmp_path)
    _categories(settings.data_dir)
    _products(settings.data_dir)
    runtime = StoreRuntime(settings)
    first = runtime.refresh()
    published = _database(settings).read_bytes()
    _products(settings.data_dir, "777.00")

    def reject_replace(_source: Path, _target: Path) -> None:
        raise PermissionError("Fixture database replacement failed.")

    with monkeypatch.context() as patch:
        patch.setattr("emporio_agent.runtime.os.replace", reject_replace)
        with pytest.raises(PermissionError, match="replacement failed"):
            runtime.refresh()
    assert _database(settings).read_bytes() == published
    assert getattr(runtime, "_current") is first
    assert not list(settings.runtime_dir.glob("*.pending"))
    recovered = runtime.refresh()
    assert recovered.generation_id != first.generation_id
    assert _scalar(_database(settings), "SELECT price_cents FROM products") == 77700


def test_source_change_during_database_write_cannot_replace_active_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The final input check happens after SQLite creation and before its atomic replacement."""
    settings = _settings(tmp_path)
    _categories(settings.data_dir)
    _products(settings.data_dir)
    runtime = StoreRuntime(settings)
    first = runtime.refresh()
    published = _database(settings).read_bytes()
    _products(settings.data_dir, "444.00")

    def change_after_write(catalog: Catalog, path: Path) -> None:
        write_catalog(catalog, path)
        _products(settings.data_dir, "555.00")

    with monkeypatch.context() as patch:
        patch.setattr("emporio_agent.runtime.write_catalog", change_after_write)
        with pytest.raises(DataError, match="changed during ingestion"):
            runtime.refresh()
    assert _database(settings).read_bytes() == published
    assert getattr(runtime, "_current") is first
    assert not list(settings.runtime_dir.glob("*.pending"))
    assert runtime.refresh().catalog.tables["products"].rows[0]["price_cents"] == 55500


def test_provider_failure_preserves_current_then_allows_retry(tmp_path: Path) -> None:
    """Failed PDF indexing cannot publish the catalog or rules from a candidate generation."""
    settings, runtime, embedder, _ = _runtime(tmp_path)
    first = runtime.refresh()
    published = _database(settings).read_bytes()
    _payments(settings.data_dir / "policies.pdf", "7")
    embedder.fail = True
    with pytest.raises(TimeoutError, match="timed out"):
        runtime.refresh()
    assert _database(settings).read_bytes() == published
    assert getattr(runtime, "_current") is first
    assert first.rules.pix_rate == Decimal("0.05")
    embedder.fail = False
    recovered = runtime.refresh()
    assert recovered.rules.pix_rate == Decimal("0.07")
    assert recovered.generation_id != first.generation_id


def test_inputs_changed_during_indexing_cannot_be_published(tmp_path: Path) -> None:
    """Recheck the whole inventory before replacing the published operational database."""
    settings, runtime, embedder, _ = _runtime(tmp_path)
    first = runtime.refresh()
    published = _database(settings).read_bytes()
    _payments(settings.data_dir / "policies.pdf", "8")
    embedder.on_documents = lambda: _products(settings.data_dir, "888.00")
    with pytest.raises(DataError, match="changed during ingestion"):
        runtime.refresh()
    assert _database(settings).read_bytes() == published
    assert getattr(runtime, "_current") is first
    assert first.catalog.tables["products"].rows[0]["price_cents"] == 123456
    recovered = runtime.refresh()
    assert recovered.catalog.tables["products"].rows[0]["price_cents"] == 88800
    assert recovered.rules.pix_rate == Decimal("0.08")


def test_pdf_rename_add_duplicate_and_removal_rebuild_citations(tmp_path: Path) -> None:
    """Only current files contribute retrieval evidence, deduplicated chunks, and rules."""
    settings, runtime, embedder, _ = _runtime(tmp_path)
    first = runtime.refresh()
    renamed = settings.data_dir / "renamed.PDF"
    (settings.data_dir / "policies.pdf").rename(renamed)
    renamed_snapshot = runtime.refresh()
    assert renamed_snapshot.generation_id != first.generation_id
    assert _citations(renamed_snapshot) == {"renamed.PDF"}
    duplicate = settings.data_dir / "duplicate.pdf"
    duplicate.write_bytes(renamed.read_bytes())
    copied = runtime.refresh()
    assert copied.policies is not None
    assert len(copied.policies.chunks) == 1
    assert len(embedder.batches[-1]) == 1
    assert _citations(copied) == {"renamed.PDF", "duplicate.pdf"}
    returns = settings.data_dir / "returns.pdf"
    _pdf(returns, ["2. Returns", "Returns require the receipt and the original box."])
    added = runtime.refresh()
    assert _citations(added) == {"renamed.PDF", "duplicate.pdf", "returns.pdf"}
    renamed.unlink()
    duplicate.unlink()
    remaining = runtime.refresh()
    assert _citations(remaining) == {"returns.pdf"}
    assert remaining.rules.pix_rate is None
    returns.unlink()
    without_pdfs = runtime.refresh()
    assert without_pdfs.policies is None
    assert without_pdfs.capabilities()["policies"] is False
    assert without_pdfs.capabilities()["catalog"] is True


def test_restart_reuses_metadata_and_persisted_vectors_without_provider_calls(
    tmp_path: Path,
) -> None:
    """Restore a complete generation from real SQLite cache with a fake hosted client."""
    settings = _settings(tmp_path)
    _categories(settings.data_dir)
    _payments(settings.data_dir / "payments.pdf")
    cache = settings.runtime_dir / "vectors.sqlite3"
    client = RecordingClient()
    config = EmbeddingConfig(dimensions=2)
    embedder = HFEmbedder(cache, client=client, config=config)
    first = StoreRuntime(settings, embedder=embedder, tokenizer=CountingTokenizer()).refresh()
    assert len(client.calls) == 1
    client.fail = True
    tokenizer = CountingTokenizer()
    tokenizer.fail = True
    restored = StoreRuntime(
        settings, embedder=HFEmbedder(cache, client=client, config=config), tokenizer=tokenizer
    ).refresh()
    assert restored.generation_id == first.generation_id
    assert tokenizer.calls == 0
    assert len(client.calls) == 1
    assert restored.rules.pix_rate == Decimal("0.05")
    assert _scalar(_database(settings), "SELECT COUNT(*) FROM categories") == 1


@pytest.mark.parametrize("empty_chunks", [False, True])
def test_corrupt_metadata_is_rebuilt_and_repaired(tmp_path: Path, empty_chunks: bool) -> None:
    """A bad persisted cache cannot block valid PDFs or remain broken across restarts."""
    settings, runtime, _, _ = _runtime(tmp_path)
    first = runtime.refresh()
    metadata = settings.runtime_dir / f"policies-{first.generation_id}.json"
    if empty_chunks:
        content = cast(dict[str, object], json.loads(metadata.read_text(encoding="utf-8")))
        content["chunks"] = []
        metadata.write_text(json.dumps(content), encoding="utf-8")
    else:
        metadata.write_text("invalid JSON", encoding="utf-8")
    tokenizer = CountingTokenizer()
    restored = StoreRuntime(settings, embedder=RecordingEmbedder(), tokenizer=tokenizer).refresh()
    assert restored.generation_id == first.generation_id
    assert tokenizer.calls > 0
    repaired = cast(dict[str, object], json.loads(metadata.read_text(encoding="utf-8")))
    assert repaired["chunks"]
    blocked_tokenizer = CountingTokenizer()
    blocked_tokenizer.fail = True
    StoreRuntime(settings, embedder=RecordingEmbedder(), tokenizer=blocked_tokenizer).refresh()
    assert blocked_tokenizer.calls == 0


def test_updated_financial_rules_and_conflicting_pdf_never_keep_old_discount(
    tmp_path: Path,
) -> None:
    """Policy values and their conflicts come from the current PDF generation."""
    settings, runtime, _, _ = _runtime(tmp_path)
    first = runtime.refresh()
    _payments(settings.data_dir / "policies.pdf", "7.5")
    updated = runtime.refresh()
    assert first.rules.quote(Decimal("100.00")).total == Decimal("95.00")
    assert updated.rules.quote(Decimal("100.00")).total == Decimal("92.50")
    conflict = settings.data_dir / "additional.pdf"
    _payments(conflict, "9")
    conflicting = runtime.refresh()
    assert conflicting.rules.conflicts
    with pytest.raises(PolicyUnavailableError, match="Conflicting"):
        conflicting.rules.quote(Decimal("100.00"))
    (settings.data_dir / "policies.pdf").unlink()
    resolved = runtime.refresh()
    assert not resolved.rules.conflicts
    assert resolved.rules.quote(Decimal("100.00")).total == Decimal("91.00")
