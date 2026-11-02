"""Validated environment settings without secrets in application output."""

import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from emporio_agent.documents import ChunkConfig
from emporio_agent.embeddings import (
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_QUERY_PREFIX,
    EmbeddingConfig,
)


def _nonempty(name: str, default: str) -> str:
    """Read a required deployment value without silently accepting an empty override."""
    value = os.environ.get(name, default).strip()
    if not value:
        raise ValueError(f"{name} must not be empty.")
    return value


def _integer_setting(name: str, default: int, *, allow_zero: bool = False) -> int:
    """Parse a positive limit or nonnegative overlap at the environment boundary."""
    description = "nonnegative" if allow_zero else "positive"
    try:
        value = int(os.environ.get(name, str(default)))
    except ValueError as error:
        raise ValueError(f"{name} must be a {description} integer.") from error
    if value < (0 if allow_zero else 1):
        raise ValueError(f"{name} must be a {description} integer.")
    return value


def _positive_float(name: str, default: float) -> float:
    """Parse a finite positive timeout without permitting NaN or infinity."""
    try:
        value = float(os.environ.get(name, str(default)))
    except ValueError as error:
        raise ValueError(f"{name} must be a finite positive number.") from error
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a finite positive number.")
    return value


def _embedding_settings(default: EmbeddingConfig) -> EmbeddingConfig:
    """Read one explicit model profile, preserving intentionally empty retrieval prefixes."""
    model = _nonempty("EMBEDDING_MODEL", default.model)
    provider = _nonempty("EMBEDDING_PROVIDER", default.provider).casefold()
    if provider == "auto":
        raise ValueError("EMBEDDING_PROVIDER must select an explicit provider, not auto.")
    query_prefix = DEFAULT_QUERY_PREFIX if model == DEFAULT_EMBEDDING_MODEL else ""
    return EmbeddingConfig(
        model=model,
        provider=provider,
        dimensions=_integer_setting("EMBEDDING_DIMENSIONS", default.dimensions),
        query_prefix=os.environ.get("EMBEDDING_QUERY_PREFIX", query_prefix).replace(r"\n", "\n"),
        document_prefix=os.environ.get(
            "EMBEDDING_DOCUMENT_PREFIX", default.document_prefix
        ).replace(r"\n", "\n"),
    )


def _store_timezone(default: str) -> str:
    """Validate the configured IANA time zone before creating any model client."""
    value = _nonempty("STORE_TIMEZONE", default)
    try:
        ZoneInfo(value)
    except (ValueError, ZoneInfoNotFoundError) as error:
        raise ValueError("STORE_TIMEZONE must identify an available IANA time zone.") from error
    return value


def _chunk_settings(default: ChunkConfig) -> ChunkConfig:
    """Read the model-specific text window without silently clipping a supplied document."""
    return ChunkConfig(
        target_tokens=_integer_setting("CHUNK_TARGET_TOKENS", default.target_tokens),
        max_tokens=_integer_setting("CHUNK_MAX_TOKENS", default.max_tokens),
        overlap_tokens=_integer_setting(
            "CHUNK_OVERLAP_TOKENS", default.overlap_tokens, allow_zero=True
        ),
    )


@dataclass(frozen=True)
class Settings:
    """Keep one explicit configuration for a local application instance."""

    data_dir: Path
    runtime_dir: Path
    anthropic_key: str = field(default="", repr=False)
    hf_token: str = field(default="", repr=False)
    model: str = "claude-haiku-5-5"
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    chunking: ChunkConfig = field(default_factory=ChunkConfig)
    tokenizer_model: str | None = None
    tokenizer_revision: str | None = None
    store_timezone: str = "America/Campo_Grande"
    claude_timeout_seconds: float = 60
    embedding_timeout_seconds: float = 30
    max_output_tokens: int = 2048
    max_tool_calls: int = 6

    @classmethod
    def from_env(cls) -> "Settings":
        """Validate deployment settings and keep generated files outside the input folder.

        Returns:
            Validated settings. Missing credentials remain an explicit UI state.

        Raises:
            ValueError: A deployment value is invalid or source/output paths overlap.
        """
        data = Path(_nonempty("DATA_DIR", "data")).resolve()
        runtime = Path(_nonempty("RUNTIME_DIR", "runtime")).resolve()
        if data == runtime or runtime.is_relative_to(data):
            raise ValueError("RUNTIME_DIR must be outside DATA_DIR.")
        defaults = cls(data, runtime)
        return cls(
            data_dir=data,
            runtime_dir=runtime,
            anthropic_key=os.environ.get("ANTHROPIC_API_KEY", ""),
            hf_token=os.environ.get("HF_TOKEN", ""),
            model=_nonempty("CLAUDE_MODEL", defaults.model),
            embedding=_embedding_settings(defaults.embedding),
            chunking=_chunk_settings(defaults.chunking),
            tokenizer_model=os.environ.get("EMBEDDING_TOKENIZER_MODEL", "").strip() or None,
            tokenizer_revision=os.environ.get("EMBEDDING_TOKENIZER_REVISION", "").strip() or None,
            store_timezone=_store_timezone(defaults.store_timezone),
            claude_timeout_seconds=_positive_float(
                "CLAUDE_TIMEOUT_SECONDS", defaults.claude_timeout_seconds
            ),
            embedding_timeout_seconds=_positive_float(
                "EMBEDDING_TIMEOUT_SECONDS", defaults.embedding_timeout_seconds
            ),
            max_output_tokens=_integer_setting("CLAUDE_MAX_TOKENS", defaults.max_output_tokens),
            max_tool_calls=_integer_setting("MAX_TOOL_CALLS", defaults.max_tool_calls),
        )
