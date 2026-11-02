"""Protect source-directory boundaries and credential diagnostics."""

from pathlib import Path

import pytest

from emporio_agent.config import Settings
from emporio_agent.embeddings import DEFAULT_EMBEDDING_MODEL


@pytest.fixture(autouse=True)
def clean_settings_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep deployment tests independent of the developer's real credentials and profile."""
    for name in (
        "ANTHROPIC_API_KEY",
        "VOYAGE_API_KEY",
        "CLAUDE_MODEL",
        "DATA_DIR",
        "RUNTIME_DIR",
        "EMBEDDING_MODEL",
        "EMBEDDING_BASE_URL",
        "EMBEDDING_DIMENSIONS",
        "EMBEDDING_REQUESTS_PER_MINUTE",
        "EMBEDDING_TOKENS_PER_MINUTE",
        "EMBEDDING_BATCH_TOKENS",
        "EMBEDDING_TOKENIZER_MODEL",
        "EMBEDDING_TOKENIZER_REVISION",
        "STORE_TIMEZONE",
        "CLAUDE_TIMEOUT_SECONDS",
        "EMBEDDING_TIMEOUT_SECONDS",
        "CLAUDE_MAX_TOKENS",
        "MAX_TOOL_CALLS",
        "CHUNK_TARGET_TOKENS",
        "CHUNK_MAX_TOKENS",
        "CHUNK_OVERLAP_TOKENS",
    ):
        monkeypatch.delenv(name, raising=False)


@pytest.mark.parametrize("runtime", ["inputs", "inputs/generated"])
def test_generated_files_never_enter_source_inventory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, runtime: str
) -> None:
    """Reject both identical directories and nested output paths before ingestion."""
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "inputs"))
    monkeypatch.setenv("RUNTIME_DIR", str(tmp_path / runtime))
    with pytest.raises(ValueError, match="outside DATA_DIR"):
        Settings.from_env()


def test_configuration_diagnostics_do_not_expose_credentials(tmp_path: Path) -> None:
    """Secrets never appear when a settings object is printed for troubleshooting."""
    settings = Settings(tmp_path / "data", tmp_path / "runtime", "secret-claude", "secret-voyage")
    assert "secret-" not in repr(settings)


def test_default_deployment_profile_is_explicit() -> None:
    """Preserve the approved model, endpoint and initial execution settings."""
    settings = Settings.from_env()
    assert settings.model == "claude-haiku-5-5"
    assert settings.embedding.model == DEFAULT_EMBEDDING_MODEL
    assert settings.embedding.base_url == "https://ai.mongodb.com/v1"
    assert settings.embedding.dimensions == 1024
    assert settings.chunking.target_tokens == 600
    assert settings.chunking.max_tokens == 800
    assert settings.chunking.overlap_tokens == 80
    assert settings.tokenizer_model is None
    assert settings.tokenizer_revision is None
    assert settings.store_timezone == "America/Campo_Grande"
    assert settings.claude_timeout_seconds == 60
    assert settings.embedding_timeout_seconds == 30
    assert settings.embedding_requests_per_minute == 3
    assert settings.embedding_tokens_per_minute == 10000
    assert settings.embedding_batch_tokens == 8000
    assert settings.max_output_tokens == 2048
    assert settings.max_tool_calls == 6


def test_custom_deployment_profile_reads_all_environment_values(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Expose endpoint, rate and model settings without leaking API credentials."""
    overrides = {
        "ANTHROPIC_API_KEY": "secret-claude",
        "VOYAGE_API_KEY": "secret-voyage",
        "CLAUDE_MODEL": "claude-custom-model",
        "DATA_DIR": str(tmp_path / "custom-inputs"),
        "RUNTIME_DIR": str(tmp_path / "custom-runtime"),
        "EMBEDDING_MODEL": "custom-voyage-model",
        "EMBEDDING_BASE_URL": "https://embeddings.example.test:8443/v1/",
        "EMBEDDING_DIMENSIONS": "384",
        "EMBEDDING_REQUESTS_PER_MINUTE": "7",
        "EMBEDDING_TOKENS_PER_MINUTE": "12000",
        "EMBEDDING_BATCH_TOKENS": "6000",
        "EMBEDDING_TOKENIZER_MODEL": "organization/custom-tokenizer",
        "EMBEDDING_TOKENIZER_REVISION": "revision-42",
        "STORE_TIMEZONE": "UTC",
        "CLAUDE_TIMEOUT_SECONDS": "12.5",
        "EMBEDDING_TIMEOUT_SECONDS": "4.5",
        "CLAUDE_MAX_TOKENS": "600",
        "MAX_TOOL_CALLS": "3",
        "CHUNK_TARGET_TOKENS": "256",
        "CHUNK_MAX_TOKENS": "384",
        "CHUNK_OVERLAP_TOKENS": "0",
    }
    for name, value in overrides.items():
        monkeypatch.setenv(name, value)
    settings = Settings.from_env()
    assert settings.data_dir == (tmp_path / "custom-inputs").resolve()
    assert settings.runtime_dir == (tmp_path / "custom-runtime").resolve()
    assert settings.model == "claude-custom-model"
    assert settings.embedding.model == "custom-voyage-model"
    assert settings.embedding.base_url == "https://embeddings.example.test:8443/v1"
    assert settings.embedding.dimensions == 384
    assert settings.chunking.target_tokens == 256
    assert settings.chunking.max_tokens == 384
    assert settings.chunking.overlap_tokens == 0
    assert settings.tokenizer_model == "organization/custom-tokenizer"
    assert settings.tokenizer_revision == "revision-42"
    assert settings.store_timezone == "UTC"
    assert settings.claude_timeout_seconds == 12.5
    assert settings.embedding_timeout_seconds == 4.5
    assert settings.embedding_requests_per_minute == 7
    assert settings.embedding_tokens_per_minute == 12000
    assert settings.embedding_batch_tokens == 6000
    assert settings.max_output_tokens == 600
    assert settings.max_tool_calls == 3
    assert settings.anthropic_key == "secret-claude"
    assert settings.voyage_key == "secret-voyage"
    assert "secret-" not in repr(settings)


def test_old_hf_credentials_cannot_authorize_the_new_endpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Require the explicit Voyage credential instead of reusing an unrelated service's key."""
    monkeypatch.setenv("HF_TOKEN", "old-service-secret")
    assert Settings.from_env().voyage_key == ""


def test_blank_tokenizer_settings_use_the_default_profile(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep empty optional overrides available for the pinned default tokenizer profile."""
    monkeypatch.setenv("EMBEDDING_TOKENIZER_MODEL", " ")
    monkeypatch.setenv("EMBEDDING_TOKENIZER_REVISION", "")
    settings = Settings.from_env()
    assert settings.tokenizer_model is None
    assert settings.tokenizer_revision is None


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("EMBEDDING_DIMENSIONS", "0"),
        ("EMBEDDING_DIMENSIONS", "-1"),
        ("EMBEDDING_DIMENSIONS", "1.5"),
        ("EMBEDDING_REQUESTS_PER_MINUTE", "0"),
        ("EMBEDDING_REQUESTS_PER_MINUTE", "-1"),
        ("EMBEDDING_REQUESTS_PER_MINUTE", "1.5"),
        ("EMBEDDING_TOKENS_PER_MINUTE", "0"),
        ("EMBEDDING_TOKENS_PER_MINUTE", "-1"),
        ("EMBEDDING_TOKENS_PER_MINUTE", "invalid"),
        ("EMBEDDING_BATCH_TOKENS", "0"),
        ("EMBEDDING_BATCH_TOKENS", "-1"),
        ("EMBEDDING_BATCH_TOKENS", "inf"),
        ("CLAUDE_MAX_TOKENS", "0"),
        ("CLAUDE_MAX_TOKENS", "-1"),
        ("CLAUDE_MAX_TOKENS", "invalid"),
        ("MAX_TOOL_CALLS", "0"),
        ("MAX_TOOL_CALLS", "-1"),
        ("MAX_TOOL_CALLS", "1.5"),
        ("CLAUDE_TIMEOUT_SECONDS", "0"),
        ("CLAUDE_TIMEOUT_SECONDS", "-1"),
        ("CLAUDE_TIMEOUT_SECONDS", "nan"),
        ("CLAUDE_TIMEOUT_SECONDS", "inf"),
        ("CLAUDE_TIMEOUT_SECONDS", "invalid"),
        ("EMBEDDING_TIMEOUT_SECONDS", "0"),
        ("EMBEDDING_TIMEOUT_SECONDS", "-1"),
        ("EMBEDDING_TIMEOUT_SECONDS", "nan"),
        ("EMBEDDING_TIMEOUT_SECONDS", "-inf"),
        ("EMBEDDING_TIMEOUT_SECONDS", "invalid"),
        ("CHUNK_TARGET_TOKENS", "0"),
        ("CHUNK_MAX_TOKENS", "-1"),
        ("CHUNK_OVERLAP_TOKENS", "-1"),
        ("CHUNK_OVERLAP_TOKENS", "1.5"),
    ],
)
def test_invalid_numeric_settings_fail_at_the_environment_boundary(
    monkeypatch: pytest.MonkeyPatch, name: str, value: str
) -> None:
    """Reject invalid dimensions, limits and timeouts before an external request is possible."""
    monkeypatch.setenv(name, value)
    with pytest.raises(ValueError, match=name):
        Settings.from_env()


@pytest.mark.parametrize(("target", "maximum", "overlap"), [(500, 400, 80), (100, 400, 100)])
def test_chunk_window_rejects_inconsistent_limits(
    monkeypatch: pytest.MonkeyPatch, target: int, maximum: int, overlap: int
) -> None:
    """Reject a target larger than the ceiling or an overlap that cannot make progress."""
    monkeypatch.setenv("CHUNK_TARGET_TOKENS", str(target))
    monkeypatch.setenv("CHUNK_MAX_TOKENS", str(maximum))
    monkeypatch.setenv("CHUNK_OVERLAP_TOKENS", str(overlap))
    with pytest.raises(ValueError, match="(?i)chunk|overlap|target"):
        Settings.from_env()


@pytest.mark.parametrize(
    "name", ["CLAUDE_MODEL", "EMBEDDING_MODEL", "EMBEDDING_BASE_URL", "STORE_TIMEZONE", "DATA_DIR"]
)
def test_required_settings_reject_empty_overrides(
    monkeypatch: pytest.MonkeyPatch, name: str
) -> None:
    """An explicit invalid value must not silently select a default deployment profile."""
    monkeypatch.setenv(name, " ")
    with pytest.raises(ValueError, match=name):
        Settings.from_env()


@pytest.mark.parametrize(
    "url",
    [
        "http://ai.mongodb.com/v1",
        "https://user:secret@ai.mongodb.com/v1",
        "https://@ai.mongodb.com/v1",
        "https://ai.mongodb.com/v1?api_key=secret",
        "https://ai.mongodb.com/v1#secret",
        "https://ai.mongodb.com/v1?",
        "https://ai.mongodb.com/v1#",
        "https:///v1",
        "https://ai.mongodb.com:invalid/v1",
        "https://ai.mongodb.com:70000/v1",
        "https://[invalid/v1",
        "https://ai.mongodb.com/\npath",
    ],
)
def test_embedding_endpoint_rejects_insecure_or_ambiguous_urls(
    monkeypatch: pytest.MonkeyPatch, url: str
) -> None:
    """Require HTTPS and keep credentials out of endpoints, query strings and diagnostics."""
    monkeypatch.setenv("EMBEDDING_BASE_URL", url)
    with pytest.raises(ValueError, match="EMBEDDING_BASE_URL") as captured:
        Settings.from_env()
    assert "secret" not in str(captured.value)


@pytest.mark.parametrize("timezone", ["Mars/Olympus", "../UTC"])
def test_invalid_store_timezones_are_configuration_errors(
    monkeypatch: pytest.MonkeyPatch, timezone: str
) -> None:
    """Show one actionable environment error for absent or malformed time-zone names."""
    monkeypatch.setenv("STORE_TIMEZONE", timezone)
    with pytest.raises(ValueError, match="STORE_TIMEZONE"):
        Settings.from_env()
