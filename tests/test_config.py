"""Protect source-directory boundaries and credential diagnostics."""

from pathlib import Path

import pytest

from emporio_agent.config import Settings
from emporio_agent.embeddings import DEFAULT_EMBEDDING_MODEL, DEFAULT_QUERY_PREFIX


@pytest.fixture(autouse=True)
def clean_settings_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep deployment tests independent of the developer's real credentials and profile."""
    for name in (
        "ANTHROPIC_API_KEY",
        "HF_TOKEN",
        "CLAUDE_MODEL",
        "DATA_DIR",
        "RUNTIME_DIR",
        "EMBEDDING_MODEL",
        "EMBEDDING_PROVIDER",
        "EMBEDDING_DIMENSIONS",
        "EMBEDDING_QUERY_PREFIX",
        "EMBEDDING_DOCUMENT_PREFIX",
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
    settings = Settings(tmp_path / "data", tmp_path / "runtime", "secret-claude", "secret-hf")
    assert "secret-" not in repr(settings)


def test_default_deployment_profile_is_explicit() -> None:
    """Preserve the approved providers, model format and initial execution settings."""
    settings = Settings.from_env()
    assert settings.model == "claude-haiku-5-5"
    assert settings.embedding.model == DEFAULT_EMBEDDING_MODEL
    assert settings.embedding.provider == "deepinfra"
    assert settings.embedding.dimensions == 1024
    assert settings.embedding.query_prefix == DEFAULT_QUERY_PREFIX
    assert settings.embedding.document_prefix == ""
    assert settings.chunking.target_tokens == 600
    assert settings.chunking.max_tokens == 800
    assert settings.chunking.overlap_tokens == 80
    assert settings.tokenizer_model is None
    assert settings.tokenizer_revision is None
    assert settings.store_timezone == "America/Campo_Grande"
    assert settings.claude_timeout_seconds == 60
    assert settings.embedding_timeout_seconds == 30
    assert settings.max_output_tokens == 2048
    assert settings.max_tool_calls == 6


def test_custom_deployment_profile_reads_all_environment_values(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Expose model/profile settings without losing prefixes or leaking API credentials."""
    overrides = {
        "ANTHROPIC_API_KEY": "secret-claude",
        "HF_TOKEN": "secret-hf",
        "CLAUDE_MODEL": "claude-custom-model",
        "DATA_DIR": str(tmp_path / "custom-inputs"),
        "RUNTIME_DIR": str(tmp_path / "custom-runtime"),
        "EMBEDDING_MODEL": "organization/custom-model",
        "EMBEDDING_PROVIDER": "hf-inference",
        "EMBEDDING_DIMENSIONS": "384",
        "EMBEDDING_QUERY_PREFIX": r"query:\nText:",
        "EMBEDDING_DOCUMENT_PREFIX": r"passage:\nText:",
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
    assert settings.embedding.model == "organization/custom-model"
    assert settings.embedding.provider == "hf-inference"
    assert settings.embedding.dimensions == 384
    assert settings.embedding.query_prefix == "query:\nText:"
    assert settings.embedding.document_prefix == "passage:\nText:"
    assert settings.chunking.target_tokens == 256
    assert settings.chunking.max_tokens == 384
    assert settings.chunking.overlap_tokens == 0
    assert settings.tokenizer_model == "organization/custom-tokenizer"
    assert settings.tokenizer_revision == "revision-42"
    assert settings.store_timezone == "UTC"
    assert settings.claude_timeout_seconds == 12.5
    assert settings.embedding_timeout_seconds == 4.5
    assert settings.max_output_tokens == 600
    assert settings.max_tool_calls == 3
    assert settings.anthropic_key == "secret-claude"
    assert settings.hf_token == "secret-hf"
    assert "secret-" not in repr(settings)


def test_a_custom_model_does_not_inherit_qwen_query_instructions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep an unset custom-model prefix distinct from the default Qwen retrieval format."""
    monkeypatch.setenv("EMBEDDING_MODEL", "organization/plain-model")
    assert Settings.from_env().embedding.query_prefix == ""


def test_explicitly_empty_prefixes_override_model_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    """Let plain-text models opt out rather than restoring prefixes through truthiness fallback."""
    monkeypatch.setenv("EMBEDDING_QUERY_PREFIX", "")
    monkeypatch.setenv("EMBEDDING_DOCUMENT_PREFIX", "")
    monkeypatch.setenv("EMBEDDING_TOKENIZER_MODEL", " ")
    monkeypatch.setenv("EMBEDDING_TOKENIZER_REVISION", "")
    settings = Settings.from_env()
    assert settings.embedding.query_prefix == ""
    assert settings.embedding.document_prefix == ""
    assert settings.tokenizer_model is None
    assert settings.tokenizer_revision is None


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("EMBEDDING_DIMENSIONS", "0"),
        ("EMBEDDING_DIMENSIONS", "-1"),
        ("EMBEDDING_DIMENSIONS", "1.5"),
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
    "name", ["CLAUDE_MODEL", "EMBEDDING_MODEL", "EMBEDDING_PROVIDER", "STORE_TIMEZONE", "DATA_DIR"]
)
def test_required_settings_reject_empty_overrides(
    monkeypatch: pytest.MonkeyPatch, name: str
) -> None:
    """An explicit invalid value must not silently select a default deployment profile."""
    monkeypatch.setenv(name, " ")
    with pytest.raises(ValueError, match=name):
        Settings.from_env()


@pytest.mark.parametrize("provider", ["auto", "AUTO"])
def test_embedding_provider_must_be_explicit(
    monkeypatch: pytest.MonkeyPatch, provider: str
) -> None:
    """Prevent the API client from choosing a different provider behind the configured profile."""
    monkeypatch.setenv("EMBEDDING_PROVIDER", provider)
    with pytest.raises(ValueError, match="EMBEDDING_PROVIDER"):
        Settings.from_env()


@pytest.mark.parametrize("timezone", ["Mars/Olympus", "../UTC"])
def test_invalid_store_timezones_are_configuration_errors(
    monkeypatch: pytest.MonkeyPatch, timezone: str
) -> None:
    """Show one actionable environment error for absent or malformed time-zone names."""
    monkeypatch.setenv("STORE_TIMEZONE", timezone)
    with pytest.raises(ValueError, match="STORE_TIMEZONE"):
        Settings.from_env()
