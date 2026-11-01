"""Protect source-directory boundaries and credential diagnostics."""

from pathlib import Path

import pytest

from emporio_agent.config import Settings


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
