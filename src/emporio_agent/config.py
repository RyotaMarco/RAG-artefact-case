"""Validated environment settings without secrets in application output."""

import os
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    """Keep one explicit configuration for a local application instance."""

    data_dir: Path
    runtime_dir: Path
    anthropic_key: str = field(default="", repr=False)
    hf_token: str = field(default="", repr=False)
    model: str = "claude-sonnet-5-5"

    @classmethod
    def from_env(cls) -> "Settings":
        """Read settings and reject generated files inside the input folder.

        Returns:
            Validated settings. Missing credentials remain an explicit UI state.

        Raises:
            ValueError: A directory setting is invalid.
        """
        data = Path(os.environ.get("DATA_DIR", "data")).resolve()
        runtime = Path(os.environ.get("RUNTIME_DIR", "runtime")).resolve()
        if data == runtime or runtime.is_relative_to(data):
            raise ValueError("RUNTIME_DIR must be outside DATA_DIR.")
        return cls(
            data_dir=data,
            runtime_dir=runtime,
            anthropic_key=os.environ.get("ANTHROPIC_API_KEY", ""),
            hf_token=os.environ.get("HF_TOKEN", ""),
            model=os.environ.get("CLAUDE_MODEL", "claude-sonnet-5-5"),
        )
