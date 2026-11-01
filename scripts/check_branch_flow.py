"""Check pull request direction while GitHub protections enforce direct-push restrictions."""

from __future__ import annotations

import json
import os
import re
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import cast

TASK_BRANCH = re.compile(r"(?:feature|test|fix|chore)/[a-z0-9][a-z0-9._/-]*")


def validate_branch_flow(
    head: str,
    base: str,
    *,
    head_repository: str,
    base_repository: str,
    actor: str,
    owner: str,
) -> None:
    """Require owner-authored task branches to pass through Development.

    Parameters:
        head: Pull request source branch.
        base: Pull request destination branch.
        head_repository: Full source repository name.
        base_repository: Full destination repository name.
        actor: Account that triggered this check.
        owner: Repository owner's account.

    Raises:
        ValueError: The pull request violates ownership or branch direction.
    """
    if actor.casefold() != owner.casefold():
        raise ValueError("Only the repository owner's account may contribute.")
    if head_repository.casefold() != base_repository.casefold():
        raise ValueError("Pull requests must originate in this repository.")
    if base == "main" and head == "Development":
        return
    if base == "Development" and TASK_BRANCH.fullmatch(head):
        return
    raise ValueError("Allowed flow: feature/test/fix/chore branches -> Development -> main.")


def _object(value: object, label: str) -> Mapping[str, object]:
    """Read a JSON object without trusting its field values."""
    if not isinstance(value, dict):
        raise ValueError(f"Missing or invalid {label} object.")
    return cast(Mapping[str, object], value)


def _text(value: object, label: str) -> str:
    """Require a nonempty string from an event field."""
    if not isinstance(value, str) or not value:
        raise ValueError(f"Missing or invalid {label} field.")
    return value


def check_event(
    event: Mapping[str, object], *, event_name: str, repository: str, owner: str, actor: str
) -> None:
    """Validate an event using repository-provided context.

    Parameters:
        event: Parsed GitHub event payload.
        event_name: GitHub event type.
        repository: Full repository name.
        owner: Repository owner's account.
        actor: Account that triggered the workflow.

    Raises:
        ValueError: Event ownership, shape, or branch direction is invalid.
    """
    if actor.casefold() != owner.casefold():
        raise ValueError("Only the repository owner's account may trigger changes.")
    if event_name == "pull_request":
        request = _object(event.get("pull_request"), "pull_request")
        head = _object(request.get("head"), "head")
        base = _object(request.get("base"), "base")
        head_repo = _object(head.get("repo"), "head repository")
        base_repo = _object(base.get("repo"), "base repository")
        if _text(base_repo.get("full_name"), "base repository name").casefold() != (
            repository.casefold()
        ):
            raise ValueError("The event destination does not match this repository.")
        validate_branch_flow(
            _text(head.get("ref"), "head ref"),
            _text(base.get("ref"), "base ref"),
            head_repository=_text(head_repo.get("full_name"), "head repository name"),
            base_repository=repository,
            actor=actor,
            owner=owner,
        )
        return
    if event_name not in {"push", "workflow_dispatch"}:
        raise ValueError("Unsupported workflow event.")
    ref = _text(event.get("ref"), "ref")
    branch = ref.removeprefix("refs/heads/")
    if branch not in {"main", "Development"} and not TASK_BRANCH.fullmatch(branch):
        raise ValueError("Use main, Development, or a feature/test/fix/chore branch.")


def main() -> int:
    """Return a CI-friendly exit status without printing the event payload."""
    try:
        event_path = Path(os.environ["GITHUB_EVENT_PATH"])
        payload: object = json.loads(event_path.read_text(encoding="utf-8"))
        check_event(
            _object(payload, "event"),
            event_name=os.environ["GITHUB_EVENT_NAME"],
            repository=os.environ["GITHUB_REPOSITORY"],
            owner=os.environ["GITHUB_REPOSITORY_OWNER"],
            actor=os.environ["GITHUB_ACTOR"],
        )
    except (KeyError, OSError, ValueError) as error:
        print(f"Branch flow rejected: {error}", file=sys.stderr)
        return 1
    print("Branch flow accepted.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
