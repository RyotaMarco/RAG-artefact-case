"""Verify the required owner-only Development promotion workflow."""

from collections.abc import Mapping

import pytest

from scripts.check_branch_flow import check_event, validate_branch_flow

REPOSITORY = "RyotaMarco/RAG-artefact-case"


@pytest.mark.parametrize("prefix", ["feature", "test", "fix", "chore"])
def test_task_branches_enter_development(prefix: str) -> None:
    """Accept each supported task category into Development."""
    validate_branch_flow(
        f"{prefix}/data-ingestion",
        "Development",
        head_repository=REPOSITORY,
        base_repository=REPOSITORY,
        actor="RyotaMarco",
        owner="RyotaMarco",
    )


def test_development_promotes_to_main() -> None:
    """Accept the sole allowed source of main changes."""
    validate_branch_flow(
        "Development",
        "main",
        head_repository=REPOSITORY,
        base_repository=REPOSITORY,
        actor="ryotamarco",
        owner="RyotaMarco",
    )


@pytest.mark.parametrize(
    ("head", "base"),
    [
        ("feature/chat", "main"),
        ("test/unit", "main"),
        ("main", "Development"),
        ("Development", "feature/chat"),
        ("development", "main"),
        ("Development", "Development"),
        ("feature/", "Development"),
        ("unscoped-work", "Development"),
    ],
)
def test_invalid_directions_are_rejected(head: str, base: str) -> None:
    """Reject shortcuts and incorrect branch names."""
    with pytest.raises(ValueError, match="Allowed flow"):
        validate_branch_flow(
            head,
            base,
            head_repository=REPOSITORY,
            base_repository=REPOSITORY,
            actor="RyotaMarco",
            owner="RyotaMarco",
        )


def test_fork_cannot_impersonate_development() -> None:
    """A matching branch name from another repository is insufficient."""
    with pytest.raises(ValueError, match="this repository"):
        validate_branch_flow(
            "Development",
            "main",
            head_repository="another-owner/RAG-artefact-case",
            base_repository=REPOSITORY,
            actor="RyotaMarco",
            owner="RyotaMarco",
        )


def test_other_actor_is_rejected() -> None:
    """Do not accept changes initiated by an additional contributor."""
    with pytest.raises(ValueError, match="owner's account"):
        validate_branch_flow(
            "Development",
            "main",
            head_repository=REPOSITORY,
            base_repository=REPOSITORY,
            actor="someone-else",
            owner="RyotaMarco",
        )


def test_pull_request_event_checks_repository_context() -> None:
    """Read a genuine-shaped event and reject a mismatched destination."""
    event: Mapping[str, object] = {
        "pull_request": {
            "head": {"ref": "Development", "repo": {"full_name": REPOSITORY}},
            "base": {"ref": "main", "repo": {"full_name": REPOSITORY}},
        }
    }
    check_event(
        event,
        event_name="pull_request",
        repository=REPOSITORY,
        owner="RyotaMarco",
        actor="RyotaMarco",
    )
    with pytest.raises(ValueError, match="destination"):
        check_event(
            event,
            event_name="pull_request",
            repository="RyotaMarco/a-different-repository",
            owner="RyotaMarco",
            actor="RyotaMarco",
        )


@pytest.mark.parametrize("branch", ["main", "Development", "feature/ingestion", "test/e2e"])
def test_allowed_push_events(branch: str) -> None:
    """Allow checks after accepted pushes without pretending to enforce GitHub protection."""
    check_event(
        {"ref": f"refs/heads/{branch}"},
        event_name="push",
        repository=REPOSITORY,
        owner="RyotaMarco",
        actor="RyotaMarco",
    )


@pytest.mark.parametrize("event", [{}, {"pull_request": []}, {"pull_request": {"head": 3}}])
def test_invalid_payloads_fail_closed(event: Mapping[str, object]) -> None:
    """Reject malformed metadata rather than skipping the flow check."""
    with pytest.raises(ValueError):
        check_event(
            event,
            event_name="pull_request",
            repository=REPOSITORY,
            owner="RyotaMarco",
            actor="RyotaMarco",
        )
