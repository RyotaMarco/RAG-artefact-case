"""Check the paid scenario recorder offline, with the native provider HTTP stubbed."""

from pathlib import Path

import pytest
from agent_fixtures import ClaudeStub, small_snapshot, text_response, tool_response

from emporio_agent.config import Settings
from scripts.run_scenarios import SCENARIOS, Scenario, main, render, require_live, run_scenario


def test_scenarios_cover_the_required_cases() -> None:
    """Eight to ten separate conversations, unique names and at least one multi-turn case."""
    assert 8 <= len(SCENARIOS) <= 10
    assert len({s.number for s in SCENARIOS}) == len({s.slug for s in SCENARIOS}) == len(SCENARIOS)
    assert any(len(s.turns) >= 3 for s in SCENARIOS)
    assert all(s.summary.isascii() and s.expectation.isascii() for s in SCENARIOS)


@pytest.mark.parametrize(
    ("environ", "anthropic", "voyage", "message"),
    [
        ({}, "key", "key", "RUN_LIVE_TESTS=1"),
        ({"RUN_LIVE_TESTS": "true"}, "key", "key", "RUN_LIVE_TESTS=1"),
        ({"RUN_LIVE_TESTS": "1"}, "", "key", "both required"),
        ({"RUN_LIVE_TESTS": "1"}, "key", "", "both required"),
    ],
)
def test_paid_run_requires_explicit_opt_in_and_both_keys(
    tmp_path: Path, environ: dict[str, str], anthropic: str, voyage: str, message: str
) -> None:
    """No paid request starts without the flag and both provider keys."""
    settings = Settings(tmp_path, tmp_path, anthropic_key=anthropic, voyage_key=voyage)
    with pytest.raises(SystemExit, match=message):
        require_live(settings, environ)
    require_live(
        Settings(tmp_path, tmp_path, anthropic_key="k", voyage_key="k"), {"RUN_LIVE_TESTS": "1"}
    )


def test_main_refuses_before_loading_sources(monkeypatch: pytest.MonkeyPatch) -> None:
    """The command line exits before any provider or source work without the opt-in."""
    monkeypatch.delenv("RUN_LIVE_TESTS", raising=False)
    with pytest.raises(SystemExit, match="RUN_LIVE_TESTS=1"):
        main(["--only", "01"])


def test_transcript_records_answers_tools_and_sources(tmp_path: Path) -> None:
    """Each turn keeps the customer text, final answer, tool calls and cited sources."""
    scenario = Scenario("99", "fixture", "Fixture scenario.", "Fixture check.", ("Preço?", "Ok"))
    stub = ClaudeStub(
        [
            tool_response("query_data", {"sql": "SELECT name, price_cents FROM products"}),
            text_response("O Demo Guitar sai por R$ 100,01."),
            text_response("Posso ajudar em algo mais?"),
        ]
    )
    settings = Settings(tmp_path / "data", tmp_path / "runtime", anthropic_key="fixture-key")
    turns = run_scenario(settings, small_snapshot(), scenario, transport=stub.transport)
    assert [turn.answer for turn in turns] == [
        "O Demo Guitar sai por R$ 100,01.",
        "Posso ajudar em algo mais?",
    ]
    assert turns[0].tool_calls == ['query_data({"sql": "SELECT name, price_cents FROM products"})']
    assert turns[0].sources == ["instruments.csv"]
    assert turns[1].tool_calls == turns[1].sources == []
    markdown = render(scenario, turns, "fixture-model", "2026-11-02T10:00-04:00")
    assert markdown.startswith("# 99. fixture\n")
    assert "**Scenario:** Fixture scenario." in markdown
    assert "> Preço?" in markdown
    assert "- `query_data(" in markdown
    assert "- instruments.csv" in markdown
    assert markdown.count("## Turn ") == 2
    assert markdown.endswith("- none\n")
