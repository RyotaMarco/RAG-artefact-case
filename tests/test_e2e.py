"""Exercise the real Streamlit, Agno, tools and sessions through a browser without paid APIs."""

from __future__ import annotations

import csv
import os
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path

import pytest
from playwright.sync_api import Browser, Page, expect

from tests.test_documents import pdf_bytes

pytestmark = pytest.mark.e2e
ROOT = Path(__file__).parents[1]


@dataclass(frozen=True)
class RunningStore:
    """Locate one isolated Streamlit process and its private source fixtures."""

    url: str
    data_dir: Path
    runtime_dir: Path


def _write_csv(path: Path, header: list[str], rows: list[list[str]]) -> None:
    """Write a schema-correct source without changing production fixtures."""
    with path.open("w", encoding="utf-8", newline="") as target:
        writer = csv.writer(target)
        writer.writerow(header)
        writer.writerows(rows)


def write_sources(directory: Path, *, price: str = "123.45") -> None:
    """Generate business data with deliberate private order boundaries.

    Parameters:
        directory: Temporary source directory owned by this test.
        price: Exact decimal price to expose through the real SQL tool.
    """
    directory.mkdir(parents=True, exist_ok=True)
    _write_csv(
        directory / "categories-export.CSV",
        ["category_id", "name", "description"],
        [["1", "Acoustic guitars", "Acoustic instruments"]],
    )
    _write_csv(
        directory / "renamed-catalog.csv",
        [
            "product_id",
            "price_brl",
            "name",
            "category_id",
            "description",
            "stock_quantity",
            "status",
            "specs",
            "created_at",
        ],
        [
            [
                "701",
                price,
                "Bluejay Acoustic Guitar",
                "1",
                "A lightweight acoustic instrument",
                "3",
                "active",
                '{"body":"wood"}',
                "2026-01-01",
            ]
        ],
    )
    _write_csv(
        directory / "offers.csv",
        ["promotion_id", "product_id", "discount_percent", "description", "is_active"],
        [],
    )
    _write_csv(
        directory / "contacts.csv",
        ["customer_id", "name", "phone", "email", "city"],
        [
            ["1", "Alex Example", "5550101", "alex@example.test", "Demo city"],
            ["2", "Blair Example", "5550102", "blair@example.test", "Demo city"],
        ],
    )
    _write_csv(
        directory / "purchases.csv",
        [
            "order_id",
            "customer_id",
            "order_date",
            "status",
            "total_brl",
            "payment_method",
            "tracking_code",
            "estimated_delivery",
            "notes",
        ],
        [
            ["7", "1", "2026-01-03", "shipped", "111.11", "pix", "PRIVATE-TRACK-A", "", ""],
            ["8", "2", "2026-01-03", "confirmed", "222.22", "pix", "PRIVATE-TRACK-B", "", ""],
        ],
    )
    _write_csv(
        directory / "lines.csv",
        ["order_id", "quantity", "product_id"],
        [["7", "1", "701"], ["8", "1", "701"]],
    )
    (directory / "store-policies.pdf").write_bytes(
        pdf_bytes(
            [
                "1. Returns",
                "Returns need the original box.",
                "2. Payments",
                "PIX 5% de desconto sobre o pre\u00e7o de tabela.",
                "Promocoes nao sao cumulativas.",
            ]
        )
    )
    (directory / "additional-policy.PDF").write_bytes(
        pdf_bytes(["1. Shipping", "Delivery is available on weekdays."])
    )


def _unused_port() -> int:
    """Ask the operating system for an available local listener port."""
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


@pytest.fixture
def store_server(tmp_path: Path) -> Iterator[Callable[[str], RunningStore]]:
    """Launch test-only wrappers with no inherited API credentials.

    Parameters:
        tmp_path: Pytest-owned temporary directory.

    Yields:
        A process factory whose children are always terminated after the test.
    """
    processes: list[subprocess.Popen[bytes]] = []

    def launch(scenario: str = "normal") -> RunningStore:
        """Start one isolated instance and wait for its actual health endpoint."""
        directory = tmp_path / f"instance-{len(processes)}"
        data_dir = directory / "data"
        runtime_dir = directory / "runtime"
        write_sources(data_dir)
        if scenario == "missing-sources":
            data_dir = directory / "empty-data"
            data_dir.mkdir()
        env = dict(os.environ)
        env.pop("ANTHROPIC_API_KEY", None)
        env.pop("HF_TOKEN", None)
        env["EMPORIO_E2E_DATA_DIR"] = str(data_dir)
        env["EMPORIO_E2E_RUNTIME_DIR"] = str(runtime_dir)
        env["EMPORIO_E2E_SCENARIO"] = scenario
        env["PYTHONPATH"] = os.pathsep.join([str(ROOT / "src"), str(ROOT)])
        env["PYTHONUNBUFFERED"] = "1"
        port = _unused_port()
        url = f"http://127.0.0.1:{port}"
        log_path = directory / "streamlit.log"
        with log_path.open("wb") as output:
            process = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "streamlit",
                    "run",
                    str(ROOT / "tests" / "e2e_app.py"),
                    "--server.address=127.0.0.1",
                    f"--server.port={port}",
                    "--server.headless=true",
                    "--server.fileWatcherType=none",
                    "--browser.gatherUsageStats=false",
                ],
                cwd=ROOT,
                env=env,
                stdout=output,
                stderr=subprocess.STDOUT,
            )
        processes.append(process)
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            if process.poll() is not None:
                break
            try:
                with urllib.request.urlopen(f"{url}/_stcore/health", timeout=1) as response:
                    if response.status == 200:
                        return RunningStore(url, data_dir, runtime_dir)
            except (OSError, urllib.error.URLError):
                time.sleep(0.1)
        raise AssertionError(
            "Test Streamlit server did not become healthy:\n"
            + log_path.read_text(encoding="utf-8", errors="replace")[-6000:]
        )

    yield launch
    for process in processes:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)


def _ask(page: Page, question: str) -> None:
    """Submit through the native chat input, including keyboard activation."""
    entry = page.get_by_placeholder("Ask the store a question")
    expect(entry).to_be_enabled(timeout=20_000)
    entry.fill(question)
    entry.press("Enter")


def test_chat_queries_real_data_and_refreshes_changed_sources(
    page: Page, store_server: Callable[[str], RunningStore]
) -> None:
    """Display exact prices from the SQL tool rather than from a scripted answer constant."""
    server = store_server("normal")
    page.goto(server.url)
    _ask(page, "What is the price of Bluejay Acoustic Guitar?")
    expect(page.get_by_text("Bluejay Acoustic Guitar costs BRL 123.45.", exact=True)).to_be_visible(
        timeout=20_000
    )
    page.get_by_text("Sources consulted", exact=True).click()
    expect(page.get_by_text("renamed-catalog.csv", exact=True)).to_be_visible()
    write_sources(server.data_dir, price="234.56")
    _ask(page, "Check the price again.")
    expect(page.get_by_text("Bluejay Acoustic Guitar costs BRL 234.56.", exact=True)).to_be_visible(
        timeout=20_000
    )


@pytest.mark.parametrize(
    ("question", "answer", "source"),
    [
        ("What is the return policy?", "Returns need the original box.", "store-policies.pdf"),
        (
            "What is the shipping policy?",
            "Delivery is available on weekdays.",
            "additional-policy.PDF",
        ),
    ],
)
def test_pdf_search_displays_citations(
    page: Page,
    store_server: Callable[[str], RunningStore],
    question: str,
    answer: str,
    source: str,
) -> None:
    """Search both generated PDFs with actual hybrid retrieval and show provenance."""
    server = store_server("normal")
    page.goto(server.url)
    _ask(page, question)
    expect(page.get_by_text(answer, exact=False)).to_be_visible(timeout=20_000)
    page.get_by_text("Sources consulted", exact=True).click()
    expect(page.get_by_text(source, exact=False).last).to_be_visible()


def test_new_chat_rotates_url_and_saved_url_resumes_history(
    page: Page, store_server: Callable[[str], RunningStore]
) -> None:
    """Resume real SQLite history without carrying it into a newly created conversation."""
    server = store_server("normal")
    page.goto(server.url)
    _ask(page, "My favourite instrument is Bluejay Acoustic Guitar.")
    expect(page.get_by_text("I will remember Bluejay Acoustic Guitar.", exact=True)).to_be_visible(
        timeout=20_000
    )
    saved = page.url
    assert "conversation=" in saved
    page.get_by_role("button", name="New chat", exact=True).click()
    expect(page).not_to_have_url(saved)
    expect(page.get_by_text("Hello! How can I help you find your next instrument?")).to_be_visible()
    assert page.url != saved
    expect(page.get_by_text("I will remember Bluejay Acoustic Guitar.", exact=True)).to_have_count(
        0
    )
    page.goto(saved)
    expect(page.get_by_text("I will remember Bluejay Acoustic Guitar.", exact=True)).to_be_visible(
        timeout=20_000
    )
    _ask(page, "Do you remember my favourite instrument?")
    expect(page.get_by_text("Your favourite instrument is Bluejay Acoustic Guitar.")).to_be_visible(
        timeout=20_000
    )


def test_browser_contexts_do_not_share_history_or_order_access(
    page: Page, browser: Browser, store_server: Callable[[str], RunningStore]
) -> None:
    """Scope an actual order query and prevent a second conversation from seeing it."""
    server = store_server("normal")
    page.goto(server.url)
    _ask(page, "Show order 7 for alex@example.test.")
    expect(page.get_by_text("Order 7 is shipped. Tracking: PRIVATE-TRACK-A.")).to_be_visible(
        timeout=20_000
    )
    second_context = browser.new_context()
    try:
        second = second_context.new_page()
        second.goto(server.url)
        _ask(second, "Show my order status.")
        expect(second.get_by_text("Please provide your order number and email.")).to_be_visible(
            timeout=20_000
        )
        expect(second.get_by_text("PRIVATE-TRACK-A", exact=False)).to_have_count(0)
        assert second.url != page.url
        saved = page.url
        page.get_by_role("button", name="New chat", exact=True).click()
        expect(page).not_to_have_url(saved)
        _ask(page, "Show my order status.")
        expect(page.get_by_text("Please provide your order number and email.")).to_be_visible(
            timeout=20_000
        )
        expect(page.get_by_text("PRIVATE-TRACK-A", exact=False)).to_have_count(0)
        page.goto(saved)
        _ask(page, "Show my order status.")
        expect(
            page.get_by_text("Order 7 is shipped. Tracking: PRIVATE-TRACK-A.").last
        ).to_be_visible(timeout=20_000)
    finally:
        second_context.close()


@pytest.mark.parametrize(
    ("scenario", "message"),
    [
        ("missing-key", "Chat is not configured yet."),
        ("missing-hf", "HF_TOKEN is required to index the supplied PDFs."),
        ("missing-sources", "The data directory contains no supported CSV or PDF files."),
    ],
)
def test_missing_configuration_or_sources_disables_chat(
    page: Page, store_server: Callable[[str], RunningStore], scenario: str, message: str
) -> None:
    """Show actionable configuration errors without trying a real service or hiding the chat."""
    server = store_server(scenario)
    page.goto(server.url)
    expect(page.get_by_text(message, exact=False)).to_be_visible(timeout=20_000)
    expect(page.get_by_placeholder("Ask the store a question")).to_be_disabled()


def test_request_failure_recovers_without_restarting_application(
    page: Page, store_server: Callable[[str], RunningStore]
) -> None:
    """An external-provider failure produces an error while the next valid turn succeeds."""
    server = store_server("normal")
    page.goto(server.url)
    _ask(page, "Force a request failure.")
    expect(page.get_by_text("The request could not be completed. Please try again.")).to_be_visible(
        timeout=20_000
    )
    _ask(page, "What is the price of Bluejay Acoustic Guitar?")
    expect(page.get_by_text("Bluejay Acoustic Guitar costs BRL 123.45.", exact=True)).to_be_visible(
        timeout=20_000
    )
