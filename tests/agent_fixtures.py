"""Native HTTP fixtures shared by agent integration and browser tests."""

import ast
import csv
import hashlib
import io
import json
from collections import deque
from collections.abc import Callable, Sequence
from typing import cast

import httpx2

from emporio_agent.data import SourceFile, build_catalog
from emporio_agent.documents import Citation, PolicyChunk, PolicyIndex, PolicySection
from emporio_agent.policies import extract_rules
from emporio_agent.runtime import Snapshot

type JsonObject = dict[str, object]


def _objects(value: object) -> list[JsonObject]:
    if not isinstance(value, list):
        return []
    return [cast(JsonObject, item) for item in cast(list[object], value) if isinstance(item, dict)]


def _message(content: list[JsonObject], stop_reason: str) -> JsonObject:
    return {
        "id": "msg_fixture",
        "type": "message",
        "role": "assistant",
        "model": "claude-sonnet-5-5",
        "content": content,
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": {
            "input_tokens": 120,
            "output_tokens": 20,
            "cache_creation_input_tokens": 40,
            "cache_read_input_tokens": 80,
        },
    }


def text_response(text: str, *, thinking: bool = False) -> JsonObject:
    """Return a native Anthropic final message, optionally with signed-block-shaped data."""
    content: list[JsonObject] = []
    if thinking:
        content.append(
            {
                "type": "thinking",
                "thinking": "Fixture provider note.",
                "signature": "fixture-signature",
            }
        )
    content.append({"type": "text", "text": text})
    return _message(content, "end_turn")


def tool_response(
    name: str,
    arguments: JsonObject,
    *,
    call_id: str = "toolu_fixture",
    thinking: bool = True,
) -> JsonObject:
    """Request a real application tool through the native provider response format."""
    content: list[JsonObject] = []
    if thinking:
        content.append(
            {
                "type": "thinking",
                "thinking": "A lookup is needed.",
                "signature": "fixture-signature",
            }
        )
    content.append({"type": "tool_use", "id": call_id, "name": name, "input": arguments})
    return _message(content, "tool_use")


def latest_user_text(body: JsonObject) -> str:
    """Read the latest actual text block, skipping tool-result-only user messages."""
    for message in reversed(_objects(body.get("messages"))):
        if message.get("role") != "user":
            continue
        content = message.get("content")
        if isinstance(content, str):
            return content
        for block in reversed(_objects(content)):
            if block.get("type") == "text" and isinstance(block.get("text"), str):
                return cast(str, block["text"])
    return ""


def tool_results(body: JsonObject) -> list[JsonObject]:
    """Decode Agno's JSON or Python-literal tool payload from a native request."""
    results: list[JsonObject] = []
    for message in _objects(body.get("messages")):
        for block in _objects(message.get("content")):
            if block.get("type") != "tool_result":
                continue
            content = block.get("content")
            if not isinstance(content, str):
                content = "\n".join(
                    cast(str, item["text"])
                    for item in _objects(content)
                    if isinstance(item.get("text"), str)
                )
            try:
                parsed: object = json.loads(content)
            except json.JSONDecodeError:
                parsed = ast.literal_eval(content)
            if isinstance(parsed, dict):
                results.append(cast(JsonObject, parsed))
    return results


class ClaudeStub:
    """Intercept only native Anthropic HTTP; leave Agno, tools and persistence real."""

    def __init__(
        self, responder: Callable[[JsonObject], JsonObject] | Sequence[JsonObject]
    ) -> None:
        """Accept a per-request callback or a bounded scripted response queue."""
        self.requests: list[JsonObject] = []
        self._responder = responder if callable(responder) else None
        self._responses = deque[JsonObject]() if callable(responder) else deque(responder)
        self.transport = httpx2.MockTransport(self._handle)

    def _handle(self, request: httpx2.Request) -> httpx2.Response:
        if request.url.path.endswith("/messages/count_tokens"):
            return httpx2.Response(200, json={"input_tokens": 120})
        if request.method != "POST" or not request.url.path.endswith("/messages"):
            raise AssertionError(
                f"Unexpected native provider route: {request.method} {request.url.path}"
            )
        body = cast(JsonObject, json.loads(request.content))
        self.requests.append(body)
        if self._responder is not None:
            response = self._responder(body)
        elif self._responses:
            response = self._responses.popleft()
        else:
            raise AssertionError("The native provider fixture has no scripted response left.")
        return httpx2.Response(200, json=response, headers={"request-id": "fixture-request"})


class FixedEmbedder:
    """Provide finite deterministic vectors without any embedding provider calls."""

    fingerprint = "fixture-embedding-v1"

    def embed_documents(self, texts: Sequence[str]) -> list[tuple[float, ...]]:
        """Return one stable vector per supplied passage."""
        return [(1.0, 0.0) for _ in texts]

    def embed_query(self, query: str) -> tuple[float, ...]:
        """Keep policy retrieval deterministic for UI and integration checks."""
        return (1.0, 0.0)


class CharacterTokenizer:
    """Bound fixture chunks without downloading tokenizer files."""

    fingerprint = "fixture-characters-v1"

    def count(self, text: str) -> int:
        """Use a conservative deterministic character count."""
        return len(text)

    def split(self, text: str, max_tokens: int, overlap: int) -> list[str]:
        """Split fixture text without dropping its tail."""
        return [
            text[start : start + max_tokens] for start in range(0, len(text), max_tokens - overlap)
        ]


def _csv(name: str, rows: list[list[str | int]]) -> SourceFile:
    output = io.StringIO(newline="")
    csv.writer(output).writerows(rows)
    content = output.getvalue().encode("utf-8")
    return SourceFile(name, content, hashlib.sha256(content).hexdigest())


def small_snapshot(
    *,
    price_cents: int = 10001,
    pix_percent: str = "7",
    promotion_percent: str | None = None,
) -> Snapshot:
    """Build independent business rows and source-derived policy values for tests."""
    sources = (
        _csv("groups.csv", [["category_id", "name", "description"], [1, "Guitars", "Instruments"]]),
        _csv(
            "people.csv",
            [
                ["customer_id", "name", "phone", "email", "city"],
                [11, "Alex Fixture", "6700000001", "alex@example.test", "Campo Grande"],
                [22, "Bea Fixture", "6700000002", "bea@example.test", "Campo Grande"],
            ],
        ),
        _csv(
            "instruments.csv",
            [
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
                    101,
                    f"{price_cents // 100}.{price_cents % 100:02}",
                    "Demo Guitar",
                    1,
                    "Acoustic instrument",
                    5,
                    "active",
                    "{}",
                    "2026-01-01",
                ],
                [
                    202,
                    "249.99",
                    "Demo Keyboard",
                    1,
                    "Electronic instrument",
                    3,
                    "active",
                    "{}",
                    "2026-01-01",
                ],
            ],
        ),
        _csv(
            "campaigns.csv",
            [
                ["promotion_id", "product_id", "discount_percent", "description", "is_active"],
                *(
                    [[301, 101, promotion_percent, "Current offer", "true"]]
                    if promotion_percent is not None
                    else []
                ),
            ],
        ),
        _csv(
            "purchases.csv",
            [
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
                    501,
                    11,
                    "2026-01-02",
                    "shipped",
                    "100.01",
                    "credit",
                    "BR000000001BR",
                    "2026-01-06",
                    "Private Alex note",
                ],
                [502, 22, "2026-01-03", "pending", "249.99", "pix", "", "", "Private Bea note"],
            ],
        ),
        _csv(
            "purchase-lines.csv",
            [
                ["order_id", "quantity", "product_id"],
                [501, 1, 101],
                [502, 1, 202],
            ],
        ),
    )
    catalog = build_catalog(sources)
    reference = (Citation("payment-policy.pdf", (1,), "Payment conditions"),)
    text = (
        f"PIX Pagamento à vista {pix_percent}% de desconto sobre o preço de tabela. "
        "Parcelamento de 4x a 12x: sem juros, parcela mínima de R$ 1,00. "
        "Promoções não são cumulativas."
    )
    section = PolicySection("Payment conditions", text, reference)
    index = PolicyIndex(
        (PolicyChunk(text, reference),), (section,), FixedEmbedder(), "fixture-policy"
    )
    rules = extract_rules((section,))
    generation = hashlib.sha256((catalog.fingerprint + pix_percent).encode()).hexdigest()
    return Snapshot(generation, catalog, index, rules)
