"""Check source inventory changes, provenance and bounded hybrid passages."""

from collections.abc import Sequence
from dataclasses import dataclass
from hashlib import sha256
from io import BytesIO
from pathlib import Path

from pypdf import PdfWriter
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

from emporio_agent.documents import build_index


@dataclass(frozen=True)
class Source:
    """Provide immutable input without depending on CSV ingestion internals."""

    relative_path: str
    content: bytes

    @property
    def digest(self) -> str:
        """Compute the source content identity."""
        return sha256(self.content).hexdigest()


class CharacterTokenizer:
    """Use characters as deterministic tokens for chunk-boundary tests."""

    fingerprint = "characters-v1"

    def count(self, text: str) -> int:
        """Count every character, including whitespace."""
        return len(text)

    def split(self, text: str, max_tokens: int, overlap: int) -> list[str]:
        """Preserve complete input with a bounded overlap."""
        return [
            text[start : start + max_tokens] for start in range(0, len(text), max_tokens - overlap)
        ]


class FakeEmbedder:
    """Rank a few policy topics without network calls."""

    fingerprint = "test-embedding-v1"

    def __init__(self) -> None:
        self.documents: list[str] = []

    @staticmethod
    def _vector(text: str) -> tuple[float, ...]:
        lowered = text.casefold()
        return (
            float("pix" in lowered),
            float("return" in lowered),
            float("credit" in lowered),
            0.1,
        )

    def embed_documents(self, texts: Sequence[str]) -> list[tuple[float, ...]]:
        """Record exactly which passages are embedded."""
        self.documents = list(texts)
        return [self._vector(text) for text in self.documents]

    def embed_query(self, query: str) -> tuple[float, ...]:
        """Produce a topic vector for a query."""
        return self._vector(query)


def pdf_bytes(lines: list[str]) -> bytes:
    """Create a tiny text PDF entirely in memory."""
    writer = PdfWriter()
    page = writer.add_blank_page(width=612, height=792)
    font = DictionaryObject(
        {
            NameObject("/Type"): NameObject("/Font"),
            NameObject("/Subtype"): NameObject("/Type1"),
            NameObject("/BaseFont"): NameObject("/Helvetica"),
        }
    )
    page[NameObject("/Resources")] = DictionaryObject(
        {NameObject("/Font"): DictionaryObject({NameObject("/F1"): font})}
    )
    commands = ["BT /F1 10 Tf 12 TL 50 750 Td"]
    for line in lines:
        escaped = line.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
        commands.append(f"({escaped}) Tj T*")
    commands.append("ET")
    stream = DecodedStreamObject()
    stream.set_data("\n".join(commands).encode("latin-1"))
    page[NameObject("/Contents")] = stream
    target = BytesIO()
    writer.write(target)
    return target.getvalue()


def test_inventory_changes_rebuild_index_and_remove_citations() -> None:
    """Changing membership must replace the active policy inventory."""
    first = Source("first.pdf", pdf_bytes(["1. Payments", "PIX has a discount."]))
    second = Source("second.pdf", pdf_bytes(["1. Returns", "Returns need the original box."]))
    tokenizer = CharacterTokenizer()
    original = build_index([first], FakeEmbedder(), tokenizer)
    added = build_index([first, second], FakeEmbedder(), tokenizer)
    removed = build_index([second], FakeEmbedder(), tokenizer)
    assert len({original.fingerprint, added.fingerprint, removed.fingerprint}) == 3
    assert all(
        citation.source == "second.pdf"
        for hit in removed.search("return")
        for citation in hit.citations
    )
    assert "Returns" in added.search("Returns")[0].text


def test_duplicate_pdf_keeps_provenance_without_duplicate_vectors() -> None:
    """Copies must not crowd the rankings or trigger duplicate document embeddings."""
    content = pdf_bytes(["1. Payments", "PIX has a discount."])
    embedder = FakeEmbedder()
    index = build_index(
        [
            Source("one.pdf", content),
            Source("copy.pdf", content),
            Source("products.csv", b"never embed"),
        ],
        embedder,
        CharacterTokenizer(),
    )
    assert len(index.chunks) == 1
    assert len(embedder.documents) == 1
    assert {citation.source for citation in index.chunks[0].citations} == {"one.pdf", "copy.pdf"}


def test_oversized_paragraph_preserves_tail_in_bounded_chunks() -> None:
    """Long paragraphs must be split rather than silently truncated."""
    body = "policy " * 700 + "final exception"
    source = Source("long.pdf", pdf_bytes(["1. Payments", body]))
    index = build_index([source], FakeEmbedder(), CharacterTokenizer())
    assert len(index.chunks) > 1
    assert all(len(chunk.text) <= 800 for chunk in index.chunks)
    assert any("final exception" in chunk.text for chunk in index.chunks)


def test_actual_manual_layout_and_page_citations() -> None:
    """Extract the supplied manual with readable headings and page provenance."""
    path = next((Path(__file__).parents[1] / "data").glob("*.pdf"))
    index = build_index(
        [Source(path.name, path.read_bytes())], FakeEmbedder(), CharacterTokenizer()
    )
    assert any("Formas de Pagamento" in section.title for section in index.sections)
    assert any("PIX" in chunk.text for chunk in index.chunks)
    assert all(citation.pages for chunk in index.chunks for citation in chunk.citations)


def test_repeated_headers_are_detected_by_position_not_shop_wording() -> None:
    """Remove repeated page furniture while preserving repeated body statements."""
    writer = PdfWriter()
    for number, heading in ((1, "1. Payments"), (2, "2. Returns")):
        writer.append(
            BytesIO(
                pdf_bytes(
                    [
                        "Another company's customer handbook",
                        "",
                        heading,
                        "Useful repeated body statement.",
                        "Unique section details.",
                        f"Page {number}",
                    ]
                )
            )
        )
    output = BytesIO()
    writer.write(output)
    index = build_index(
        [Source("different-handbook.pdf", output.getvalue())],
        FakeEmbedder(),
        CharacterTokenizer(),
    )
    assert not any("Another company's" in chunk.text for chunk in index.chunks)
    assert sum("Useful repeated body statement." in item.text for item in index.sections) == 2
