"""Build a source-traceable hybrid index from the current PDF inventory."""

import json
import re
import unicodedata
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from hashlib import sha256
from io import BytesIO
from typing import Protocol, cast

import numpy as np
from pypdf import PdfReader
from rank_bm25 import BM25Okapi  # pyright: ignore[reportMissingTypeStubs]

from emporio_agent.embeddings import Embedder, TextTokenizer


class PDFSource(Protocol):
    """Accept the ingestion layer's immutable source records structurally."""

    @property
    def relative_path(self) -> str:
        """Return the path beneath the data directory."""
        ...

    @property
    def content(self) -> bytes:
        """Return the complete source bytes."""
        ...

    @property
    def digest(self) -> str:
        """Return the SHA-256 content digest."""
        ...


@dataclass(frozen=True)
class Citation:
    """Identify the original document, page span and policy section."""

    source: str
    pages: tuple[int, ...]
    section: str


@dataclass(frozen=True)
class PolicySection:
    """Preserve complete subsection text for policy interpretation."""

    title: str
    text: str
    citations: tuple[Citation, ...]


@dataclass(frozen=True)
class PolicyChunk:
    """Store one bounded embedding passage and all matching provenance."""

    text: str
    citations: tuple[Citation, ...]


@dataclass(frozen=True)
class PolicyHit:
    """Return grounded policy text ranked by reciprocal rank fusion."""

    text: str
    citations: tuple[Citation, ...]
    score: float


class _LexicalIndex(Protocol):
    def get_scores(self, query: list[str]) -> Sequence[float]: ...


def _terms(text: str) -> list[str]:
    normalized = unicodedata.normalize("NFKD", text.casefold())
    normalized = "".join(char for char in normalized if not unicodedata.combining(char))
    return re.findall(r"[^\W_]+", normalized)


def _page_content(page: str, headers: Counter[str], footers: Counter[str]) -> list[str]:
    lines = page.splitlines()
    visible = [index for index, line in enumerate(lines) if line.strip()]
    if not visible:
        return []
    first, last = visible[0], visible[-1]
    content: list[str] = []
    for index, line in enumerate(lines):
        stripped = line.strip()
        if index == first and headers[stripped] > 1:
            continue
        if index == last and (
            footers[stripped] > 1 or re.fullmatch(r"(?:Página|Page)\s+\d+", stripped)
        ):
            continue
        content.append(line)
    return content


def _sections(pages: list[list[str]]) -> list[tuple[str, list[tuple[int, str]]]]:
    sections: list[tuple[str, list[tuple[int, str]]]] = []
    title = "Document overview"
    paragraphs: list[tuple[int, str]] = []
    for number, lines in enumerate(pages, start=1):
        pending: list[str] = []

        def flush() -> None:
            if pending:
                paragraphs.append((number, " ".join(pending)))
                pending.clear()

        for line in lines:
            stripped = line.strip()
            if not stripped:
                flush()
                continue
            heading = re.match(r"^\d+(?:\.\d+)*\.?\s+[^\d\s].*", line)
            if heading is not None:
                flush()
                if paragraphs:
                    sections.append((title, paragraphs))
                title = " ".join(stripped.split())
                paragraphs = []
            else:
                pending.append(" ".join(stripped.split()))
        flush()
    if paragraphs:
        sections.append((title, paragraphs))
    return sections


def _extract(source: PDFSource) -> list[tuple[str, list[tuple[int, str]]]]:
    reader = PdfReader(BytesIO(source.content))
    if reader.is_encrypted:
        raise ValueError(f"Encrypted PDF is unsupported: {source.relative_path}")
    pages = [page.extract_text(extraction_mode="layout") for page in reader.pages]
    visible = [[line.strip() for line in page.splitlines() if line.strip()] for page in pages]
    headers = Counter(lines[0] for lines in visible if lines)
    footers = Counter(lines[-1] for lines in visible if lines)
    sections = _sections([_page_content(page, headers, footers) for page in pages])
    if not sections:
        raise ValueError(
            f"PDF has no extractable text; OCR is not supported: {source.relative_path}"
        )
    return sections


def _chunks(
    title: str, paragraphs: list[tuple[int, str]], source: str, tokenizer: TextTokenizer
) -> list[PolicyChunk]:
    heading = title
    if tokenizer.count(heading) > 120:
        heading = tokenizer.split(heading, 120, 0)[0]
    prefix = heading + "\n\n"
    allowance = 600 - tokenizer.count(prefix) - 8
    parts: list[tuple[int, str]] = []
    for page, paragraph in paragraphs:
        if tokenizer.count(paragraph) <= allowance:
            parts.append((page, paragraph))
        else:
            parts.extend((page, text) for text in tokenizer.split(paragraph, allowance, 0))
    result: list[PolicyChunk] = []
    body: list[str] = []
    page_numbers: set[int] = set()
    last_page = 0

    def emit(keep_overlap: bool = False) -> None:
        if body:
            text = prefix + "\n\n".join(body)
            if tokenizer.count(text) > 800:
                raise ValueError("Chunk exceeds the configured 800-token limit.")
            result.append(
                PolicyChunk(text, (Citation(source, tuple(sorted(page_numbers)), title),))
            )
            tail = tokenizer.split(body[-1], 80, 0)[-1] if keep_overlap else None
            body.clear()
            page_numbers.clear()
            if tail:
                body.append(tail)
                page_numbers.add(last_page)

    for page, text in parts:
        candidate = prefix + "\n\n".join([*body, text])
        if body and tokenizer.count(candidate) > 600:
            emit(keep_overlap=True)
        body.append(text)
        page_numbers.add(page)
        last_page = page
    emit()
    return result


class PolicyIndex:
    """Keep BM25 and normalized vectors over the same deduplicated passages."""

    def __init__(
        self,
        chunks: tuple[PolicyChunk, ...],
        sections: tuple[PolicySection, ...],
        embedder: Embedder,
        fingerprint: str,
    ) -> None:
        if not chunks:
            raise ValueError("At least one readable policy PDF is required.")
        self.chunks = chunks
        self.sections = sections
        self.fingerprint = fingerprint
        self._embedder = embedder
        constructor = cast(Callable[[list[list[str]]], _LexicalIndex], BM25Okapi)
        self._lexical = constructor([_terms(chunk.text) for chunk in chunks])
        self._vectors = np.asarray(
            embedder.embed_documents([chunk.text for chunk in chunks]), dtype=np.float64
        )
        if self._vectors.ndim != 2 or self._vectors.shape[0] != len(chunks):
            raise ValueError("Document embeddings do not match the policy chunks.")
        norms = np.linalg.norm(self._vectors, axis=1)
        if not np.isfinite(self._vectors).all() or np.any(norms == 0):
            raise ValueError("Document embeddings must be finite, nonzero vectors.")
        self._vectors = self._vectors / norms[:, None]

    def search(self, query: str, limit: int = 5) -> list[PolicyHit]:
        """Fuse independent lexical and cosine rankings without mixing raw scores."""
        if not query.strip() or limit <= 0:
            return []
        query_vector = np.asarray(self._embedder.embed_query(query), dtype=np.float64)
        if query_vector.shape != (self._vectors.shape[1],):
            raise ValueError("Query embedding does not match document dimensions.")
        norm = float(np.linalg.norm(query_vector))
        if not np.isfinite(query_vector).all() or norm == 0:
            raise ValueError("Query embedding must be finite and nonzero.")
        cosine = self._vectors @ (query_vector / norm)
        lexical = [float(value) for value in self._lexical.get_scores(_terms(query))]
        vector_ranks = sorted(range(len(self.chunks)), key=lambda i: (-float(cosine[i]), i))
        lexical_ranks = sorted(
            (i for i, value in enumerate(lexical) if value > 0),
            key=lambda i: (-lexical[i], i),
        )
        scores: dict[int, float] = {}
        for ranking in (vector_ranks, lexical_ranks):
            for rank, index in enumerate(ranking[:10], start=1):
                scores[index] = scores.get(index, 0.0) + 1.0 / (60 + rank)
        ordered = sorted(scores, key=lambda i: (-scores[i], i))[:limit]
        return [
            PolicyHit(self.chunks[i].text, self.chunks[i].citations, scores[i]) for i in ordered
        ]


def build_index(
    sources: Sequence[PDFSource], embedder: Embedder, tokenizer: TextTokenizer
) -> PolicyIndex:
    """Rebuild from the complete active PDF inventory, including deletions and renames."""
    pdfs = sorted(
        (source for source in sources if source.relative_path.casefold().endswith(".pdf")),
        key=lambda source: source.relative_path,
    )
    inventory: list[tuple[str, str]] = []
    sections: list[PolicySection] = []
    unique: dict[str, PolicyChunk] = {}
    for source in pdfs:
        digest = sha256(source.content).hexdigest()
        if digest != source.digest:
            raise ValueError(f"Source digest does not match its bytes: {source.relative_path}")
        inventory.append((source.relative_path, digest))
        for title, paragraphs in _extract(source):
            pages = tuple(sorted({page for page, _ in paragraphs}))
            sections.append(
                PolicySection(
                    title,
                    "\n\n".join(text for _, text in paragraphs),
                    (Citation(source.relative_path, pages, title),),
                )
            )
            for chunk in _chunks(title, paragraphs, source.relative_path, tokenizer):
                key = sha256(" ".join(chunk.text.split()).encode()).hexdigest()
                previous = unique.get(key)
                if previous is None:
                    unique[key] = chunk
                else:
                    unique[key] = PolicyChunk(
                        previous.text, tuple(dict.fromkeys((*previous.citations, *chunk.citations)))
                    )
    fingerprint = sha256(
        json.dumps(
            (
                inventory,
                embedder.fingerprint,
                tokenizer.fingerprint,
                "layout-sections-v1-600-800-80",
            )
        ).encode()
    ).hexdigest()
    return PolicyIndex(tuple(unique.values()), tuple(sections), embedder, fingerprint)
