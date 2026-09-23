"""Small domain models shared by document ingestion and retrieval."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True, slots=True)
class DocumentPage:
    """A text unit from a source document, retaining its origin."""

    document_id: str
    file_name: str
    page_number: int | None
    content: str


@dataclass(frozen=True, slots=True)
class Chunk:
    """A chunk that can be traced back to a source document and page."""

    chunk_id: str
    document_id: str
    content: str
    file_name: str
    page_number: int | None
    chunk_index: int

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, object]) -> "Chunk":
        return cls(
            chunk_id=str(value["chunk_id"]),
            document_id=str(value["document_id"]),
            content=str(value["content"]),
            file_name=str(value["file_name"]),
            page_number=(int(value["page_number"]) if value["page_number"] is not None else None),
            chunk_index=int(value["chunk_index"]),
        )


@dataclass(frozen=True, slots=True)
class RetrievalResult:
    """A chunk and its cosine similarity score."""

    chunk: Chunk
    score: float


@dataclass(frozen=True, slots=True)
class Citation:
    """A source reference attached to a generated answer."""

    chunk_id: str
    file_name: str
    page_number: int | None
    text: str


@dataclass(frozen=True, slots=True)
class QueryResponse:
    """The RAG answer and source references returned to API/UI clients."""

    answer: str
    rejected: bool
    citations: tuple[Citation, ...]
    results: tuple[RetrievalResult, ...] = ()


class UnsupportedDocumentError(ValueError):
    """Raised when a file format is outside the V0.1 document scope."""
