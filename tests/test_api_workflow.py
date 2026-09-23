from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app
from app.models import Citation, Chunk, QueryResponse, RetrievalResult, UnsupportedDocumentError
from app.runtime import IngestResult


class FakeRuntime:
    def __init__(self) -> None:
        self.ingested: tuple[str, bytes] | None = None

    def ingest(self, file_name: str, data: bytes) -> IngestResult:
        self.ingested = (file_name, data)
        if file_name.endswith(".docx"):
            raise UnsupportedDocumentError("Unsupported document type '.docx'.")
        return IngestResult("doc-1", file_name, 1, 2, "generation-1", False)

    def query(self, query: str, top_k: int = 5) -> QueryResponse:
        chunk = Chunk("chunk-1", "doc-1", "evidence", "policy.txt", None, 0)
        retrieval = RetrievalResult(chunk, 0.9)
        citation = Citation("chunk-1", "policy.txt", None, "evidence")
        return QueryResponse("The answer is supported. [1]", False, (citation,), (retrieval,))


def _client(runtime: FakeRuntime | None = None, *, api_key: str | None = "test-key") -> TestClient:
    return TestClient(
        create_app(
            Settings(api_key=api_key),
            runtime_factory=(lambda _: runtime) if runtime is not None else None,
        )
    )


def test_query_route_returns_answer_citations_and_debug_retrievals() -> None:
    with _client(FakeRuntime()) as client:
        response = client.post("/query", json={"query": "question", "top_k": 3})

    assert response.status_code == 200
    result = response.json()
    assert result["answer"] == "The answer is supported. [1]"
    assert result["rejected"] is False
    assert result["citations"][0]["file_name"] == "policy.txt"
    assert result["retrievals"][0]["score"] == 0.9


def test_document_route_uploads_bytes_and_returns_index_metadata() -> None:
    runtime = FakeRuntime()
    with _client(runtime) as client:
        response = client.post(
            "/documents", files={"file": ("policy.txt", b"policy text", "text/plain")}
        )

    assert response.status_code == 200
    assert runtime.ingested == ("policy.txt", b"policy text")
    assert response.json()["chunk_count"] == 2
    assert response.json()["index_generation"] == "generation-1"


def test_data_routes_require_api_credentials() -> None:
    with _client(api_key=None) as client:
        response = client.post("/query", json={"query": "question"})

    assert response.status_code == 503
    assert "TRACERAG_API_KEY" in response.json()["detail"]


def test_upload_rejects_unsupported_files_and_oversized_bodies() -> None:
    with _client(FakeRuntime()) as client:
        unsupported = client.post(
            "/documents", files={"file": ("policy.docx", b"content")}
        )
        oversized = client.post(
            "/documents",
            files={"file": ("large.txt", b"x" * (10 * 1024 * 1024 + 1))},
        )

    assert unsupported.status_code == 422
    assert oversized.status_code == 413


def test_query_route_validates_input() -> None:
    with _client(FakeRuntime()) as client:
        response = client.post("/query", json={"query": " ", "top_k": 0})

    assert response.status_code == 422
