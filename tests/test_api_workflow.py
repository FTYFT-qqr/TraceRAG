"""验证 FastAPI 上传、查询、错误映射和界面所需状态字段。"""

from fastapi.testclient import TestClient
import httpx
from openai import AuthenticationError

from app.config import Settings
from app.main import create_app
from app.models import (
    Citation,
    Chunk,
    EmbeddingIndexCompatibilityError,
    QueryResponse,
    RetrievalResult,
    UnsupportedDocumentError,
)
from app.runtime import IngestResult


class FakeRuntime:
    def __init__(self) -> None:
        """初始化测试替身及其记录状态。"""
        self.ingested: tuple[str, bytes] | None = None

    def ingest(self, file_name: str, data: bytes) -> IngestResult:
        """模拟文档入库并返回固定入库结果。"""
        self.ingested = (file_name, data)
        if file_name.endswith(".docx"):
            raise UnsupportedDocumentError("Unsupported document type '.docx'.")
        return IngestResult("doc-1", file_name, 1, 2, "generation-1", False)

    def query(self, query: str, top_k: int = 5) -> QueryResponse:
        """模拟查询结果，供接口测试隔离模型依赖。"""
        chunk = Chunk("chunk-1", "doc-1", "evidence", "policy.txt", None, 0)
        retrieval = RetrievalResult(chunk, 0.9)
        citation = Citation("chunk-1", "policy.txt", None, "evidence")
        return QueryResponse("The answer is supported. [1]", False, (citation,), (retrieval,))


def _client(runtime: FakeRuntime | None = None, *, api_key: str | None = "test-key") -> TestClient:
    """创建测试用 FastAPI 客户端及隔离运行时。"""
    return TestClient(
        create_app(
            Settings(api_key=api_key),
            runtime_factory=(lambda _: runtime) if runtime is not None else None,
        )
    )


def test_query_route_returns_answer_citations_and_debug_retrievals() -> None:
    """验证查询接口返回回答、引用和调试检索结果。"""
    with _client(FakeRuntime()) as client:
        response = client.post("/query", json={"query": "question", "top_k": 3})

    assert response.status_code == 200
    result = response.json()
    assert result["answer"] == "The answer is supported. [1]"
    assert result["rejected"] is False
    assert result["citations"][0]["file_name"] == "policy.txt"
    assert result["retrievals"][0]["score"] == 0.9
    assert result["retrievals"][0]["candidate_rank"] is None
    assert result["retrievals"][0]["score_kind"] == "cosine"
    assert result["retrievals"][0]["bm25_score"] is None


def test_query_api_preserves_hybrid_channel_scores() -> None:
    """接口保留融合类型与两路原始分，供前端正确解释排序和门槛。"""

    class HybridRuntime(FakeRuntime):
        def query(self, query: str, top_k: int = 5) -> QueryResponse:
            """返回带两路原始分的固定融合结果。"""
            original = super().query(query, top_k)
            result = RetrievalResult(original.results[0].chunk, 0.03, 1, "rrf", 0.5, 2.1)
            return QueryResponse(original.answer, False, original.citations, (result,))

    with _client(HybridRuntime()) as client:
        response = client.post("/query", json={"query": "设备借用"})
    result = response.json()["retrievals"][0]
    assert result["score_kind"] == "rrf"
    assert result["vector_score"] == 0.5
    assert result["bm25_score"] == 2.1


def test_document_route_uploads_bytes_and_returns_index_metadata() -> None:
    """验证上传接口能够处理文件字节并返回索引元数据。"""
    runtime = FakeRuntime()
    with _client(runtime) as client:
        response = client.post(
            "/documents", files={"file": ("policy.txt", b"policy text", "text/plain")}
        )

    assert response.status_code == 200
    assert runtime.ingested == ("policy.txt", b"policy text")
    assert response.json()["chunk_count"] == 2
    assert response.json()["index_generation"] == "generation-1"
    assert response.json()["already_indexed"] is False
    assert response.json()["replaced_existing"] is False


def test_document_route_reports_same_name_replacement() -> None:
    """确认 API 把同名替换状态传给前端，而不只返回新增片段数。"""
    class ReplacingRuntime(FakeRuntime):
        def ingest(self, file_name: str, data: bytes) -> IngestResult:
            """模拟文档入库并返回固定入库结果。"""
            return IngestResult(
                "doc-2", file_name, 1, 3, "generation-2", False, replaced_existing=True
            )

    with _client(ReplacingRuntime()) as client:
        response = client.post(
            "/documents", files={"file": ("policy.txt", b"new policy", "text/plain")}
        )

    assert response.status_code == 200
    assert response.json()["replaced_existing"] is True


def test_data_routes_require_api_credentials() -> None:
    """验证上传和查询接口在缺少凭据时返回明确错误。"""
    with _client(api_key=None) as client:
        response = client.post("/query", json={"query": "question"})

    assert response.status_code == 503
    assert "TRACERAG_API_KEY" in response.json()["detail"]


def test_query_route_explains_how_to_resolve_index_identity_mismatch() -> None:
    """验证索引模型不匹配时接口提供可执行的修复说明。"""
    def incompatible_runtime(_: Settings) -> FakeRuntime:
        """构造索引身份不匹配场景的运行时替身。"""
        raise EmbeddingIndexCompatibilityError(
            "Configured embedding model does not match the saved vector index."
        )

    app = create_app(
        Settings(api_key="test-key"),
        runtime_factory=incompatible_runtime,
    )
    with TestClient(app) as client:
        response = client.post("/query", json={"query": "question"})

    assert response.status_code == 409
    assert "does not match" in response.json()["detail"]


def test_upload_rejects_unsupported_files_and_oversized_bodies() -> None:
    """验证上传接口拒绝不支持的文件类型和超大请求。"""
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
    """验证查询接口校验空问题和 Top-K 参数。"""
    with _client(FakeRuntime()) as client:
        response = client.post("/query", json={"query": " ", "top_k": 0})
        api_limit = client.post("/query", json={"query": "question", "top_k": 20})
        over_limit = client.post("/query", json={"query": "question", "top_k": 21})

    assert response.status_code == 422
    assert api_limit.status_code == 200
    assert over_limit.status_code == 422


def test_query_route_reports_rejected_provider_credentials_without_leaking_response() -> None:
    """验证模型凭据错误会安全报告且不泄露供应商响应正文。"""
    class RejectingRuntime(FakeRuntime):
        def query(self, query: str, top_k: int = 5) -> QueryResponse:
            """模拟查询结果，供接口测试隔离模型依赖。"""
            response = httpx.Response(
                401,
                request=httpx.Request("POST", "https://provider.example/chat/completions"),
                json={"error": {"message": "secret response detail"}},
            )
            raise AuthenticationError("secret response detail", response=response, body=None)

    with _client(RejectingRuntime()) as client:
        result = client.post("/query", json={"query": "question"})

    assert result.status_code == 502
    assert "TRACERAG_API_KEY" in result.json()["detail"]
    assert "secret response detail" not in result.text
