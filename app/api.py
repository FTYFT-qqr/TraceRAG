"""提供文档上传入库与基于知识库的查询 API。"""

from __future__ import annotations

import logging
import threading
from typing import Callable

from fastapi import APIRouter, File, HTTPException, UploadFile
from openai import APIConnectionError, AuthenticationError
from pydantic import BaseModel, Field

from app.config import Settings
from app.models import (
    EmbeddingIndexCompatibilityError,
    QueryResponse,
    UnsupportedDocumentError,
)
from app.runtime import RAGRuntime


logger = logging.getLogger(__name__)
MAX_UPLOAD_BYTES = 10 * 1024 * 1024
RuntimeFactory = Callable[[Settings], RAGRuntime]


class DocumentUploadResponse(BaseModel):
    """上传结果及用于区分首次、重复和更新的状态字段。"""

    document_id: str
    file_name: str
    page_count: int
    chunk_count: int
    index_generation: str
    already_indexed: bool
    replaced_existing: bool = False


class DocumentSummaryResponse(BaseModel):
    """展示当前文档的真实片段数量及已索引 PDF 页数。"""

    document_id: str
    file_name: str
    chunk_count: int
    indexed_page_count: int | None


class DocumentListResponse(BaseModel):
    """返回当前知识库目录与快照标识。"""

    documents: list[DocumentSummaryResponse]
    document_count: int
    chunk_count: int
    index_generation: str


class DocumentDeleteResponse(BaseModel):
    """返回删除后的快照及剩余文档和片段数。"""

    document_id: str
    file_name: str
    deleted_chunk_count: int
    index_generation: str
    document_count: int
    chunk_count: int


class QueryRequest(BaseModel):
    """约束问题长度及最终交给模型参考的片段数量。"""

    query: str = Field(min_length=1, max_length=8000)
    top_k: int = Field(
        default=5,
        ge=1,
        le=20,
        description="最终供模型参考且在检索结果中返回的片段上限。",
    )


class CitationResponse(BaseModel):
    """返回给调用方的来源引用。"""

    chunk_id: str
    file_name: str
    page_number: int | None
    text: str


class RetrievalResponse(BaseModel):
    """用于检查最终参考片段、原始候选名次和相似度分数。"""

    chunk_id: str
    file_name: str
    page_number: int | None
    text: str
    score: float
    candidate_rank: int | None
    score_kind: str = "cosine"
    vector_score: float | None = None
    bm25_score: float | None = None


class QueryApiResponse(BaseModel):
    """统一封装回答、拒答标记、引用和候选检索结果。"""

    answer: str
    rejected: bool
    citations: list[CitationResponse]
    retrievals: list[RetrievalResponse]


def _query_response(response: QueryResponse) -> QueryApiResponse:
    """将领域模型转换为 FastAPI 对外响应结构。"""
    return QueryApiResponse(
        answer=response.answer,
        rejected=response.rejected,
        citations=[
            CitationResponse(
                chunk_id=item.chunk_id,
                file_name=item.file_name,
                page_number=item.page_number,
                text=item.text,
            )
            for item in response.citations
        ],
        retrievals=[
            RetrievalResponse(
                chunk_id=item.chunk.chunk_id,
                file_name=item.chunk.file_name,
                page_number=item.chunk.page_number,
                text=item.chunk.content,
                score=item.score,
                score_kind=item.score_kind,
                vector_score=item.vector_score,
                bm25_score=item.bm25_score,
                candidate_rank=item.candidate_rank,
            )
            for item in response.results
        ],
    )


def create_api_router(
    settings: Settings,
    *,
    runtime_factory: RuntimeFactory | None = None,
) -> APIRouter:
    """创建文档管理和问答路由，并按首次请求延迟初始化运行时。"""

    router = APIRouter()
    factory = runtime_factory or RAGRuntime.from_settings
    runtime: RAGRuntime | None = None
    runtime_lock = threading.RLock()

    def get_runtime() -> RAGRuntime:
        """在锁内只初始化一次模型和索引，避免并发重复加载。"""
        nonlocal runtime
        if not settings.api_key:
            raise HTTPException(
                status_code=503,
                detail="Configure TRACERAG_API_KEY or OPENAI_API_KEY to use ingestion and queries.",
            )
        with runtime_lock:
            if runtime is None:
                try:
                    runtime = factory(settings)
                except EmbeddingIndexCompatibilityError as exc:
                    # 索引身份冲突需要用户更换模型或新目录，返回明确的恢复方式。
                    logger.warning("Configured embedding does not match the saved index: %s", exc)
                    raise HTTPException(status_code=409, detail=str(exc)) from exc
                except Exception as exc:
                    logger.exception("Could not initialize the TraceRAG runtime")
                    raise HTTPException(
                        status_code=503,
                        detail="TraceRAG runtime is unavailable; check provider credentials and index data.",
                    ) from exc
            return runtime

    @router.get("/documents", response_model=DocumentListResponse, tags=["documents"])
    def list_documents() -> DocumentListResponse:
        """在运行时锁内读取当前目录，避免列表与索引切换发生竞争。"""
        active_runtime = get_runtime()
        try:
            with runtime_lock:
                documents = active_runtime.list_documents()
                generation = active_runtime._current_generation()
        except EmbeddingIndexCompatibilityError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except Exception as exc:
            logger.exception("Could not read the current document catalog")
            raise HTTPException(status_code=503, detail="Could not read the document catalog.") from exc
        return DocumentListResponse(
            documents=[DocumentSummaryResponse(
                document_id=document.document_id,
                file_name=document.file_name,
                chunk_count=document.chunk_count,
                indexed_page_count=document.indexed_page_count,
            ) for document in documents],
            document_count=len(documents),
            chunk_count=sum(document.chunk_count for document in documents),
            index_generation=generation,
        )

    @router.delete("/documents/{document_id}", response_model=DocumentDeleteResponse, tags=["documents"])
    def delete_document(document_id: str) -> DocumentDeleteResponse:
        """按文档标识提交删除，不接收或删除调用方指定的文件路径。"""
        active_runtime = get_runtime()
        try:
            with runtime_lock:
                result = active_runtime.delete_document(document_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Document is no longer in the current knowledge base.") from exc
        except EmbeddingIndexCompatibilityError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except Exception as exc:
            logger.exception("Document deletion failed; the previous snapshot is retained")
            raise HTTPException(status_code=503, detail="Could not save the document deletion; try again.") from exc
        return DocumentDeleteResponse(
            document_id=result.document_id,
            file_name=result.file_name,
            deleted_chunk_count=result.deleted_chunk_count,
            index_generation=result.generation,
            document_count=result.document_count,
            chunk_count=result.chunk_count,
        )

    @router.post("/documents", response_model=DocumentUploadResponse, tags=["documents"])
    def upload_document(file: UploadFile = File(...)) -> DocumentUploadResponse:
        """读取上限加 1 字节，以便不将超限文件完整载入内存。"""
        if not file.filename:
            raise HTTPException(status_code=422, detail="A file name is required.")
        data = file.file.read(MAX_UPLOAD_BYTES + 1)
        if len(data) > MAX_UPLOAD_BYTES:
            raise HTTPException(
                status_code=413,
                detail=f"File is too large; the limit is {MAX_UPLOAD_BYTES // (1024 * 1024)} MiB.",
            )
        active_runtime = get_runtime()
        try:
            with runtime_lock:
                result = active_runtime.ingest(file.filename, data)
        except (UnsupportedDocumentError, ValueError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except AuthenticationError as exc:
            logger.warning("Document embedding provider rejected API credentials")
            raise HTTPException(
                status_code=502,
                detail="Provider rejected the API key; check TRACERAG_API_KEY and TRACERAG_BASE_URL.",
            ) from exc
        except APIConnectionError as exc:
            logger.warning("Document embedding provider connection failed")
            raise HTTPException(
                status_code=502,
                detail="Cannot reach the provider; check TRACERAG_BASE_URL and TRACERAG_API_PROXY_URL.",
            ) from exc
        except Exception as exc:
            logger.exception("Document ingestion failed")
            raise HTTPException(
                status_code=502,
                detail="Document ingestion failed while embedding or saving the index.",
            ) from exc
        return DocumentUploadResponse(
            document_id=result.document_id,
            file_name=result.file_name,
            page_count=result.page_count,
            chunk_count=result.chunk_count,
            index_generation=result.generation,
            already_indexed=result.already_indexed,
            replaced_existing=result.replaced_existing,
        )

    @router.post("/query", response_model=QueryApiResponse, tags=["query"])
    def query_documents(request: QueryRequest) -> QueryApiResponse:
        """在运行时锁内查询，避免查询与索引快照切换发生竞争。"""
        active_runtime = get_runtime()
        if not request.query.strip():
            raise HTTPException(status_code=422, detail="Query cannot be empty.")
        try:
            with runtime_lock:
                result = active_runtime.query(request.query, top_k=request.top_k)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except AuthenticationError as exc:
            logger.warning("Query provider rejected API credentials")
            raise HTTPException(
                status_code=502,
                detail="Provider rejected the API key; check TRACERAG_API_KEY and TRACERAG_BASE_URL.",
            ) from exc
        except APIConnectionError as exc:
            logger.warning("Query provider connection failed")
            raise HTTPException(
                status_code=502,
                detail="Cannot reach the provider; check TRACERAG_BASE_URL and TRACERAG_API_PROXY_URL.",
            ) from exc
        except Exception as exc:
            logger.exception("TraceRAG query failed")
            raise HTTPException(
                status_code=502,
                detail="The embedding or chat provider request failed.",
            ) from exc
        return _query_response(result)

    return router
