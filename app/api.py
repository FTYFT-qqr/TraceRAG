"""FastAPI routes for document ingestion and RAG queries."""

from __future__ import annotations

import logging
import threading
from typing import Callable

from fastapi import APIRouter, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

from app.config import Settings
from app.models import QueryResponse, UnsupportedDocumentError
from app.runtime import RAGRuntime


logger = logging.getLogger(__name__)
MAX_UPLOAD_BYTES = 10 * 1024 * 1024
RuntimeFactory = Callable[[Settings], RAGRuntime]


class DocumentUploadResponse(BaseModel):
    document_id: str
    file_name: str
    page_count: int
    chunk_count: int
    index_generation: str
    already_indexed: bool


class QueryRequest(BaseModel):
    query: str = Field(min_length=1, max_length=8000)
    top_k: int = Field(default=5, ge=1, le=20)


class CitationResponse(BaseModel):
    chunk_id: str
    file_name: str
    page_number: int | None
    text: str


class RetrievalResponse(BaseModel):
    chunk_id: str
    file_name: str
    page_number: int | None
    text: str
    score: float


class QueryApiResponse(BaseModel):
    answer: str
    rejected: bool
    citations: list[CitationResponse]
    retrievals: list[RetrievalResponse]


def _query_response(response: QueryResponse) -> QueryApiResponse:
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
            )
            for item in response.results
        ],
    )


def create_api_router(
    settings: Settings,
    *,
    runtime_factory: RuntimeFactory | None = None,
) -> APIRouter:
    """Create the two V0.1 data routes with lazy provider initialization."""

    router = APIRouter()
    factory = runtime_factory or RAGRuntime.from_settings
    runtime: RAGRuntime | None = None
    runtime_lock = threading.RLock()

    def get_runtime() -> RAGRuntime:
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
                except Exception as exc:
                    logger.exception("Could not initialize the TraceRAG runtime")
                    raise HTTPException(
                        status_code=503,
                        detail="TraceRAG runtime is unavailable; check provider credentials and index data.",
                    ) from exc
            return runtime

    @router.post("/documents", response_model=DocumentUploadResponse, tags=["documents"])
    def upload_document(file: UploadFile = File(...)) -> DocumentUploadResponse:
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
        )

    @router.post("/query", response_model=QueryApiResponse, tags=["query"])
    def query_documents(request: QueryRequest) -> QueryApiResponse:
        active_runtime = get_runtime()
        if not request.query.strip():
            raise HTTPException(status_code=422, detail="Query cannot be empty.")
        try:
            with runtime_lock:
                result = active_runtime.query(request.query, top_k=request.top_k)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except Exception as exc:
            logger.exception("TraceRAG query failed")
            raise HTTPException(
                status_code=502,
                detail="The embedding or chat provider request failed.",
            ) from exc
        return _query_response(result)

    return router
