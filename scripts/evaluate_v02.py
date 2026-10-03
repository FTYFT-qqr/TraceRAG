"""生成 V0.2 可复现检索基线和单变量对照报告。"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from dataclasses import asdict, dataclass, replace
from datetime import date
from pathlib import Path
from typing import Any, Protocol
from urllib.parse import urlsplit, urlunsplit

import numpy as np
from dotenv import load_dotenv

# 允许 README 中的 `python scripts/evaluate_v02.py` 直接运行，
# 因为 Python 默认只把 scripts 目录加入模块搜索路径。
if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.chat import ChatClient
from app.chunking import chunk_pages
from app.config import Settings
from app.document_loader import load_document
from app.chat import SYSTEM_PROMPT
from app.embeddings import EmbeddingClient, SentenceTransformerEmbeddingClient
from app.evaluation import assess_case
from app.models import RetrievalResult
from app.rag import RAGService
from app.relevance import RelevancePolicy
from app.retriever import BM25Retriever, HybridRetriever, VectorRetriever
from app.vector_store import FaissVectorStore


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CORPUS_DIR = PROJECT_ROOT / "samples" / "acceptance"
DEFAULT_QUESTIONS = CORPUS_DIR / "questions_v02.json"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "docs" / "评测数据" / "版本0.2" / "临时运行"
REPORT_FILE_NAMES = {
    "baseline-vector": "向量基线",
    "chunk-short-vector": "向量短切分",
    "chunk-long-vector": "向量长切分",
    "bm25-fixed-chunk": "BM25固定切分",
    "hybrid-rrf-fixed-chunk": "混合召回RRF",
}
EMBEDDING_BATCH_SIZE = 64
TOP_K_VALUES = (1, 3, 5)


@dataclass(frozen=True, slots=True)
class BenchmarkSpec:
    """描述一轮对照的唯一可变因素和其余固定参数。"""

    name: str
    strategy: str = "vector"
    chunk_size: int = 700
    overlap: int = 100
    rrf_k: int = 60
    embedding_batch_size: int = 1
    bm25_min_score: float = 0.0


class BatchRetriever(Protocol):
    """定义基准评测所需的批量检索接口，隔离底层检索实现。"""

    def retrieve_many(self, queries: list[str], top_k: int = 5) -> list[list[RetrievalResult]]:
        """批量返回与输入问题顺序一致的 Top-K 检索结果。"""
        ...


class PrecomputedRetriever:
    """适配预先算好的检索排名，供 RAG 回答验收复用同一证据。"""

    def __init__(self, results_by_query: dict[str, list[RetrievalResult]]) -> None:
        """保存每道问题的候选排名，不持有 Embedding 或向量索引。"""

        self._results_by_query = results_by_query

    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievalResult]:
        """按 RAG 请求的 Top-K 截取预计算结果，并保留其原始排名。"""

        if not query.strip():
            raise ValueError("Query cannot be empty.")
        if top_k <= 0:
            raise ValueError("top_k must be a positive integer.")
        if query not in self._results_by_query:
            raise KeyError("Query was not part of the precomputed evaluation batch.")
        return self._results_by_query[query][:top_k]


def _parse_args() -> argparse.Namespace:
    """解析基线、单轮对照或完整检索套件的命令行参数。"""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--strategy", choices=("vector", "bm25", "hybrid"), default="vector")
    parser.add_argument("--chunk-size", type=int, default=700)
    parser.add_argument("--overlap", type=int, default=100)
    parser.add_argument("--rrf-k", type=int, default=60)
    parser.add_argument("--bm25-min-score", type=float, default=0.0,
                        help="BM25 原始分数须严格高于此值才可生成回答；不使用余弦门槛")
    parser.add_argument("--name", default=None, help="报告名称；默认按策略和切分参数生成")
    parser.add_argument("--quality", action="store_true", help="调用 Chat 模型执行回答质量与拒答验收")
    parser.add_argument("--suite", action="store_true", help="运行固定切分、短切分、长切分、BM25 和混合召回对照")
    parser.add_argument("--embedding-model", default=None, help="仅用于远程 Embedding 的替代模型名")
    parser.add_argument("--embedding-path", type=Path, default=None, help="替代本地 Embedding 模型目录")
    parser.add_argument("--compare-embedding-model", default=None, help="在当前模型之外增加远程 Embedding 对照")
    parser.add_argument("--compare-embedding-path", type=Path, default=None, help="在当前模型之外增加本地 Embedding 对照")
    parser.add_argument("--embedding-batch-size", type=int, default=1, help="固定 Embedding 推理批次大小")
    parser.add_argument("--compare-embedding-batch-size", type=int, default=None, help="在相同 Embedding 模型下对照另一批次大小")
    return parser.parse_args()


def load_cases(path: Path) -> list[dict[str, Any]]:
    """加载并严格校验 V0.2 固定评测集 schema。"""

    cases = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(cases, list) or len(cases) != 50:
        raise ValueError("V0.2 evaluation set must contain exactly 50 cases.")
    required = {
        "id",
        "query",
        "answerable",
        "expected_source",
        "expected_evidence",
        "expected_answer_facts",
        "allow_citations",
        "refusal_expected",
        "manual_review_condition",
        "tags",
    }
    ids: set[str] = set()
    for case in cases:
        if not isinstance(case, dict) or not required.issubset(case):
            raise ValueError("Every V0.2 case must contain the complete acceptance schema.")
        case_id = case["id"]
        if not isinstance(case_id, str) or not case_id or case_id in ids:
            raise ValueError("Evaluation case IDs must be non-empty and unique.")
        ids.add(case_id)
        if not isinstance(case["query"], str) or not case["query"].strip():
            raise ValueError(f"{case_id} has an empty query.")
        if not isinstance(case["tags"], list) or not case["tags"]:
            raise ValueError(f"{case_id} must have at least one category tag.")
        if case["answerable"]:
            evidence = case["expected_evidence"]
            if isinstance(evidence, str):
                evidence = [evidence]
            if not evidence or not case["expected_answer_facts"]:
                raise ValueError(f"{case_id} needs evidence and answer facts.")
            if case["refusal_expected"] or not case["allow_citations"]:
                raise ValueError(f"{case_id} has inconsistent answerable flags.")
        else:
            if case["expected_evidence"] is not None or case["expected_answer_facts"]:
                raise ValueError(f"{case_id} must not define answer evidence.")
            if not case["refusal_expected"] or case["allow_citations"]:
                raise ValueError(f"{case_id} has inconsistent refusal flags.")
    if sum(bool(case["answerable"]) for case in cases) < 30:
        raise ValueError("The V0.2 set must contain at least 30 answerable cases.")
    if sum(not bool(case["answerable"]) for case in cases) < 5:
        raise ValueError("The V0.2 set must contain at least 5 refusal cases.")
    if len({case["query"] for case in cases}) != len(cases):
        raise ValueError("Evaluation case queries must be unique for precomputed ranking reuse.")
    return cases


def _load_pages() -> list:
    """通过正式文档 Loader 读取固定样本文档。"""

    pages = []
    for path in sorted(CORPUS_DIR.glob("*.txt")):
        pages.extend(load_document(path.name, path.read_bytes()))
    if not pages:
        raise RuntimeError(f"No evaluation documents found in {CORPUS_DIR}.")
    return pages


def validate_cases_against_corpus(cases: list[dict[str, Any]]) -> None:
    """确认每道可回答题的预期来源和全部证据短语都存在于原文。"""

    source_texts = {
        path.name: "\n".join(
            page.content for page in load_document(path.name, path.read_bytes())
        )
        for path in sorted(CORPUS_DIR.glob("*.txt"))
    }
    unknown_sources = sorted(
        {case["expected_source"] for case in cases} - source_texts.keys()
    )
    if unknown_sources:
        raise ValueError(f"Evaluation set references unknown sources: {unknown_sources}.")
    mismatches = []
    for case in cases:
        if not case["answerable"]:
            continue
        raw_evidence = case["expected_evidence"]
        evidence_items = [raw_evidence] if isinstance(raw_evidence, str) else raw_evidence
        if not isinstance(evidence_items, list) or not evidence_items:
            mismatches.append(case["id"])
            continue
        if any(
            not isinstance(evidence, str)
            or evidence not in source_texts[case["expected_source"]]
            for evidence in evidence_items
        ):
            mismatches.append(case["id"])
    if mismatches:
        raise ValueError(f"Expected evidence is absent from source documents: {mismatches}.")


def _build_chunks(pages: list, spec: BenchmarkSpec) -> list:
    """按本轮切分参数构造完全隔离的 Chunk 列表。"""

    return chunk_pages(pages, chunk_size=spec.chunk_size, overlap=spec.overlap)


def _create_embedder(settings: Settings, *, batch_size: int) -> Any:
    """按冻结配置创建本地或兼容 API Embedding 客户端。"""

    if settings.embedding_provider == "local":
        return SentenceTransformerEmbeddingClient(
            model_path=settings.local_embedding_path,
            device=settings.embedding_device,
            batch_size=batch_size,
        )
    return EmbeddingClient(
        api_key=settings.api_key,
        base_url=settings.base_url,
        proxy_url=settings.api_proxy_url,
        model=settings.embedding_model,
    )


def _build_vector_store(chunks: list, embedder: Any) -> FaissVectorStore:
    """只在内存中建立本轮向量索引，不触碰应用 indexes 目录。"""

    store = FaissVectorStore()
    vectors: list[np.ndarray] = []
    for start in range(0, len(chunks), EMBEDDING_BATCH_SIZE):
        batch = chunks[start : start + EMBEDDING_BATCH_SIZE]
        vectors.append(embedder.embed_texts([chunk.content for chunk in batch]))
    if vectors:
        store.add(chunks, np.concatenate(vectors, axis=0))
    return store


def build_retriever(
    spec: BenchmarkSpec,
    chunks: list,
    embedder: Any | None = None,
    vector_store: FaissVectorStore | None = None,
):
    """按策略组装独立检索器，只为向量策略创建稠密索引。"""

    if spec.strategy == "bm25":
        return BM25Retriever(chunks)
    if vector_store is None:
        if embedder is None:
            raise ValueError("Vector and hybrid strategies require an embedder.")
        vector_store = _build_vector_store(chunks, embedder)
    if embedder is None:
        raise ValueError("Vector and hybrid strategies require an embedder.")
    vector = VectorRetriever(embedder, vector_store)
    if spec.strategy == "vector":
        return vector
    sparse = BM25Retriever(chunks)
    return HybridRetriever(vector, sparse, rrf_k=spec.rrf_k)


def _make_chat(settings: Settings) -> ChatClient:
    """创建质量验收所需 Chat 客户端，统一沿用 V0.1 的提示约束。"""

    if not settings.api_key:
        raise RuntimeError("--quality requires TRACERAG_API_KEY or OPENAI_API_KEY.")
    return ChatClient(
        api_key=settings.api_key,
        base_url=settings.base_url,
        proxy_url=settings.api_proxy_url,
        model=settings.chat_model,
    )


def _precompute_rankings(
    retriever: BatchRetriever, cases: list[dict[str, Any]], *, top_k: int
) -> dict[str, list[RetrievalResult]]:
    """一次批量检索全套问题，供指标和回答质量评估共用排名。"""

    queries = [case["query"] for case in cases]
    rankings = retriever.retrieve_many(queries, top_k=top_k)
    if len(rankings) != len(cases):
        raise ValueError("Batch retriever returned a different number of rankings than queries.")
    return dict(zip(queries, rankings))


def _find_expected_rank(case: dict[str, Any], results: list) -> int | None:
    """用原始候选名次查找预期证据，避免对照报告虚报 Top-K。"""

    if not case["answerable"] or not case["expected_evidence"]:
        return None
    evidence_items = (
        [case["expected_evidence"]]
        if isinstance(case["expected_evidence"], str)
        else case["expected_evidence"]
    )
    evidence_ranks = []
    for evidence in evidence_items:
        matching_ranks = [
            result.candidate_rank or rank
            for rank, result in enumerate(results, start=1)
            if result.chunk.file_name == case["expected_source"]
            and evidence in result.chunk.content
        ]
        if matching_ranks:
            evidence_ranks.append(min(matching_ranks))
    if len(evidence_ranks) != len(evidence_items):
        return None
    return max(evidence_ranks)


def _retrieval_outcome(case: dict[str, Any], results: list) -> dict[str, Any]:
    """保存每题完整排名，并计算 Top-1、Top-3、Top-5 命中。"""

    expected_rank = _find_expected_rank(case, results)
    return {
        "expected_evidence_rank": expected_rank,
        "hit_top1": bool(expected_rank is not None and expected_rank <= 1),
        "hit_top3": bool(expected_rank is not None and expected_rank <= 3),
        "hit_top5": bool(expected_rank is not None and expected_rank <= 5),
        "ranking": [
            {
                "rank": result.candidate_rank or rank,
                "chunk_id": result.chunk.chunk_id,
                "file_name": result.chunk.file_name,
                "page_number": result.chunk.page_number,
                "score": round(float(result.score), 8),
                "score_kind": result.score_kind,
                "vector_score": result.vector_score,
                "bm25_score": result.bm25_score,
                "text": result.chunk.content,
            }
            for rank, result in enumerate(results, start=1)
        ],
    }


def run_benchmark(
    settings: Settings,
    cases: list[dict[str, Any]],
    spec: BenchmarkSpec,
    *,
    quality: bool,
) -> dict[str, Any]:
    """运行一轮检索和可选回答质量评测，返回可序列化的逐题结果。"""

    pages = _load_pages()
    chunks = _build_chunks(pages, spec)
    embedder = (
        _create_embedder(settings, batch_size=spec.embedding_batch_size)
        if spec.strategy in {"vector", "hybrid"}
        else None
    )
    vector_store = _build_vector_store(chunks, embedder) if embedder is not None else None
    retriever = build_retriever(spec, chunks, embedder, vector_store)
    rankings_by_query = _precompute_rankings(retriever, cases, top_k=len(chunks))
    chat = _make_chat(settings) if quality else None
    cached_retriever = PrecomputedRetriever(rankings_by_query)
    relevance_policy = RelevancePolicy(
        vector_min_score=settings.reject_threshold, bm25_min_score=spec.bm25_min_score
    )
    rag = RAGService(
        cached_retriever, chat, reject_threshold=settings.reject_threshold,
        relevance_policy=relevance_policy,
    ) if chat else None
    outcomes = []
    for case in cases:
        retrieval_results = rankings_by_query[case["query"]]
        outcome: dict[str, Any] = {
            "id": case["id"],
            **_retrieval_outcome(case, retrieval_results),
        }
        try:
            if rag is not None:
                # 回答质量始终使用同一套 Top-5 RAG 规则，方便跨轮比较。
                response = rag.query(case["query"], top_k=5)
                assessed = assess_case(case, response)
                for key in (
                    "rejected",
                    "normal_answer",
                    "answer_correct",
                    "expected_fact_phrases_match",
                    "expected_facts_supported_by_citations",
                    "citations_traceable",
                    "citation_source_scope_valid",
                    "unexpected_citation_sources",
                    "citation_claims_supported",
                    "citation_claims",
                    "contradictory_fact",
                    "citation_supports_expected_evidence",
                    "quality_pass",
                    "manual_review_status",
                    "answer",
                    "citations",
                    "provided_chunk_count",
                    "unsupported_numeric_claims",
                    "unverified_claims",
                    "refusal_text_present",
                    "answer_has_citation",
                    "error",
                ):
                    if key in assessed:
                        outcome[key] = assessed[key]
            else:
                outcome.update(
                    {
                        "quality_status": "未执行",
                        "quality_pass": None,
                        "manual_review_status": "未执行",
                    }
                )
        except Exception as exc:
            # Chat/API 失败只标记质量评估，不丢弃已经取得的检索排名。
            outcome["quality_error"] = f"{type(exc).__name__}: {exc}"
            outcome["quality_pass"] = False
            outcome["manual_review_status"] = "执行错误"
        outcomes.append(outcome)

    answerable = [case for case in cases if case["answerable"]]
    by_id = {outcome["id"]: outcome for outcome in outcomes}
    errors = sum(
        bool(outcome.get("error") or outcome.get("quality_error"))
        for outcome in outcomes
    )
    metrics = {
        "top1_hits": sum(by_id[case["id"]].get("hit_top1", False) for case in answerable),
        "top3_hits": sum(by_id[case["id"]].get("hit_top3", False) for case in answerable),
        "top5_hits": sum(by_id[case["id"]].get("hit_top5", False) for case in answerable),
        "answerable_count": len(answerable),
        "quality_pass": sum(bool(by_id[case["id"]].get("quality_pass")) for case in answerable),
        "answer_fact_match": sum(
            bool(by_id[case["id"]].get("expected_fact_phrases_match"))
            for case in answerable
        ),
        "citation_traceable": sum(
            bool(by_id[case["id"]].get("citations_traceable"))
            for case in answerable
        ),
        "citation_original_support": sum(
            bool(by_id[case["id"]].get("expected_facts_supported_by_citations"))
            and by_id[case["id"]].get("citation_source_scope_valid") is True
            and by_id[case["id"]].get("citation_claims_supported") is True
            for case in answerable
        ),
        "expected_evidence_cited": sum(
            bool(by_id[case["id"]].get("citation_supports_expected_evidence"))
            for case in answerable
        ),
        "refusal_pass": sum(
            bool(by_id[case["id"]].get("quality_pass"))
            for case in cases
            if not case["answerable"]
        ),
        "refusal_count": sum(not case["answerable"] for case in cases),
        "manual_review_count": sum(
            by_id[case["id"]].get("manual_review_status") == "待人工复核"
            for case in cases
        ),
        "errors": errors,
    }
    metrics["retrieval_pass"] = metrics["top5_hits"] >= 10
    metrics["quality_executed"] = quality
    metrics["acceptance_pass"] = bool(
        quality
        and metrics["retrieval_pass"]
        and metrics["quality_pass"] == metrics["answerable_count"]
        and metrics["refusal_pass"] == metrics["refusal_count"]
        and metrics["manual_review_count"] == 0
        and errors == 0
    )
    return {
        "schema_version": 1,
        "relevance_policy": {
            **asdict(relevance_policy),
            "bm25_rule": "raw_score_strictly_above_minimum",
            "hybrid_rule": "vector_or_bm25_support; rrf_score_is_ranking_only",
        },
        "generated_on": date.today().isoformat(),
        "spec": asdict(spec),
        "chunk_count": len(chunks),
        "embedding_dimension": vector_store.dimension if vector_store is not None else None,
        "cases": outcomes,
        "metrics": metrics,
    }


def _safe_settings_snapshot(settings: Settings) -> dict[str, Any]:
    """生成不含 API 密钥的配置快照，保证报告可复现且不泄密。"""

    identity = settings.embedding_identity.to_dict()
    parsed_base_url = urlsplit(settings.base_url or "https://api.openai.com/v1")
    hostname = parsed_base_url.hostname or ""
    if ":" in hostname and not hostname.startswith("["):
        hostname = f"[{hostname}]"
    try:
        port = parsed_base_url.port
    except ValueError:
        port = None
    if port and not (
        (parsed_base_url.scheme.lower() == "https" and port == 443)
        or (parsed_base_url.scheme.lower() == "http" and port == 80)
    ):
        hostname = f"{hostname}:{port}"
    safe_base_url = urlunsplit(
        (parsed_base_url.scheme, hostname, parsed_base_url.path.rstrip("/"), "", "")
    )
    return {
        "embedding_provider": settings.embedding_provider,
        "embedding_model": settings.embedding_model,
        "local_embedding_path": settings.local_embedding_path,
        "embedding_identity": identity,
        "embedding_device": settings.embedding_device,
        "chat_model": settings.chat_model,
        "reject_threshold": settings.reject_threshold,
        "base_url": safe_base_url,
        "application_index_used": False,
        "evaluation_index_backend": "FAISS IndexFlatIP / process-local",
        "chat_prompt_sha256": hashlib.sha256(SYSTEM_PROMPT.encode("utf-8")).hexdigest(),
        "answer_protocol": "original-passage-selection-v1",
        # 提示相同时，短句切分或评分代码变化也会影响结果，单独固定实现指纹。
        "implementation_sha256": {
            f"app/{name}.py": _file_sha256(PROJECT_ROOT / "app" / f"{name}.py")
            for name in ("chat", "grounded_answer", "rag", "evaluation", "citations", "relevance", "retriever", "chunking")
        },
    }


def _file_sha256(path: Path) -> str:
    """计算评测语料或题集的 SHA-256，固定报告输入版本。"""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _markdown_report(
    settings: Settings,
    questions_path: Path,
    cases: list[dict[str, Any]],
    result: dict[str, Any],
) -> str:
    """输出含配置、汇总和逐题完整排名的 Markdown 报告。"""

    spec = result["spec"]
    metrics = result["metrics"]
    answerable_count = metrics["answerable_count"]
    lines = [
        "# TraceRAG V0.2 评测基线与检索对照记录",
        "",
        f"- 日期：{result['generated_on']}",
        f"- 评测集：`{questions_path.as_posix()}`，共 {len(cases)} 题（有答案 {answerable_count}，无答案 {metrics['refusal_count']}）。",
        f"- 本轮：`{spec['name']}`；策略 `{spec['strategy']}`；Chunk `{spec['chunk_size']}/{spec['overlap']}`；RRF-K `{spec['rrf_k']}`。",
        f"- Embedding：`{settings.embedding_provider}` / `{settings.embedding_model}`；模型身份：`{settings.embedding_identity.to_dict()}`。",
        f"- 语料 Chunk 数：{result['chunk_count']}；Embedding 维度：{result['embedding_dimension'] or '无（仅稀疏检索）'}；应用生产索引目录未被写入。",
        f"- Top-1：{metrics['top1_hits']}/{answerable_count}；Top-3：{metrics['top3_hits']}/{answerable_count}；Top-5：{metrics['top5_hits']}/{answerable_count}。",
        f"- 预期事实覆盖：{metrics['answer_fact_match']}/{answerable_count if metrics['quality_executed'] else '未执行'}；引用映射可追溯：{metrics['citation_traceable']}/{answerable_count if metrics['quality_executed'] else '未执行'}；引用原文支持全部预期事实：{metrics['citation_original_support']}/{answerable_count if metrics['quality_executed'] else '未执行'}；引用覆盖全部预期证据：{metrics['expected_evidence_cited']}/{answerable_count if metrics['quality_executed'] else '未执行'}。",
        f"- 回答质量：{metrics['quality_pass']}/{answerable_count}（{'已执行' if metrics['quality_executed'] else '未执行'}）。",
        f"- 正确拒答：{metrics['refusal_pass']}/{metrics['refusal_count']}（{'已执行' if metrics['quality_executed'] else '未执行'}）。",
        f"- 待人工复核：{metrics['manual_review_count'] if metrics['quality_executed'] else '未执行'}；执行错误：{metrics['errors']}。",
        f"- 本轮验收：{'通过' if metrics['acceptance_pass'] else '未通过/未完成质量执行'}。",
        "",
        "## 可复现配置",
        "",
        "```json",
        json.dumps({"settings": _safe_settings_snapshot(settings), "spec": spec,
                    "relevance_policy": result.get("relevance_policy")}, ensure_ascii=False, indent=2),
        "```",
        "",
        "## 逐题结果",
        "",
    ]
    outcome_by_id = {outcome["id"]: outcome for outcome in result["cases"]}
    for case in cases:
        outcome = outcome_by_id[case["id"]]
        lines.extend(
            [
                f"### {case['id']} — {'有答案' if case['answerable'] else '无答案'}",
                f"- 问题：{case['query']}",
                f"- 标签：{', '.join(case['tags'])}",
                f"- 预期来源：{case['expected_source']}",
                f"- 预期证据：{json.dumps(case['expected_evidence'], ensure_ascii=False)}",
                f"- 预期证据排名：{outcome.get('expected_evidence_rank', '无')}",
                f"- Top-1/3/5：{'是' if outcome.get('hit_top1') else '否'} / {'是' if outcome.get('hit_top3') else '否'} / {'是' if outcome.get('hit_top5') else '否'}",
                f"- 回答质量：{outcome.get('quality_pass', '未执行')}",
                f"- 引用映射可追溯：{outcome.get('citations_traceable', '未执行')}；引用原文支持预期事实：{outcome.get('expected_facts_supported_by_citations', '未执行')}；引用支持全部预期证据：{outcome.get('citation_supports_expected_evidence', '未执行')}",
                f"- 人工复核条件：{case['manual_review_condition']}",
            ]
        )
        if outcome.get("error"):
            lines.append(f"- 执行错误：`{outcome['error']}`")
        if "answer" in outcome:
            lines.append(f"- 回答：{outcome['answer']}")
            lines.append(f"- 引用：{json.dumps(outcome.get('citations', []), ensure_ascii=False)}")
            lines.append(f"- 逐句引用支持：{json.dumps(outcome.get('citation_claims', []), ensure_ascii=False)}")
        lines.append("- 完整检索排名：")
        for item in outcome.get("ranking", []):
            lines.append(
                f"  - #{item['rank']} `{item['file_name']}` score={item['score']} "
                f"kind={item.get('score_kind', 'historical')} vector={item.get('vector_score')} "
                f"bm25={item.get('bm25_score')}: {item['text']}"
            )
        lines.append("")
    return "\n".join(lines)


def _report_file_stem(name: str) -> str:
    """将内置评测标识转换成中文报告文件名，并保留自定义名称。"""

    suffix = "-embedding-alternative"
    alternative = name.endswith(suffix)
    base_name = name[: -len(suffix)] if alternative else name
    stem = REPORT_FILE_NAMES.get(base_name)
    if stem is None:
        match = re.fullmatch(r"(vector|bm25|hybrid)-chunk-(\d+)-(\d+)", base_name)
        if match:
            strategy = {"vector": "向量", "bm25": "BM25", "hybrid": "混合召回"}[match[1]]
            stem = f"{strategy}切分{match[2]}_{match[3]}"
        else:
            stem = base_name
    return f"{stem}_备选嵌入配置" if alternative else stem


def _write_result(
    settings: Settings,
    questions_path: Path,
    cases: list[dict[str, Any]],
    result: dict[str, Any],
    output_dir: Path,
) -> tuple[Path, Path]:
    """写入逐题 JSON 和便于人工阅读的 Markdown 两份证据。"""

    output_dir.mkdir(parents=True, exist_ok=True)
    name = _report_file_stem(result["spec"]["name"])
    json_path = output_dir / f"{name}.json"
    markdown_path = output_dir / f"{name}.md"
    payload = {
        **result,
        "inputs": {
            "questions": questions_path.as_posix(),
            "questions_sha256": _file_sha256(questions_path),
            "corpus": {
                path.name: _file_sha256(path)
                for path in sorted(CORPUS_DIR.glob("*.txt"))
            },
        },
        "settings": _safe_settings_snapshot(settings),
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    markdown_path.write_text(_markdown_report(settings, questions_path, cases, payload), encoding="utf-8")
    return json_path, markdown_path


def _default_suite() -> list[BenchmarkSpec]:
    """返回 V0.2 要求的固定切分、BM25 和混合召回对照矩阵。"""

    return [
        BenchmarkSpec("baseline-vector", "vector", 700, 100),
        BenchmarkSpec("chunk-short-vector", "vector", 350, 50),
        BenchmarkSpec("chunk-long-vector", "vector", 1000, 150),
        BenchmarkSpec("bm25-fixed-chunk", "bm25", 700, 100),
        BenchmarkSpec("hybrid-rrf-fixed-chunk", "hybrid", 700, 100),
    ]


def _settings_from_args(args: argparse.Namespace) -> Settings:
    """从当前环境构造设置，并只允许显式替代 Embedding 参数。"""

    load_dotenv(PROJECT_ROOT / ".env", override=False)
    settings = Settings.from_environment()
    changes: dict[str, Any] = {}
    if args.embedding_model:
        changes["embedding_model"] = args.embedding_model
    if args.embedding_path:
        changes["embedding_provider"] = "local"
        changes["local_embedding_path"] = str(args.embedding_path)
    return replace(settings, **changes) if changes else settings


def _embedding_comparison_settings(
    settings: Settings, args: argparse.Namespace
) -> list[tuple[str, Settings, int]]:
    """按单变量原则可选生成一个替代 Embedding 配置。"""

    comparison_count = sum(
        value is not None
        for value in (
            args.compare_embedding_model,
            args.compare_embedding_path,
            args.compare_embedding_batch_size,
        )
    )
    if comparison_count > 1:
        raise ValueError("Compare only one embedding model or configuration at a time.")
    if args.embedding_batch_size <= 0:
        raise ValueError("Embedding batch size must be a positive integer.")
    if args.compare_embedding_model:
        if settings.embedding_provider != "openai":
            raise ValueError("Remote model comparison requires the baseline provider to be openai.")
        alternate = replace(settings, embedding_model=args.compare_embedding_model)
    elif args.compare_embedding_path:
        if settings.embedding_provider != "local":
            raise ValueError("Local path comparison requires the baseline provider to be local.")
        alternate = replace(settings, local_embedding_path=str(args.compare_embedding_path))
    elif args.compare_embedding_batch_size is not None:
        if settings.embedding_provider != "local":
            raise ValueError("Embedding batch size comparison is only available for the local provider.")
        if args.compare_embedding_batch_size <= 0:
            raise ValueError("Compared embedding batch size must be a positive integer.")
        if args.compare_embedding_batch_size == args.embedding_batch_size:
            raise ValueError("Compared embedding batch size must differ from the baseline.")
        return [
            ("baseline", settings, args.embedding_batch_size),
            ("embedding-alternative", settings, args.compare_embedding_batch_size),
        ]
    else:
        return [("baseline", settings, args.embedding_batch_size)]
    return [
        ("baseline", settings, args.embedding_batch_size),
        ("embedding-alternative", alternate, args.embedding_batch_size),
    ]


def _comparison_markdown(comparison: dict[str, Any]) -> str:
    """将同一评测集各对照轮次的增益和退化整理为人工可读摘要。"""

    lines = [
        "# TraceRAG V0.2 检索对照汇总",
        "",
        f"- 日期：{comparison['generated_on']}",
        f"- 参考基线：`{comparison['baseline']}`",
        "- 回答质量指标如标记为未执行，表示本轮仅评估本地检索，尚未进行 Chat 回答验收。",
        "",
        "| 轮次 | Top-1 | Top-3 | Top-5 | Top-5 新增命中 | Top-5 退化 | Top-5 Bad Case |",
        "| --- | ---: | ---: | ---: | --- | --- | --- |",
    ]
    for run in comparison["runs"]:
        metrics = run["metrics"]
        delta = run.get("delta_vs_baseline", {})
        lines.append(
            "| `{name}` | {top1}/{count} ({d1:+d}) | {top3}/{count} ({d3:+d}) | "
            "{top5}/{count} ({d5:+d}) | {gained} | {regressed} | {bad} |".format(
                name=run["name"],
                top1=metrics["top1_hits"],
                top3=metrics["top3_hits"],
                top5=metrics["top5_hits"],
                count=metrics["answerable_count"],
                d1=delta.get("top1_hits", 0),
                d3=delta.get("top3_hits", 0),
                d5=delta.get("top5_hits", 0),
                gained=", ".join(run.get("top5_gained_case_ids", [])) or "无",
                regressed=", ".join(run.get("top5_regressed_case_ids", [])) or "无",
                bad=", ".join(run.get("top5_bad_case_ids", [])) or "无",
            )
        )
    quality_runs = [
        run for run in comparison["runs"] if run["metrics"].get("quality_executed")
    ]
    if quality_runs:
        lines.extend(
            [
                "",
                "## 回答质量",
                "",
                "| 轮次 | 有答案题通过 | 引用可追溯 | 引用原文支持 | 正确拒答 | 待人工复核 | 整体验收 |",
                "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
            ]
        )
        for run in quality_runs:
            metrics = run["metrics"]
            lines.append(
                f"| `{run['name']}` | {metrics['quality_pass']}/{metrics['answerable_count']} | "
                f"{metrics['citation_traceable']}/{metrics['answerable_count']} | "
                f"{metrics['citation_original_support']}/{metrics['answerable_count']} | "
                f"{metrics['refusal_pass']}/{metrics['refusal_count']} | "
                f"{metrics['manual_review_count']} | "
                f"{'通过' if metrics['acceptance_pass'] else '未通过'} |"
            )
    lines.extend(["", "## 对照参数", ""])
    for run in comparison["runs"]:
        spec = run["spec"]
        lines.append(
            f"- `{run['name']}`：策略 `{spec['strategy']}`；Chunk "
            f"`{spec['chunk_size']}/{spec['overlap']}`；Embedding 批次大小 "
            f"`{spec['embedding_batch_size']}`。"
        )
    lines.extend(
        [
            "",
            "各轮逐题完整排名和模型配置分别保存在同名 JSON/Markdown 文件中；输入语料及题集 SHA-256 记录在 JSON 结果。",
            "",
        ]
    )
    return "\n".join(lines)


def build_comparison(
    results: list[dict[str, Any]], cases: list[dict[str, Any]]
) -> dict[str, Any]:
    """统一汇总单轮与多轮评测，保留基线的 Top-5 Bad Case。"""

    if not results:
        raise ValueError("At least one evaluation result is required.")
    baseline = results[0]
    baseline_metrics = baseline["metrics"]
    baseline_by_id = {item["id"]: item for item in baseline["cases"]}
    answerable_ids = {case["id"] for case in cases if case["answerable"]}
    comparison = {
        "generated_on": date.today().isoformat(),
        "baseline": baseline["spec"]["name"],
        "runs": [],
    }
    for result in results:
        outcome_by_id = {item["id"]: item for item in result["cases"]}
        comparison["runs"].append(
            {
                "name": result["spec"]["name"],
                "spec": result["spec"],
                "metrics": result["metrics"],
                "delta_vs_baseline": {
                    key: result["metrics"].get(key, 0) - baseline_metrics.get(key, 0)
                    for key in (
                        "top1_hits",
                        "top3_hits",
                        "top5_hits",
                        "answer_fact_match",
                        "citation_traceable",
                        "citation_original_support",
                        "refusal_pass",
                    )
                },
                "top5_gained_case_ids": sorted(
                    case_id
                    for case_id, outcome in outcome_by_id.items()
                    if case_id in answerable_ids
                    if not baseline_by_id[case_id].get("hit_top5") and outcome.get("hit_top5")
                ),
                "top5_regressed_case_ids": sorted(
                    case_id
                    for case_id, outcome in outcome_by_id.items()
                    if case_id in answerable_ids
                    if baseline_by_id[case_id].get("hit_top5") and not outcome.get("hit_top5")
                ),
                "top5_bad_case_ids": sorted(
                    case_id
                    for case_id, outcome in outcome_by_id.items()
                    if case_id in answerable_ids and not outcome.get("hit_top5")
                ),
            }
        )
    return comparison


def _evaluation_exit_code(results: list[dict[str, Any]], *, quality_requested: bool) -> int:
    """按检索门槛、质量验收和执行错误返回退出码，失败或未完成均非零。"""

    if not results:
        return 1
    quality_seen = False
    for result in results:
        metrics = result["metrics"]
        if metrics["errors"] or not metrics["retrieval_pass"]:
            return 1
        if metrics["quality_executed"]:
            quality_seen = True
            if (not metrics["acceptance_pass"] or metrics["manual_review_count"]
                    or metrics["quality_pass"] != metrics["answerable_count"]
                    or metrics["refusal_pass"] != metrics["refusal_count"]):
                return 1
    return 1 if quality_requested and not quality_seen else 0


def main() -> int:
    """执行单轮或完整套件，并以验收状态作为退出码。"""

    args = _parse_args()
    if args.chunk_size <= 0 or args.overlap < 0 or args.overlap >= args.chunk_size:
        raise ValueError("Chunk size must be positive and overlap must be smaller than chunk size.")
    settings = _settings_from_args(args)
    cases = load_cases(args.questions)
    validate_cases_against_corpus(cases)
    base_specs = _default_suite() if args.suite else [
        BenchmarkSpec(
            name=args.name or f"{args.strategy}-chunk-{args.chunk_size}-{args.overlap}",
            strategy=args.strategy,
            chunk_size=args.chunk_size,
            overlap=args.overlap,
            rrf_k=args.rrf_k,
        )
    ]
    settings_runs = _embedding_comparison_settings(settings, args)
    planned_runs = []
    for settings_label, run_settings, batch_size in settings_runs:
        for spec in base_specs:
            if settings_label != "baseline" and spec.strategy == "bm25":
                # BM25 不使用 Embedding；重复运行不会形成有效的单变量对照。
                continue
            run_spec = spec
            if settings_label != "baseline":
                run_spec = replace(spec, name=f"{spec.name}-embedding-alternative")
            run_spec = replace(run_spec, embedding_batch_size=batch_size,
                               bm25_min_score=getattr(args, "bm25_min_score", 0.0))
            planned_runs.append((run_settings, run_spec))
    all_results = []
    for run_index, (run_settings, spec) in enumerate(planned_runs):
        result = run_benchmark(
            run_settings,
            cases,
            spec,
            # 仅对固定参考基线运行 50 题回答验收，避免对照矩阵重复消耗 Chat API。
            quality=args.quality and run_index == 0,
        )
        json_path, markdown_path = _write_result(run_settings, args.questions, cases, result, args.output_dir)
        all_results.append(result)
        metrics = result["metrics"]
        print(
            f"{spec.name}: Top-1 {metrics['top1_hits']}/{metrics['answerable_count']}; "
            f"Top-3 {metrics['top3_hits']}/{metrics['answerable_count']}; "
            f"Top-5 {metrics['top5_hits']}/{metrics['answerable_count']}; "
            f"quality {metrics['quality_pass']}/{metrics['answerable_count'] if metrics['quality_executed'] else 'not-run'}; "
            f"refusals {metrics['refusal_pass']}/{metrics['refusal_count'] if metrics['quality_executed'] else 'not-run'}"
        )
        print(f"JSON: {json_path}")
        print(f"Markdown: {markdown_path}")
    comparison = build_comparison(all_results, cases)
    comparison_path = args.output_dir / "对照摘要.json"
    comparison_path.parent.mkdir(parents=True, exist_ok=True)
    comparison_path.write_text(json.dumps(comparison, ensure_ascii=False, indent=2), encoding="utf-8")
    comparison_markdown_path = args.output_dir / "对照摘要.md"
    comparison_markdown_path.write_text(_comparison_markdown(comparison), encoding="utf-8")
    return _evaluation_exit_code(all_results, quality_requested=args.quality)


if __name__ == "__main__":
    raise SystemExit(main())
