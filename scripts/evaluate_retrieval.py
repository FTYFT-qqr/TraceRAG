"""用项目样本文档和当前真实模型生成 V0.1 检索、引用与拒答验收记录。"""

from __future__ import annotations

import json
import os
import re
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
from dotenv import load_dotenv

from app.chat import ChatClient
from app.chunking import chunk_pages
from app.config import Settings
from app.document_loader import load_document
from app.embeddings import EmbeddingClient, SentenceTransformerEmbeddingClient
from app.models import Chunk
from app.rag import RAGService
from app.retriever import VectorRetriever
from app.vector_store import FaissVectorStore


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CORPUS_DIR = PROJECT_ROOT / "samples" / "acceptance"
QUESTION_FILE = CORPUS_DIR / "questions.json"
REPORT_FILE = PROJECT_ROOT / "docs" / "评测数据" / "版本0.1" / "验收记录" / "检索验收记录.md"
EMBEDDING_BATCH_SIZE = 64
def _load_settings() -> Settings:
    """从项目根目录读取配置，但不输出任何密钥字段。"""
    load_dotenv(PROJECT_ROOT / ".env", override=False)
    return Settings.from_environment()


def _create_embedder(settings: Settings) -> Any:
    """按项目配置创建真实的本地或兼容 API Embedding 客户端。"""
    if settings.embedding_provider == "local":
        return SentenceTransformerEmbeddingClient(
            model_path=settings.local_embedding_path,
            device=settings.embedding_device,
        )
    return EmbeddingClient(
        api_key=settings.api_key,
        base_url=settings.base_url,
        proxy_url=settings.api_proxy_url,
        model=settings.embedding_model,
    )


def _load_corpus() -> list[Chunk]:
    """通过正式的 Loader 和 Chunk Pipeline 读取全部验收文档。"""
    chunks = []
    for path in sorted(CORPUS_DIR.glob("*.txt")):
        pages = load_document(path.name, path.read_bytes())
        chunks.extend(chunk_pages(pages))
    if not chunks:
        raise RuntimeError(f"No evaluation documents found in {CORPUS_DIR}.")
    return chunks



from app.evaluation import assess_case as _assess_case

def _acceptance_passes(
    cases: list[dict[str, Any]], outcomes: list[dict[str, Any]]
) -> bool:
    """检索门槛之外，还要求每题回答质量达标且没有任何单题错误。"""

    if len(outcomes) != len(cases) or any(outcome.get("error") for outcome in outcomes):
        return False
    hits = sum(
        bool(outcome.get("expected_evidence_hit"))
        for outcome in outcomes
        if outcome["answerable"]
    )
    return hits >= 10 and all(outcome.get("quality_pass") for outcome in outcomes)


def _markdown_report(
    settings: Settings,
    cases: list[dict[str, Any]],
    outcomes: list[dict[str, Any]],
) -> str:
    """保留逐题预期来源、Top-5 命中、拒答结果和可人工核对的引用片段。"""
    answerable = [case for case in cases if case["answerable"]]
    unanswerable = [case for case in cases if not case["answerable"]]
    hits = sum(bool(result.get("expected_evidence_hit")) for result in outcomes if result["answerable"])
    answer_quality = sum(
        bool(result.get("quality_pass")) for result in outcomes if result["answerable"]
    )
    refusals = sum(
        bool(result.get("quality_pass")) for result in outcomes if not result["answerable"]
    )
    errors = sum(bool(result.get("error")) for result in outcomes)
    pending_reviews = sum(
        result.get("manual_review_status") == "待人工复核" for result in outcomes
    )
    all_questions_pass = (
        len(outcomes) == len(cases)
        and all(result.get("quality_pass") for result in outcomes)
        and errors == 0
    )
    passed = hits >= 10 and all_questions_pass

    model_label = (
        Path(settings.local_embedding_path).name
        if settings.embedding_provider == "local" and settings.local_embedding_path
        else settings.embedding_model
    )
    lines = [
        "# TraceRAG V0.1 检索与拒答验收记录",
        "",
        f"- 日期：{date.today().isoformat()}",
        f"- 语料：`samples/acceptance/` 内 {len(list(CORPUS_DIR.glob('*.txt')))} 份项目样本文档；问题均为人工编写。",
        f"- Embedding：`{settings.embedding_provider}` / `{model_label}`。",
        f"- 拒答阈值：`{settings.reject_threshold}`；Top-K：`5`。",
        f"- 有答案问题：{hits}/{len(answerable)} 条在 Top-5 找到预期证据（目标至少 10 条）。",
        f"- 有答案回答质量：{answer_quality}/{len(answerable)} 条同时正常作答、匹配预期事实，且引用可追溯并支持事实。",
        f"- 无答案问题：{refusals}/{len(unanswerable)} 条正确拒答且未附引用（目标 3 条）。",
        f"- 单题执行错误：{errors} 条（必须为 0）。",
        f"- 待人工复核：{pending_reviews} 条（全部核清前不能通过）。",
        f"- 指标结果：{'通过' if passed else '未通过'}。",
        "- 本记录使用真实 Embedding 与 Chat 配置；只发送本目录的公开测试语料和问题，不读取用户上传目录。",
        "- 下面同时保留实际回答和引用片段，供人工核对事实与来源是否对应；编号映射本身不代表通用事实核验。",
        "",
        "## 逐题结果",
        "",
    ]

    outcome_by_id = {result["id"]: result for result in outcomes}
    for case in cases:
        result = outcome_by_id[case["id"]]
        lines.extend(
            [
                f"### {case['id']} — {'有答案' if case['answerable'] else '无答案'}",
                f"- 问题：{case['query']}",
                f"- 预期来源：{case['expected_source']}",
                f"- 预期证据 Top-5 命中：{'是' if result.get('expected_evidence_hit') else '否'}",
                f"- 命中排名：{result.get('expected_evidence_rank', '无')}",
                f"- 最终供模型参考的片段数：{result.get('provided_chunk_count', 0)}",
                f"- Top-1 分数：{result.get('top1_score', '无')}",
                f"- 系统拒答：{'是' if result.get('rejected') else '否'}",
            ]
        )
        if result.get("error"):
            lines.append(f"- 执行错误：`{result['error']}`")
        else:
            lines.append(f"- 回答：{result.get('answer', '')}")
            if case["answerable"]:
                automatic_fact_result = result.get("answer_correct")
                if automatic_fact_result is True:
                    automatic_fact_label = "正确"
                elif automatic_fact_result is False:
                    automatic_fact_label = "错误"
                else:
                    automatic_fact_label = "待人工复核"
                lines.extend(
                    [
                        f"- 预期事实：{'；'.join(' / '.join(group) for group in case.get('expected_answer_facts', []))}",
                        f"- 正常作答：{'是' if result.get('normal_answer') else '否'}",
                        f"- 预期事实短语覆盖：{'是' if result.get('expected_fact_phrases_match') else '否'}",
                        f"- 自动事实核验结论：{automatic_fact_label}",
                        f"- 自动核验/人工复核状态：{result.get('manual_review_status', '未知')}",
                        f"- 相反事实：{'是' if result.get('contradictory_fact') else '否'}",
                        f"- 引用不支持的数值：{'；'.join(result.get('unsupported_numeric_claims', [])) or '无'}",
                        f"- 待人工复核子句：{'；'.join(result.get('unverified_claims', [])) or '无'}",
                        f"- 引用映射有效：{'是' if result.get('citations_traceable') else '否'}",
                        f"- 全部预期事实均受引用支持：{'是' if result.get('expected_facts_supported_by_citations') else '否'}",
                        f"- 引用支持预期事实：{'是' if result.get('citation_supports_expected_evidence') else '否'}",
                        f"- 本题质量验收：{'通过' if result.get('quality_pass') else '失败'}",
                    ]
                )
            else:
                lines.extend(
                    [
                        f"- 明确拒答文本：{'是' if result.get('refusal_text_present') else '否'}",
                        f"- 回答含引用编号：{'是' if result.get('answer_has_citation') else '否'}",
                        f"- 正确拒答且未附引用：{'是' if result.get('quality_pass') else '否'}",
                        f"- 本题质量验收：{'通过' if result.get('quality_pass') else '失败'}",
                    ]
                )
            citations = result.get("citations", [])
            if citations:
                lines.append("- 引用来源与片段：")
                for citation in citations:
                    lines.append(
                        f"  - `{citation['file_name']}`：{citation['text']}"
                    )
            else:
                lines.append("- 引用来源与片段：无")
        lines.append("")
    return "\n".join(lines)


def main() -> int:
    """运行隔离的真实模型验收，不写入应用使用的 FAISS 索引目录。"""
    os.chdir(PROJECT_ROOT)
    settings = _load_settings()
    if not settings.api_key:
        raise RuntimeError("Set TRACERAG_API_KEY to run the Chat refusal evaluation.")

    cases = json.loads(QUESTION_FILE.read_text(encoding="utf-8"))
    answerable_count = sum(bool(case["answerable"]) for case in cases)
    unanswerable_count = len(cases) - answerable_count
    if answerable_count < 12 or unanswerable_count < 3:
        raise ValueError("The evaluation set needs at least 12 answerable and 3 unanswerable questions.")
    if any(
        case["answerable"] and not case.get("expected_answer_facts")
        for case in cases
    ):
        raise ValueError("Every answerable case needs explicit expected answer facts.")

    embedder = _create_embedder(settings)
    chunks = _load_corpus()
    store = FaissVectorStore()
    vector_batches = [
        chunks[start : start + EMBEDDING_BATCH_SIZE]
        for start in range(0, len(chunks), EMBEDDING_BATCH_SIZE)
    ]
    vectors = np.concatenate(
        [embedder.embed_texts([chunk.content for chunk in batch]) for batch in vector_batches],
        axis=0,
    )
    store.add(chunks, vectors)

    retriever = VectorRetriever(embedder, store)
    # 完整拒答验收需要经过真实 Chat 模型；Embedding 本身仍使用项目当前配置。
    chat_client = ChatClient(
        api_key=settings.api_key,
        base_url=settings.base_url,
        proxy_url=settings.api_proxy_url,
        model=settings.chat_model,
    )
    rag = RAGService(retriever, chat_client, reject_threshold=settings.reject_threshold)
    outcomes: list[dict[str, Any]] = []
    for case in cases:
        try:
            response = rag.query(case["query"], top_k=5)
            outcome = _assess_case(case, response)
        except Exception as exc:  # 将单题失败写入报告，继续保留其余验收结果。
            outcome = _assess_case(case, None, error=type(exc).__name__)
        outcomes.append(outcome)

    REPORT_FILE.write_text(
        _markdown_report(settings, cases, outcomes), encoding="utf-8"
    )
    hits = sum(bool(result["expected_evidence_hit"]) for result in outcomes if result["answerable"])
    answer_quality = sum(bool(result["quality_pass"]) for result in outcomes if result["answerable"])
    refusals = sum(bool(result["quality_pass"]) for result in outcomes if not result["answerable"])
    errors = sum(bool(result.get("error")) for result in outcomes)
    passed = _acceptance_passes(cases, outcomes)
    print(f"Report: {REPORT_FILE}")
    print(
        f"Top-5 evidence: {hits}/{answerable_count}; "
        f"answer quality: {answer_quality}/{answerable_count}; "
        f"correct refusals: {refusals}/{unanswerable_count}; errors: {errors}"
    )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
