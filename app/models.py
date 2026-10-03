"""定义文档、向量身份、检索结果和问答响应等共享数据结构。"""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True, slots=True)
class DocumentPage:
    """保留来源信息的文档页或文本单元。"""

    document_id: str
    file_name: str
    page_number: int | None
    content: str


@dataclass(frozen=True, slots=True)
class Chunk:
    """可追溯到原始文件和页码的文本片段。"""

    chunk_id: str
    document_id: str
    content: str
    file_name: str
    page_number: int | None
    chunk_index: int

    def to_dict(self) -> dict[str, object]:
        """将数据类转换为可写入快照 JSON 的普通字典。"""
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, object]) -> "Chunk":
        """从索引快照还原片段，并规范化 JSON 中的字段类型。"""
        return cls(
            chunk_id=str(value["chunk_id"]),
            document_id=str(value["document_id"]),
            content=str(value["content"]),
            file_name=str(value["file_name"]),
            page_number=(int(value["page_number"]) if value["page_number"] is not None else None),
            chunk_index=int(value["chunk_index"]),
        )


@dataclass(frozen=True, slots=True)
class EmbeddingIdentity:
    """描述向量空间来源，不包含 API 密钥或其他凭据。"""

    provider: str
    model: str
    endpoint: str | None = None
    model_fingerprint: str | None = None

    def __post_init__(self) -> None:
        """校验向量身份字段，防止不同来源的模型共用索引。"""
        if self.provider not in {"openai", "local"}:
            raise ValueError("Embedding provider must be 'openai' or 'local'.")
        if not self.model.strip():
            raise ValueError("Embedding model identity cannot be empty.")
        if self.provider == "openai" and not self.endpoint:
            raise ValueError("Remote embedding identity must include its service endpoint.")
        if self.provider == "local" and not self.model_fingerprint:
            raise ValueError("Local embedding identity must include a model fingerprint.")

    def to_dict(self) -> dict[str, str]:
        """生成可写入 manifest 的稳定身份字段。"""
        identity = {"provider": self.provider, "model": self.model}
        if self.endpoint is not None:
            identity["endpoint"] = self.endpoint
        if self.model_fingerprint is not None:
            identity["model_fingerprint"] = self.model_fingerprint
        return identity

    @classmethod
    def from_dict(cls, value: object) -> "EmbeddingIdentity":
        """解析快照身份字段；缺字段时不把旧索引误认为兼容。"""
        if not isinstance(value, dict):
            raise ValueError("Saved vector index is missing its embedding identity.")
        provider = value.get("provider")
        model = value.get("model")
        if not isinstance(provider, str) or not isinstance(model, str):
            raise ValueError("Saved vector index has an invalid embedding identity.")
        endpoint = value.get("endpoint")
        fingerprint = value.get("model_fingerprint")
        if endpoint is not None and not isinstance(endpoint, str):
            raise ValueError("Saved vector index has an invalid embedding endpoint.")
        if fingerprint is not None and not isinstance(fingerprint, str):
            raise ValueError("Saved vector index has an invalid model fingerprint.")
        return cls(
            provider=provider,
            model=model,
            endpoint=endpoint,
            model_fingerprint=fingerprint,
        )


@dataclass(frozen=True, slots=True)
class RetrievalResult:
    """携带来源、排序分和各召回通道原始分数，避免混用拒答门槛。"""

    chunk: Chunk
    score: float
    # 保留内部扩展候选中的原始名次，便于报告准确统计 Top-K 命中。
    candidate_rank: int | None = None
    score_kind: str = "cosine"
    vector_score: float | None = None
    bm25_score: float | None = None


@dataclass(frozen=True, slots=True)
class Citation:
    """回答中引用的来源片段。"""

    chunk_id: str
    file_name: str
    page_number: int | None
    text: str


@dataclass(frozen=True, slots=True)
class QueryResponse:
    """返回给 API 和界面的回答、拒答状态、引用和检索候选。"""

    answer: str
    rejected: bool
    citations: tuple[Citation, ...]
    results: tuple[RetrievalResult, ...] = ()


class UnsupportedDocumentError(ValueError):
    """文件类型超出 V0.1 支持范围时抛出的异常。"""


class EmbeddingIndexCompatibilityError(ValueError):
    """当前 Embedding 配置与已有索引身份不兼容时抛出的异常。"""


class LegacyEmbeddingIndexError(EmbeddingIndexCompatibilityError):
    """旧索引缺少模型身份，需要从保存的文本重新生成向量。"""


class EmbeddingModelMismatchError(EmbeddingIndexCompatibilityError):
    """已有索引记录的模型与当前配置不一致，不能直接混用。"""
