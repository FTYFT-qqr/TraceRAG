"""读取并校验 TraceRAG 服务的环境配置。"""

from __future__ import annotations

import os
import hashlib
import math
from dataclasses import dataclass, field
from pathlib import Path
from functools import lru_cache
from typing import Mapping
from urllib.parse import urlsplit, urlunsplit

from dotenv import load_dotenv

from app import __version__
from app.models import EmbeddingIdentity


_ENV_PREFIX = "TRACERAG_"
_DEFAULT_EMBEDDING_ENDPOINT = "https://api.openai.com/v1"
_MODEL_FINGERPRINT_CHUNK_SIZE = 4 * 1024 * 1024
_IGNORED_MODEL_PATHS = {".git", ".cache", "__pycache__"}


def _normalize_embedding_endpoint(base_url: str | None) -> str:
    """保留远程服务地址，去掉 URL 中的用户名、密码、查询参数和片段。"""

    parsed = urlsplit((base_url or _DEFAULT_EMBEDDING_ENDPOINT).strip())
    if not parsed.scheme or not parsed.hostname:
        raise ValueError("TRACERAG_BASE_URL must be an absolute URL for remote embeddings.")
    scheme = parsed.scheme.lower()
    hostname = parsed.hostname.lower()
    if ":" in hostname and not hostname.startswith("["):
        hostname = f"[{hostname}]"
    try:
        port = parsed.port
    except ValueError as exc:
        raise ValueError("TRACERAG_BASE_URL contains an invalid port.") from exc
    if port and not ((scheme == "https" and port == 443) or (scheme == "http" and port == 80)):
        hostname = f"{hostname}:{port}"
    # 路径用于区分同主机上的 API 服务；凭据和 query 不进入索引身份。
    path = parsed.path.rstrip("/")
    return urlunsplit((scheme, hostname, path, "", ""))


def _fingerprint_local_model(model_path: Path) -> str:
    """对本地模型目录中的文件名和完整内容计算 SHA-256 指纹。"""

    if not model_path.is_dir():
        raise ValueError(f"Local embedding model directory does not exist: {model_path}.")
    root = model_path.resolve()
    model_files = []
    for path in root.rglob("*"):
        relative_path = path.relative_to(root)
        if any(part in _IGNORED_MODEL_PATHS for part in relative_path.parts):
            continue
        if path.is_file() and not path.name.endswith(".lock"):
            model_files.append((relative_path.as_posix(), path))
    if not model_files:
        raise ValueError(f"No model files were found in local embedding directory {root}.")

    digest = hashlib.sha256()
    for relative_name, path in sorted(model_files):
        digest.update(relative_name.encode("utf-8"))
        digest.update(b"\0")
        # 分块读取可为数 GB 的权重文件计算摘要，而不把整个模型载入内存。
        with path.open("rb") as model_file:
            while block := model_file.read(_MODEL_FINGERPRINT_CHUNK_SIZE):
                digest.update(block)
        digest.update(b"\0")
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class Settings:
    """从环境变量读取的运行时配置。"""

    app_name: str = "TraceRAG"
    environment: str = "development"
    version: str = __version__
    host: str = "127.0.0.1"
    port: int = 8000
    log_level: str = "INFO"
    api_key: str | None = None
    base_url: str | None = None
    api_proxy_url: str | None = None
    embedding_provider: str = "openai"
    embedding_model: str = "text-embedding-3-small"
    local_embedding_path: str | None = None
    embedding_device: str | None = None
    chat_model: str = "gpt-4o-mini"
    index_dir: str = "indexes/default"
    reject_threshold: float = 0.25
    retrieval_strategy: str = "vector"
    bm25_min_score: float = 0.0
    rrf_k: int = 60
    _embedding_identity: EmbeddingIdentity = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        """实例化配置时固定一次模型身份，避免请求期间反复散列大型权重。"""

        provider = self.embedding_provider.strip().lower()
        strategy = self.retrieval_strategy.strip().lower()
        if strategy not in {"vector", "bm25", "hybrid"}:
            raise ValueError("TRACERAG_RETRIEVAL_STRATEGY must be vector, bm25 or hybrid.")
        if not math.isfinite(self.bm25_min_score) or self.bm25_min_score < 0:
            raise ValueError("TRACERAG_BM25_MIN_SCORE must be finite and nonnegative.")
        if type(self.rrf_k) is not int or self.rrf_k <= 0:
            raise ValueError("TRACERAG_RRF_K must be a positive integer.")
        object.__setattr__(self, "retrieval_strategy", strategy)
        if provider not in {"openai", "local"}:
            raise ValueError("TRACERAG_EMBEDDING_PROVIDER must be 'openai' or 'local'.")
        object.__setattr__(self, "embedding_provider", provider)
        if provider == "local":
            if not self.local_embedding_path:
                raise ValueError(
                    "Set TRACERAG_LOCAL_EMBEDDING_PATH to identify the local embedding model."
                )
            model_path = Path(self.local_embedding_path).expanduser().resolve()
            identity = EmbeddingIdentity(
                provider=provider,
                model=os.path.normcase(str(model_path)),
                model_fingerprint=_fingerprint_local_model(model_path),
            )
        else:
            identity = EmbeddingIdentity(
                provider=provider,
                model=self.embedding_model.strip(),
                endpoint=_normalize_embedding_endpoint(self.base_url),
            )
        object.__setattr__(self, "_embedding_identity", identity)

    @property
    def embedding_identity(self) -> EmbeddingIdentity:
        """返回向量空间身份；同维度不代表不同模型可以共用索引。"""
        return self._embedding_identity

    @classmethod
    def from_environment(cls, environment: Mapping[str, str] | None = None) -> "Settings":
        """从给定环境映射构造配置；显式传入时不读取 .env 文件。"""

        source = os.environ if environment is None else environment
        defaults = cls()
        port_value = source.get(f"{_ENV_PREFIX}PORT", str(defaults.port))

        try:
            port = int(port_value)
        except ValueError as exc:
            raise ValueError(f"{_ENV_PREFIX}PORT must be an integer.") from exc

        if not 1 <= port <= 65535:
            raise ValueError(f"{_ENV_PREFIX}PORT must be between 1 and 65535.")
        threshold_value = source.get(
            f"{_ENV_PREFIX}REJECT_THRESHOLD", str(defaults.reject_threshold)
        )
        try:
            reject_threshold = float(threshold_value)
        except ValueError as exc:
            raise ValueError(
                f"{_ENV_PREFIX}REJECT_THRESHOLD must be a number between -1 and 1."
            ) from exc
        if not -1.0 <= reject_threshold <= 1.0:
            raise ValueError(
                f"{_ENV_PREFIX}REJECT_THRESHOLD must be between -1 and 1."
            )

        embedding_provider = source.get(
            f"{_ENV_PREFIX}EMBEDDING_PROVIDER", defaults.embedding_provider
        ).strip().lower()
        if embedding_provider not in {"openai", "local"}:
            raise ValueError(
                f"{_ENV_PREFIX}EMBEDDING_PROVIDER must be 'openai' or 'local'."
            )

        try:
            bm25_min_score = float(source.get(f"{_ENV_PREFIX}BM25_MIN_SCORE", str(defaults.bm25_min_score)))
            rrf_k = int(source.get(f"{_ENV_PREFIX}RRF_K", str(defaults.rrf_k)))
        except ValueError as exc:
            raise ValueError("TRACERAG_BM25_MIN_SCORE and TRACERAG_RRF_K must be numeric.") from exc

        return cls(
            app_name=source.get(f"{_ENV_PREFIX}APP_NAME", defaults.app_name),
            environment=source.get(f"{_ENV_PREFIX}ENVIRONMENT", defaults.environment),
            version=source.get(f"{_ENV_PREFIX}VERSION", defaults.version),
            host=source.get(f"{_ENV_PREFIX}HOST", defaults.host),
            port=port,
            api_key=(source.get(f"{_ENV_PREFIX}API_KEY") or source.get("OPENAI_API_KEY") or None),
            base_url=(source.get(f"{_ENV_PREFIX}BASE_URL") or None),
            api_proxy_url=(source.get(f"{_ENV_PREFIX}API_PROXY_URL") or None),
            embedding_provider=embedding_provider,
            embedding_model=source.get(
                f"{_ENV_PREFIX}EMBEDDING_MODEL", defaults.embedding_model
            ),
            local_embedding_path=(
                source.get(f"{_ENV_PREFIX}LOCAL_EMBEDDING_PATH") or None
            ),
            embedding_device=(source.get(f"{_ENV_PREFIX}EMBEDDING_DEVICE") or None),
            chat_model=source.get(f"{_ENV_PREFIX}CHAT_MODEL", defaults.chat_model),
            index_dir=source.get(f"{_ENV_PREFIX}INDEX_DIR", defaults.index_dir),
            reject_threshold=reject_threshold,
            retrieval_strategy=source.get(f"{_ENV_PREFIX}RETRIEVAL_STRATEGY", defaults.retrieval_strategy),
            bm25_min_score=bm25_min_score,
            rrf_k=rrf_k,
            log_level=source.get(f"{_ENV_PREFIX}LOG_LEVEL", defaults.log_level).upper(),
        )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """读取项目根目录 .env 并缓存应用启动配置。"""
    # override=False 保留进程显式注入的环境变量优先级。
    load_dotenv(Path.cwd() / ".env", override=False)

    return Settings.from_environment()
