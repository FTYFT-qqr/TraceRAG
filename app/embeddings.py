"""提供远程兼容 API 和本地 Sentence Transformers 两种 Embedding 客户端。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from openai import DefaultHttpxClient, OpenAI


class EmbeddingClient:
    """通过 OpenAI 兼容接口生成向量，并校验返回顺序及维度。"""

    def __init__(
        self,
        *,
        api_key: str | None,
        base_url: str | None,
        proxy_url: str | None = None,
        model: str,
        client: Any | None = None,
    ) -> None:
        """初始化远程 Embedding 客户端及其模型配置。"""
        if client is None:
            if not api_key:
                raise ValueError("Set TRACERAG_API_KEY or OPENAI_API_KEY to use embeddings.")
            client = OpenAI(
                api_key=api_key,
                base_url=base_url,
                http_client=(
                    DefaultHttpxClient(proxy=proxy_url, trust_env=False)
                    if proxy_url
                    else None
                ),
            )
        self._client = client
        self.model = model

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        """按输入顺序返回有限值组成的 float32 二维矩阵。"""

        if not texts:
            return np.empty((0, 0), dtype=np.float32)
        if any(not text.strip() for text in texts):
            raise ValueError("Embedding input text cannot be empty.")

        response = self._client.embeddings.create(model=self.model, input=texts)
        # 远程响应可能乱序；按服务端 index 排回请求顺序以对齐 Chunk。
        items = sorted(response.data, key=lambda item: item.index)
        if len(items) != len(texts):
            raise ValueError("Embedding API returned an unexpected number of vectors.")
        vectors = np.asarray([item.embedding for item in items], dtype=np.float32)
        if vectors.ndim != 2 or vectors.shape[1] == 0 or not np.isfinite(vectors).all():
            raise ValueError("Embedding API returned invalid vectors.")
        return vectors


class SentenceTransformerEmbeddingClient:
    """使用本地 Sentence Transformers 模型生成归一化向量。"""

    def __init__(
        self,
        *,
        model_path: str | Path | None,
        device: str | None = None,
        batch_size: int = 16,
    ) -> None:
        """加载本地 Sentence Transformers 模型并固定批量大小。"""
        if not model_path:
            raise ValueError(
                "Set TRACERAG_LOCAL_EMBEDDING_PATH to the local model directory."
            )
        path = Path(model_path).expanduser()
        if not path.is_dir() or not (path / "modules.json").is_file():
            raise FileNotFoundError(
                f"Sentence Transformers model files were not found in {path}."
            )
        if batch_size <= 0:
            raise ValueError("Embedding batch size must be a positive integer.")

        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "Install local embedding support with "
                "`python -m pip install -e \".[local-embeddings]\"`."
            ) from exc

        self.model_path = path
        self.batch_size = batch_size
        self._model = SentenceTransformer(str(path), device=device)

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        """保留输入顺序并验证本地模型输出形状和值域。"""

        if not texts:
            return np.empty((0, 0), dtype=np.float32)
        if any(not text.strip() for text in texts):
            raise ValueError("Embedding input text cannot be empty.")

        vectors = np.asarray(
            self._model.encode(
                texts,
                batch_size=self.batch_size,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            ),
            dtype=np.float32,
        )
        if (
            vectors.ndim != 2
            or vectors.shape[0] != len(texts)
            or vectors.shape[1] == 0
            or not np.isfinite(vectors).all()
        ):
            raise ValueError("Local embedding model returned invalid vectors.")
        return vectors
