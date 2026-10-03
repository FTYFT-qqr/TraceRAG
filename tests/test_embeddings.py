"""验证兼容 Embedding API 的输入校验、响应排序和维度处理。"""

from types import SimpleNamespace

import numpy as np
import pytest

from app.embeddings import EmbeddingClient


class FakeEmbeddingEndpoint:
    def __init__(self, vectors: list[list[float]]) -> None:
        """初始化测试替身及其记录状态。"""
        self.vectors = vectors
        self.request: dict[str, object] | None = None

    def create(self, *, model: str, input: list[str]) -> SimpleNamespace:
        """根据测试输入生成预设的模型响应。"""
        self.request = {"model": model, "input": input}
        data = [
            SimpleNamespace(index=index, embedding=vector)
            for index, vector in reversed(list(enumerate(self.vectors)))
        ]
        return SimpleNamespace(data=data)


def test_embedding_client_sorts_vectors_into_input_order() -> None:
    """模拟服务端乱序响应，确认向量仍和原文本一一对应。"""
    endpoint = FakeEmbeddingEndpoint([[1.0, 2.0], [3.0, 4.0]])
    client = EmbeddingClient(
        api_key="test-key",
        base_url="https://example.test/v1",
        model="test-embedding",
        client=SimpleNamespace(embeddings=endpoint),
    )

    vectors = client.embed_texts(["first", "second"])

    np.testing.assert_array_equal(vectors, np.array([[1, 2], [3, 4]], dtype=np.float32))
    assert endpoint.request == {"model": "test-embedding", "input": ["first", "second"]}


def test_embedding_client_requires_credentials_only_for_live_client() -> None:
    """验证仅真实远程调用要求配置 API 凭据。"""
    with pytest.raises(ValueError, match="TRACERAG_API_KEY"):
        EmbeddingClient(api_key=None, base_url=None, model="model")


def test_embedding_client_validates_inputs_and_api_response() -> None:
    """验证远程向量客户端校验输入并检查服务响应。"""
    endpoint = FakeEmbeddingEndpoint([[1.0, 2.0]])
    client = EmbeddingClient(
        api_key="test-key",
        base_url=None,
        model="test-embedding",
        client=SimpleNamespace(embeddings=endpoint),
    )
    with pytest.raises(ValueError, match="cannot be empty"):
        client.embed_texts([" "])
    with pytest.raises(ValueError, match="unexpected number"):
        client.embed_texts(["one", "two"])
    assert client.embed_texts([]).shape == (0, 0)
