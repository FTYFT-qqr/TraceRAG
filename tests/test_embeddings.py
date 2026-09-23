from types import SimpleNamespace

import numpy as np
import pytest

from app.embeddings import EmbeddingClient


class FakeEmbeddingEndpoint:
    def __init__(self, vectors: list[list[float]]) -> None:
        self.vectors = vectors
        self.request: dict[str, object] | None = None

    def create(self, *, model: str, input: list[str]) -> SimpleNamespace:
        self.request = {"model": model, "input": input}
        data = [
            SimpleNamespace(index=index, embedding=vector)
            for index, vector in reversed(list(enumerate(self.vectors)))
        ]
        return SimpleNamespace(data=data)


def test_embedding_client_sorts_vectors_into_input_order() -> None:
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
    with pytest.raises(ValueError, match="TRACERAG_API_KEY"):
        EmbeddingClient(api_key=None, base_url=None, model="model")


def test_embedding_client_validates_inputs_and_api_response() -> None:
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
