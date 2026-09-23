"""OpenAI-compatible embedding client with input and response validation."""

from __future__ import annotations

from typing import Any

import numpy as np
from openai import OpenAI


class EmbeddingClient:
    """Create embedding vectors through an OpenAI-compatible API."""

    def __init__(
        self,
        *,
        api_key: str | None,
        base_url: str | None,
        model: str,
        client: Any | None = None,
    ) -> None:
        if client is None:
            if not api_key:
                raise ValueError("Set TRACERAG_API_KEY or OPENAI_API_KEY to use embeddings.")
            client = OpenAI(api_key=api_key, base_url=base_url)
        self._client = client
        self.model = model

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        """Embed texts in request order as a finite float32 matrix."""

        if not texts:
            return np.empty((0, 0), dtype=np.float32)
        if any(not text.strip() for text in texts):
            raise ValueError("Embedding input text cannot be empty.")

        response = self._client.embeddings.create(model=self.model, input=texts)
        items = sorted(response.data, key=lambda item: item.index)
        if len(items) != len(texts):
            raise ValueError("Embedding API returned an unexpected number of vectors.")
        vectors = np.asarray([item.embedding for item in items], dtype=np.float32)
        if vectors.ndim != 2 or vectors.shape[1] == 0 or not np.isfinite(vectors).all():
            raise ValueError("Embedding API returned invalid vectors.")
        return vectors
