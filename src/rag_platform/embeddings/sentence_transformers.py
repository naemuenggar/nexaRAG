"""Hugging Face Sentence Transformers embedder (local inference, CPU by default)."""

import asyncio
import threading
from collections.abc import Sequence
from typing import TYPE_CHECKING

from rag_platform.core.config import Settings
from rag_platform.embeddings.base import Vector

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer


class SentenceTransformerEmbedder:
    def __init__(
        self,
        model_name: str,
        *,
        device: str = "cpu",
        batch_size: int = 32,
        query_prefix: str = "",
    ) -> None:
        self._model_name = model_name
        self._device = device
        self._batch_size = batch_size
        self._query_prefix = query_prefix
        self._model: SentenceTransformer | None = None
        # Loading can download weights; make sure two first callers don't both do it.
        self._load_lock = threading.Lock()

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def dimension(self) -> int:
        dimension = self._load().get_sentence_embedding_dimension()
        if dimension is None:
            raise RuntimeError(f"{self._model_name} does not report an embedding dimension")
        return dimension

    async def warm_up(self) -> None:
        await asyncio.to_thread(self._load)

    async def embed_passages(self, texts: Sequence[str]) -> list[Vector]:
        if not texts:
            return []
        return await asyncio.to_thread(self._encode, list(texts))

    async def embed_query(self, text: str) -> Vector:
        vectors = await asyncio.to_thread(self._encode, [self._query_prefix + text])
        return vectors[0]

    def _load(self) -> "SentenceTransformer":
        if self._model is None:
            with self._load_lock:
                if self._model is None:
                    # Imported lazily: torch takes seconds to import and the API process
                    # doesn't need it until query embedding is used.
                    from sentence_transformers import SentenceTransformer

                    self._model = SentenceTransformer(self._model_name, device=self._device)
        return self._model

    def _encode(self, texts: list[str]) -> list[Vector]:
        embeddings = self._load().encode(
            texts,
            batch_size=self._batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        vectors: list[Vector] = embeddings.tolist()
        return vectors


def create_embedder(settings: Settings) -> SentenceTransformerEmbedder:
    return SentenceTransformerEmbedder(
        settings.embedding_model,
        device=settings.embedding_device,
        batch_size=settings.embedding_batch_size,
        query_prefix=settings.embedding_query_prefix,
    )
