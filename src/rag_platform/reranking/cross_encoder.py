"""Sentence Transformers cross-encoder reranker (local inference, CPU by default)."""

import asyncio
import threading
from collections.abc import Sequence
from typing import TYPE_CHECKING

from rag_platform.core.config import Settings

if TYPE_CHECKING:
    from sentence_transformers import CrossEncoder


class CrossEncoderReranker:
    def __init__(
        self, model_name: str, *, device: str = "cpu", batch_size: int = 16, max_length: int = 512
    ) -> None:
        self._model_name = model_name
        self._device = device
        self._batch_size = batch_size
        self._max_length = max_length
        self._model: CrossEncoder | None = None
        self._load_lock = threading.Lock()

    @property
    def model_name(self) -> str:
        return self._model_name

    async def warm_up(self) -> None:
        await asyncio.to_thread(self._load)

    async def score(self, query: str, passages: Sequence[str]) -> list[float]:
        if not passages:
            return []
        return await asyncio.to_thread(self._predict, query, list(passages))

    def _load(self) -> "CrossEncoder":
        if self._model is None:
            with self._load_lock:
                if self._model is None:
                    from sentence_transformers import CrossEncoder

                    self._model = CrossEncoder(
                        self._model_name, device=self._device, max_length=self._max_length
                    )
        return self._model

    def _predict(self, query: str, passages: list[str]) -> list[float]:
        scores = self._load().predict(
            [(query, p) for p in passages],
            batch_size=self._batch_size,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        values: list[float] = scores.tolist()
        return values


def create_reranker(settings: Settings) -> CrossEncoderReranker:
    return CrossEncoderReranker(
        settings.reranker_model,
        device=settings.reranker_device,
        batch_size=settings.reranker_batch_size,
    )
