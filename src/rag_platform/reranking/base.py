from collections.abc import Sequence
from typing import Protocol


class Reranker(Protocol):
    """Scores (query, passage) pairs jointly; higher means more relevant.

    Scores are only comparable within one call.
    """

    @property
    def model_name(self) -> str: ...

    async def warm_up(self) -> None:
        """Load model weights now, so the first real request isn't slow."""

    async def score(self, query: str, passages: Sequence[str]) -> list[float]: ...
