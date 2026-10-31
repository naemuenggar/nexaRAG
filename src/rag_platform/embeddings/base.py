from collections.abc import Sequence
from typing import Protocol

Vector = list[float]


class Embedder(Protocol):
    """Turns text into L2-normalised vectors, so cosine similarity equals the dot product.

    Passages and queries are separate calls because many retrieval models encode them
    differently (e.g. an instruction prefix on queries only).
    """

    @property
    def model_name(self) -> str: ...

    @property
    def dimension(self) -> int: ...

    async def embed_passages(self, texts: Sequence[str]) -> list[Vector]: ...

    async def embed_query(self, text: str) -> Vector: ...
