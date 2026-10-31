"""Test doubles. Kept out of the application package so production code can't pick them up."""

import hashlib
import math
import re
from collections.abc import Sequence

from rag_platform.embeddings.base import Vector

_TOKEN = re.compile(r"\w+")


class HashingEmbedder:
    """Deterministic bag-of-words embedder: texts sharing words get similar vectors.

    Fast and dependency-free, so tests exercise indexing without downloading a model.
    """

    def __init__(self, dimension: int = 64) -> None:
        self._dimension = dimension
        self.passage_calls = 0

    @property
    def model_name(self) -> str:
        return f"hashing-{self._dimension}"

    @property
    def dimension(self) -> int:
        return self._dimension

    async def warm_up(self) -> None:
        return None

    async def embed_passages(self, texts: Sequence[str]) -> list[Vector]:
        self.passage_calls += 1
        return [self._embed(t) for t in texts]

    async def embed_query(self, text: str) -> Vector:
        return self._embed(text)

    def _embed(self, text: str) -> Vector:
        vector = [0.0] * self._dimension
        for token in _TOKEN.findall(text.lower()):
            bucket = int.from_bytes(hashlib.blake2b(token.encode(), digest_size=4).digest(), "big")
            vector[bucket % self._dimension] += 1.0
        norm = math.sqrt(sum(x * x for x in vector)) or 1.0
        return [x / norm for x in vector]
