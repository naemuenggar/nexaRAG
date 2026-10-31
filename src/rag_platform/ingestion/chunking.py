"""Split page text into overlapping, sentence-aligned chunks that remember their pages.

Text is first cut into sentences (sentences longer than a chunk are cut into word windows), then
sentences are packed greedily up to ``size_words``. Each new chunk starts with the last sentences
of the previous one, up to ``overlap_words``, so a fact that straddles a boundary is still found.
"""

import re
from collections.abc import Iterator, Sequence
from dataclasses import dataclass

from rag_platform.ingestion.pdf import ParsedPage

_PARAGRAPH_BREAK = re.compile(r"\n\s*\n")
# End of sentence followed by something that looks like the start of a new one.
_SENTENCE_BREAK = re.compile(r"(?<=[.!?])\s+(?=[\"'(\[]?[A-Z0-9])")
_WHITESPACE = re.compile(r"\s+")


@dataclass(frozen=True, slots=True)
class ChunkingConfig:
    size_words: int
    overlap_words: int
    cross_pages: bool = False

    def __post_init__(self) -> None:
        if self.size_words < 1:
            raise ValueError("size_words must be positive")
        if not 0 <= self.overlap_words < self.size_words:
            raise ValueError("overlap_words must be in [0, size_words)")


@dataclass(frozen=True, slots=True)
class Chunk:
    ordinal: int  # position within the document, from 0
    page_start: int
    page_end: int
    text: str
    word_count: int


@dataclass(frozen=True, slots=True)
class _Unit:
    page: int
    text: str
    words: int


def count_words(text: str) -> int:
    return len(text.split())


def _sentences(text: str) -> Iterator[str]:
    for paragraph in _PARAGRAPH_BREAK.split(text):
        flat = _WHITESPACE.sub(" ", paragraph).strip()
        if flat:
            yield from (s for s in _SENTENCE_BREAK.split(flat) if s)


def _units(page: ParsedPage, max_words: int) -> Iterator[_Unit]:
    for sentence in _sentences(page.text):
        words = sentence.split()
        for start in range(0, len(words), max_words):
            window = words[start : start + max_words]
            yield _Unit(page.number, " ".join(window), len(window))


def chunk_pages(pages: Sequence[ParsedPage], config: ChunkingConfig) -> list[Chunk]:
    chunks: list[Chunk] = []
    current: list[_Unit] = []
    current_words = 0
    fresh_units = 0  # units in `current` that are not carried over from the previous chunk

    def emit() -> None:
        nonlocal current, current_words, fresh_units
        if fresh_units == 0:
            return  # only overlap left: it is already part of the previous chunk
        chunks.append(
            Chunk(
                ordinal=len(chunks),
                page_start=current[0].page,
                page_end=current[-1].page,
                text=" ".join(u.text for u in current),
                word_count=current_words,
            )
        )
        carried: list[_Unit] = []
        carried_words = 0
        for unit in reversed(current):
            if carried_words + unit.words > config.overlap_words:
                break
            carried.insert(0, unit)
            carried_words += unit.words
        current, current_words, fresh_units = carried, carried_words, 0

    def reset() -> None:
        nonlocal current, current_words, fresh_units
        current, current_words, fresh_units = [], 0, 0

    for page in pages:
        for unit in _units(page, config.size_words):
            if current_words + unit.words > config.size_words:
                if fresh_units:
                    emit()
                # Drop carried overlap that can't fit next to this unit.
                while current and current_words + unit.words > config.size_words:
                    current_words -= current.pop(0).words
            current.append(unit)
            current_words += unit.words
            fresh_units += 1
        if not config.cross_pages:
            emit()
            reset()
    emit()
    return chunks
