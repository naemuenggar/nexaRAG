import itertools

import pytest

from rag_platform.ingestion.chunking import Chunk, ChunkingConfig, chunk_pages, count_words
from rag_platform.ingestion.pdf import ParsedPage


def sentences(prefix: str, n: int, words_each: int = 10) -> str:
    """n sentences of exactly `words_each` words, each starting with "<PREFIX><index>"."""
    return " ".join(
        " ".join([f"{prefix.upper()}{i}"] + ["word"] * (words_each - 2) + ["end."])
        for i in range(n)
    )


def chunk(pages: list[str], size: int = 50, overlap: int = 0, cross: bool = False) -> list[Chunk]:
    return chunk_pages(
        [ParsedPage(i + 1, t) for i, t in enumerate(pages)], ChunkingConfig(size, overlap, cross)
    )


def test_respects_size_and_keeps_sentences_whole() -> None:
    chunks = chunk([sentences("s", 12)], size=50)
    assert [c.word_count for c in chunks] == [50, 50, 20]
    assert all(c.text.endswith("end.") for c in chunks)
    assert all(c.word_count == count_words(c.text) for c in chunks)


def test_ordinals_are_sequential_across_pages() -> None:
    chunks = chunk([sentences("a", 6), sentences("b", 6)], size=50)
    assert [c.ordinal for c in chunks] == list(range(len(chunks)))


def test_overlap_repeats_trailing_sentences() -> None:
    chunks = chunk([sentences("s", 10)], size=50, overlap=20)
    # Each chunk after the first starts with the last two sentences of the previous one.
    for previous, current in itertools.pairwise(chunks):
        tail = previous.text.split("end.")[-3:-1]
        assert current.text.startswith(tail[0].strip())
    assert all(c.word_count <= 50 for c in chunks)
    all_text = " ".join(c.text for c in chunks)
    assert all(f"S{i} " in all_text for i in range(10))


def test_overlap_terminates_and_never_emits_overlap_only_chunks() -> None:
    chunks = chunk([sentences("s", 3)], size=50, overlap=40)
    assert len(chunks) == 1
    assert chunks[0].word_count == 30


def test_chunks_stay_on_one_page_by_default() -> None:
    chunks = chunk([sentences("a", 2), sentences("b", 2)], size=100)
    assert [(c.page_start, c.page_end) for c in chunks] == [(1, 1), (2, 2)]
    assert "B0" not in chunks[0].text


def test_no_overlap_carried_across_pages_by_default() -> None:
    chunks = chunk([sentences("a", 5), sentences("b", 5)], size=30, overlap=10)
    assert all(c.page_start == c.page_end for c in chunks)
    assert not any(c.text.startswith("A") for c in chunks if c.page_start == 2)


def test_cross_page_chunks_record_their_page_span() -> None:
    chunks = chunk([sentences("a", 2), sentences("b", 2)], size=100, cross=True)
    assert len(chunks) == 1
    assert (chunks[0].page_start, chunks[0].page_end) == (1, 2)


def test_blank_pages_produce_no_chunks_but_keep_numbering() -> None:
    chunks = chunk(["", sentences("x", 1), ""], size=50)
    assert [(c.page_start, c.text.split()[0]) for c in chunks] == [(2, "X0")]


def test_overlong_sentence_is_split_into_windows() -> None:
    long_sentence = " ".join(f"w{i}" for i in range(120)) + "."
    chunks = chunk([long_sentence], size=50)
    assert [c.word_count for c in chunks] == [50, 50, 20]
    assert chunks[0].text.split()[0] == "w0"
    assert chunks[1].text.split()[0] == "w50"


def test_paragraph_and_line_breaks_are_flattened() -> None:
    chunks = chunk(["First line\ncontinues here.\n\nSecond paragraph."], size=50)
    assert chunks[0].text == "First line continues here. Second paragraph."


def test_sentence_split_does_not_break_decimals_or_lowercase_abbreviations() -> None:
    chunks = chunk(["Revenue was 4.2 million, e.g. in Q3. Costs fell."], size=7)
    assert chunks[0].text == "Revenue was 4.2 million, e.g. in Q3."


def test_output_is_deterministic() -> None:
    pages = [sentences("a", 9), sentences("b", 4)]
    assert chunk(pages, size=40, overlap=15) == chunk(pages, size=40, overlap=15)


@pytest.mark.parametrize(("size", "overlap"), [(0, 0), (10, 10), (10, -1)])
def test_invalid_config_is_rejected(size: int, overlap: int) -> None:
    with pytest.raises(ValueError, match="_words"):
        ChunkingConfig(size, overlap)
