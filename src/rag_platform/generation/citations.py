"""Turning the model's ``[n]`` markers into verified citations.

A citation is kept only if ``n`` is one of the sources actually given to the model, so the model
cannot cite a document it never saw. Each citation carries the sentence of the source that best
supports the answer text around the marker, as a snippet the user can check.
"""

import re
from collections.abc import Sequence
from dataclasses import dataclass

from rag_platform.generation.prompts import NO_ANSWER
from rag_platform.retrieval.tokenize import tokenize

_MARKER = re.compile(r"\[(\d{1,3})\]")
_SENTENCE = re.compile(r"(?<=[.!?])\s+")
_MAX_SNIPPET_CHARS = 400
_CURLY_APOSTROPHE = chr(0x2019)  # models often write "don't" with a typographic apostrophe


@dataclass(frozen=True, slots=True)
class ParsedAnswer:
    text: str
    cited: list[int]  # valid source numbers, in order of first use
    invalid: list[int]  # numbers the model cited that were never provided
    is_refusal: bool


def parse_answer(text: str, valid_numbers: set[int]) -> ParsedAnswer:
    cleaned = text.strip()
    if _is_refusal(cleaned):
        return ParsedAnswer(NO_ANSWER, [], [], True)
    cited: list[int] = []
    invalid: list[int] = []
    for match in _MARKER.finditer(cleaned):
        number = int(match.group(1))
        target = cited if number in valid_numbers else invalid
        if number not in target:
            target.append(number)
    # Remove markers pointing at sources that don't exist rather than show broken references.
    if invalid:
        cleaned = _MARKER.sub(
            lambda m: m.group(0) if int(m.group(1)) in valid_numbers else "", cleaned
        )
        cleaned = re.sub(r"[ \t]+([.,;:!?])", r"\1", cleaned)
        cleaned = re.sub(r"[ \t]{2,}", " ", cleaned).strip()
    return ParsedAnswer(cleaned, cited, invalid, False)


def _is_refusal(text: str) -> bool:
    normalised = " ".join(text.lower().replace(_CURLY_APOSTROPHE, "'").split()).rstrip(".")
    return normalised == NO_ANSWER.lower().rstrip(".")


def claims_for(answer: str, number: int) -> str:
    """The answer sentences that cite source ``number``."""
    sentences = [s for s in _SENTENCE.split(answer) if f"[{number}]" in s]
    return " ".join(_MARKER.sub("", s) for s in sentences)


def best_snippet(source_text: str, claim: str) -> str:
    """The source sentence sharing the most terms with the claim (first sentence on a tie)."""
    sentences = [s.strip() for s in _SENTENCE.split(source_text) if s.strip()] or [source_text]
    claim_terms = set(tokenize(claim))
    best = max(sentences, key=lambda s: len(claim_terms & set(tokenize(s))))
    if len(best) > _MAX_SNIPPET_CHARS:
        best = best[: _MAX_SNIPPET_CHARS - 1].rstrip() + "…"
    return best


def select_snippets(answer: str, cited: Sequence[int], texts: dict[int, str]) -> dict[int, str]:
    return {n: best_snippet(texts[n], claims_for(answer, n)) for n in cited}
