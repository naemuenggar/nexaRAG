"""Evaluation datasets: a corpus of documents (pages of text) plus questions with expected
answers and the (file, page) locations that contain the evidence."""

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

DATASETS_DIR = Path(__file__).resolve().parents[3] / "evaluation" / "datasets"


class CorpusDocument(BaseModel):
    filename: str
    pages: list[str] = Field(min_length=1)


class Corpus(BaseModel):
    description: str = ""
    documents: list[CorpusDocument] = Field(min_length=1)


class SourceRef(BaseModel):
    filename: str
    page: int = Field(ge=1)


class Question(BaseModel):
    id: str
    question: str
    expected_answer: str | None
    sources: list[SourceRef]
    style: Literal["lexical", "paraphrase", "unanswerable"]

    @property
    def answerable(self) -> bool:
        return bool(self.sources)


class Dataset(BaseModel):
    name: str
    corpus: Corpus
    questions: list[Question]

    @property
    def answerable(self) -> list[Question]:
        return [q for q in self.questions if q.answerable]

    @property
    def unanswerable(self) -> list[Question]:
        return [q for q in self.questions if not q.answerable]


def load_dataset(path_or_name: str | Path) -> Dataset:
    """Load ``<dir>/corpus.json`` and ``<dir>/questions.jsonl``; a bare name means a bundled set."""
    directory = Path(path_or_name)
    if not directory.exists():
        directory = DATASETS_DIR / str(path_or_name)
    corpus = Corpus.model_validate_json((directory / "corpus.json").read_text(encoding="utf-8"))
    questions = [
        Question.model_validate(json.loads(line))
        for line in (directory / "questions.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    _validate(corpus, questions)
    return Dataset(name=directory.name, corpus=corpus, questions=questions)


def _validate(corpus: Corpus, questions: list[Question]) -> None:
    pages = {d.filename: len(d.pages) for d in corpus.documents}
    ids = [q.id for q in questions]
    if len(ids) != len(set(ids)):
        raise ValueError("question ids must be unique")
    for question in questions:
        for source in question.sources:
            if source.page > pages.get(source.filename, 0):
                raise ValueError(f"{question.id}: {source.filename} p.{source.page} not in corpus")
        if question.answerable != (question.style != "unanswerable"):
            raise ValueError(f"{question.id}: unanswerable questions must have no sources")
