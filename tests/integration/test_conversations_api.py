"""Conversations: stored turns, history in the prompt, per-user isolation."""

import pytest
from fastapi.testclient import TestClient

from rag_platform.generation.prompts import NO_ANSWER
from rag_platform.ingestion.worker import IngestionWorker
from tests.fakes import ScriptedLLM
from tests.integration.test_documents_api import drain, signup, upload
from tests.pdf_factory import make_pdf

pytestmark = pytest.mark.integration


@pytest.fixture
async def owner(api: TestClient, worker: IngestionWorker) -> dict[str, str]:
    headers = signup(api)
    upload(
        api,
        headers,
        make_pdf(
            [
                "Revenue in 2024 was 4.2 million dollars.",
                "Revenue in 2023 was 3.1 million dollars.",
            ]
        ),
        "revenue.pdf",
    )
    await drain(worker)
    return headers


def start(api: TestClient, headers: dict[str, str], **body: object) -> str:
    response = api.post("/conversations", headers=headers, json=body)
    assert response.status_code == 201, response.text
    return response.json()["id"]  # type: ignore[no-any-return]


def send(api: TestClient, headers: dict[str, str], conversation_id: str, question: str) -> dict:  # type: ignore[type-arg]
    response = api.post(
        f"/conversations/{conversation_id}/messages", headers=headers, json={"question": question}
    )
    assert response.status_code == 200, response.text
    return response.json()  # type: ignore[no-any-return]


async def test_turns_are_stored_with_citations_and_history_is_used(
    api: TestClient, owner: dict[str, str]
) -> None:
    llm = ScriptedLLM("Revenue in 2024 was 4.2 million dollars [1].", "It was 3.1 million [1].")
    api.app.state.container.llm = llm  # type: ignore[attr-defined]
    conversation_id = start(api, owner)

    first = send(api, owner, conversation_id, "What was revenue in 2024?")
    assert first["answer"]["answered"] is True
    assert first["answer"]["citations"][0]["filename"] == "revenue.pdf"
    second = send(api, owner, conversation_id, "And in 2023?")
    assert second["answer"]["content"] == "It was 3.1 million [1]."

    # The follow-up prompt carries the earlier turn, with old citation markers stripped.
    roles = [m.role for m in llm.prompts[1]]
    assert roles == ["system", "user", "assistant", "user"]
    assert llm.prompts[1][1].content == "What was revenue in 2024?"
    assert llm.prompts[1][2].content == "Revenue in 2024 was 4.2 million dollars."
    assert "3.1 million" in llm.prompts[1][3].content  # retrieval found the 2023 page

    detail = api.get(f"/conversations/{conversation_id}", headers=owner).json()
    assert detail["title"] == "What was revenue in 2024?"
    assert [(m["role"], m["content"]) for m in detail["messages"]] == [
        ("user", "What was revenue in 2024?"),
        ("assistant", "Revenue in 2024 was 4.2 million dollars [1]."),
        ("user", "And in 2023?"),
        ("assistant", "It was 3.1 million [1]."),
    ]
    stored = detail["messages"][1]
    assert stored["citations"][0]["page_start"] == 1
    assert stored["usage"]["model"] == "scripted-llm"


async def test_refusals_are_stored_too(api: TestClient, owner: dict[str, str]) -> None:
    api.app.state.container.llm = ScriptedLLM(NO_ANSWER)  # type: ignore[attr-defined]
    conversation_id = start(api, owner, title="Board questions")
    body = send(api, owner, conversation_id, "Who sits on the board?")
    assert body["answer"]["answered"] is False
    assert body["answer"]["refusal_reason"] == "model_declined"
    detail = api.get(f"/conversations/{conversation_id}", headers=owner).json()
    assert detail["title"] == "Board questions"  # an explicit title is kept
    assert len(detail["messages"]) == 2


async def test_failed_answer_stores_nothing(api: TestClient, owner: dict[str, str]) -> None:
    api.app.state.container.llm = None  # type: ignore[attr-defined]
    conversation_id = start(api, owner)
    response = api.post(
        f"/conversations/{conversation_id}/messages", headers=owner, json={"question": "Revenue?"}
    )
    assert response.status_code == 503
    assert api.get(f"/conversations/{conversation_id}", headers=owner).json()["messages"] == []


async def test_conversations_are_private(api: TestClient, owner: dict[str, str]) -> None:
    api.app.state.container.llm = ScriptedLLM("Revenue was 4.2 million [1].")  # type: ignore[attr-defined]
    conversation_id = start(api, owner)
    send(api, owner, conversation_id, "Revenue in 2024?")
    other = signup(api)

    assert api.get(f"/conversations/{conversation_id}", headers=other).status_code == 404
    post = api.post(
        f"/conversations/{conversation_id}/messages", headers=other, json={"question": "hi"}
    )
    assert post.status_code == 404
    assert api.delete(f"/conversations/{conversation_id}", headers=other).status_code == 404
    assert api.get("/conversations", headers=other).json()["total"] == 0

    listing = api.get("/conversations", headers=owner).json()
    assert [c["id"] for c in listing["items"]] == [conversation_id]
    assert api.delete(f"/conversations/{conversation_id}", headers=owner).status_code == 204
    assert api.get(f"/conversations/{conversation_id}", headers=owner).status_code == 404
