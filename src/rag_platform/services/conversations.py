"""Conversations: multi-turn question answering with stored history.

Each new question is answered by ``AskService`` with the last few turns as context. Answers are
stored with the citations and usage the user saw. A failed answer (LLM down, bad request)
stores nothing, so history never contains half a turn.
"""

import re
import uuid
from collections.abc import Sequence
from dataclasses import asdict
from datetime import UTC, datetime, timedelta

from sqlalchemy.ext.asyncio import AsyncSession

from rag_platform.core.config import Settings
from rag_platform.core.errors import NotFoundError
from rag_platform.db.models import Conversation, Message, MessageRole
from rag_platform.generation.llm import ChatMessage
from rag_platform.repositories.conversations import ConversationRepository
from rag_platform.services.ask import AskResult, AskService
from rag_platform.services.search import DocumentFilter

DEFAULT_TITLE = "New conversation"
_TITLE_LENGTH = 80
_CITATION_MARKER = re.compile(r"\s*\[\d{1,3}\]")


def title_from(question: str) -> str:
    flat = " ".join(question.split())
    return flat if len(flat) <= _TITLE_LENGTH else flat[: _TITLE_LENGTH - 1].rstrip() + "…"


def history_for_prompt(messages: Sequence[Message]) -> list[ChatMessage]:
    """Earlier turns as plain text. Old ``[n]`` markers are stripped: they referred to that
    turn's sources, and would point at the wrong passages in the new prompt."""
    turns: list[ChatMessage] = []
    for message in messages:
        content = message.content
        if message.role is MessageRole.ASSISTANT:
            content = _CITATION_MARKER.sub("", content)
        turns.append(ChatMessage(message.role.value, content))
    return turns


class ConversationService:
    def __init__(self, session: AsyncSession, ask: AskService, settings: Settings) -> None:
        self._session = session
        self._repo = ConversationRepository(session)
        self._ask = ask
        self._settings = settings

    async def create(self, owner_id: uuid.UUID, title: str | None = None) -> Conversation:
        conversation = await self._repo.add(
            Conversation(owner_id=owner_id, title=title_from(title) if title else DEFAULT_TITLE)
        )
        await self._session.commit()
        return conversation

    async def list(
        self, owner_id: uuid.UUID, *, limit: int, offset: int
    ) -> tuple[Sequence[Conversation], int]:
        return await self._repo.list_owned(owner_id, limit=limit, offset=offset)

    async def get(
        self, owner_id: uuid.UUID, conversation_id: uuid.UUID
    ) -> tuple[Conversation, Sequence[Message]]:
        conversation = await self._get_owned(owner_id, conversation_id)
        return conversation, await self._repo.messages(conversation.id)

    async def delete(self, owner_id: uuid.UUID, conversation_id: uuid.UUID) -> None:
        await self._repo.delete(await self._get_owned(owner_id, conversation_id))
        await self._session.commit()

    async def send(
        self,
        owner_id: uuid.UUID,
        conversation_id: uuid.UUID,
        question: str,
        *,
        document_filter: DocumentFilter | None = None,
    ) -> tuple[Message, Message, AskResult]:
        conversation = await self._get_owned(owner_id, conversation_id)
        recent = await self._repo.recent_messages(
            conversation.id, self._settings.conversation_history_messages
        )
        previous_question = next(
            (m.content for m in reversed(recent) if m.role is MessageRole.USER), None
        )
        result = await self._ask.ask(
            owner_id,
            question,
            document_filter=document_filter,
            history=history_for_prompt(recent),
            retrieval_query=f"{previous_question}\n{question}" if previous_question else None,
        )

        # Explicit timestamps: both rows are written in one transaction, where now() is constant,
        # and history order must not depend on random ids.
        now = datetime.now(UTC)
        user_message = Message(
            conversation_id=conversation.id, role=MessageRole.USER, content=question, created_at=now
        )
        assistant_message = Message(
            conversation_id=conversation.id,
            role=MessageRole.ASSISTANT,
            content=result.answer,
            answered=result.answered,
            refusal_reason=result.refusal_reason.value if result.refusal_reason else None,
            citations=[
                {**asdict(c), "chunk_id": str(c.chunk_id), "document_id": str(c.document_id)}
                for c in result.citations
            ],
            usage=asdict(result.usage) if result.usage else None,
            created_at=now + timedelta(microseconds=1),
        )
        await self._repo.add_messages(user_message, assistant_message)
        if conversation.title == DEFAULT_TITLE:
            conversation.title = title_from(question)
        conversation.updated_at = now
        await self._session.commit()
        return user_message, assistant_message, result

    async def _get_owned(self, owner_id: uuid.UUID, conversation_id: uuid.UUID) -> Conversation:
        conversation = await self._repo.get_owned(owner_id, conversation_id)
        if conversation is None:
            raise NotFoundError("Conversation not found")
        return conversation
