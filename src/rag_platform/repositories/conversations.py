import uuid
from collections.abc import Sequence

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from rag_platform.db.models import Conversation, Message


class ConversationRepository:
    """Conversations are private to their owner; every lookup includes the owner id."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def add(self, conversation: Conversation) -> Conversation:
        self._session.add(conversation)
        await self._session.flush()
        return conversation

    async def get_owned(
        self, owner_id: uuid.UUID, conversation_id: uuid.UUID
    ) -> Conversation | None:
        result = await self._session.execute(
            select(Conversation).where(
                Conversation.id == conversation_id, Conversation.owner_id == owner_id
            )
        )
        return result.scalar_one_or_none()

    async def list_owned(
        self, owner_id: uuid.UUID, *, limit: int, offset: int
    ) -> tuple[Sequence[Conversation], int]:
        rows = await self._session.execute(
            select(Conversation)
            .where(Conversation.owner_id == owner_id)
            .order_by(Conversation.updated_at.desc(), Conversation.id)
            .limit(limit)
            .offset(offset)
        )
        total = await self._session.scalar(
            select(func.count()).select_from(Conversation).where(Conversation.owner_id == owner_id)
        )
        return rows.scalars().all(), total or 0

    async def delete(self, conversation: Conversation) -> None:
        await self._session.delete(conversation)
        await self._session.flush()

    async def messages(self, conversation_id: uuid.UUID) -> Sequence[Message]:
        rows = await self._session.execute(
            select(Message)
            .where(Message.conversation_id == conversation_id)
            .order_by(Message.created_at, Message.id)
        )
        return rows.scalars().all()

    async def recent_messages(self, conversation_id: uuid.UUID, limit: int) -> list[Message]:
        if limit <= 0:
            return []
        rows = await self._session.execute(
            select(Message)
            .where(Message.conversation_id == conversation_id)
            .order_by(Message.created_at.desc(), Message.id.desc())
            .limit(limit)
        )
        return list(reversed(rows.scalars().all()))

    async def add_messages(self, *messages: Message) -> None:
        self._session.add_all(messages)
        await self._session.flush()
