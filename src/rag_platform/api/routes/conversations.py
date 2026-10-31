import uuid
from typing import Annotated

from fastapi import APIRouter, Query, Response, status

from rag_platform.api.deps import ConversationServiceDep, CurrentUserDep
from rag_platform.api.routes.search import to_filter
from rag_platform.api.schemas import (
    ConversationCreate,
    ConversationDetail,
    ConversationList,
    ConversationOut,
    LatencyOut,
    MessageCreate,
    MessageExchange,
    MessageOut,
)

router = APIRouter(prefix="/conversations", tags=["conversations"])

Limit = Annotated[int, Query(ge=1, le=100)]
Offset = Annotated[int, Query(ge=0)]


@router.post("", response_model=ConversationOut, status_code=status.HTTP_201_CREATED)
async def create_conversation(
    body: ConversationCreate, user: CurrentUserDep, service: ConversationServiceDep
) -> ConversationOut:
    return ConversationOut.model_validate(await service.create(user.id, body.title))


@router.get("", response_model=ConversationList, summary="Your conversations, latest first")
async def list_conversations(
    user: CurrentUserDep, service: ConversationServiceDep, limit: Limit = 20, offset: Offset = 0
) -> ConversationList:
    items, total = await service.list(user.id, limit=limit, offset=offset)
    return ConversationList(
        items=[ConversationOut.model_validate(c) for c in items],
        total=total,
        limit=limit,
        offset=offset,
    )


@router.get("/{conversation_id}", response_model=ConversationDetail)
async def get_conversation(
    conversation_id: uuid.UUID, user: CurrentUserDep, service: ConversationServiceDep
) -> ConversationDetail:
    conversation, messages = await service.get(user.id, conversation_id)
    return ConversationDetail(
        **ConversationOut.model_validate(conversation).model_dump(),
        messages=[MessageOut.model_validate(m) for m in messages],
    )


@router.delete("/{conversation_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_conversation(
    conversation_id: uuid.UUID, user: CurrentUserDep, service: ConversationServiceDep
) -> Response:
    await service.delete(user.id, conversation_id)
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.post(
    "/{conversation_id}/messages",
    response_model=MessageExchange,
    summary="Ask a question in this conversation; earlier turns are used as context",
)
async def send_message(
    conversation_id: uuid.UUID,
    body: MessageCreate,
    user: CurrentUserDep,
    service: ConversationServiceDep,
) -> MessageExchange:
    question, answer, result = await service.send(
        user.id, conversation_id, body.question, document_filter=to_filter(body.filters)
    )
    return MessageExchange(
        question=MessageOut.model_validate(question),
        answer=MessageOut.model_validate(answer),
        sources_considered=result.sources_considered,
        suspicious_sources=result.suspicious_sources,
        latency=LatencyOut(
            retrieval_ms=result.retrieval_ms,
            generation_ms=result.generation_ms,
            total_ms=result.total_ms,
        ),
    )
