"""Signed JWT access tokens. The subject is the user id; nothing else is trusted from the token."""

import uuid
from datetime import UTC, datetime, timedelta

import jwt

from rag_platform.core.config import Settings
from rag_platform.core.errors import AuthenticationError

_AUDIENCE = "rag-platform-api"


def create_access_token(
    user_id: uuid.UUID, settings: Settings, *, now: datetime | None = None
) -> str:
    issued = now or datetime.now(UTC)
    payload = {
        "sub": str(user_id),
        "aud": _AUDIENCE,
        "iat": issued,
        "exp": issued + timedelta(minutes=settings.access_token_ttl_minutes),
        "jti": uuid.uuid4().hex,
    }
    return jwt.encode(
        payload, settings.jwt_secret.get_secret_value(), algorithm=settings.jwt_algorithm
    )


def decode_access_token(token: str, settings: Settings) -> uuid.UUID:
    try:
        payload = jwt.decode(
            token,
            settings.jwt_secret.get_secret_value(),
            # Pin the algorithm list: never let the token header choose (e.g. "none").
            algorithms=[settings.jwt_algorithm],
            audience=_AUDIENCE,
            options={"require": ["sub", "exp", "iat", "aud"]},
        )
        return uuid.UUID(payload["sub"])
    except (jwt.PyJWTError, ValueError, KeyError) as exc:
        raise AuthenticationError("Invalid or expired token") from exc
