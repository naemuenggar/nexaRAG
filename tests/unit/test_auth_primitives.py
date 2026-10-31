import uuid
from datetime import UTC, datetime, timedelta

import jwt
import pytest

from rag_platform.auth.passwords import hash_password, verify_password
from rag_platform.auth.tokens import create_access_token, decode_access_token
from rag_platform.core.errors import AuthenticationError
from tests.conftest import make_settings


def test_password_round_trip() -> None:
    hashed = hash_password("correct horse battery staple")
    assert hashed.startswith("$argon2id$")
    assert verify_password("correct horse battery staple", hashed)
    assert not verify_password("wrong password", hashed)


def test_verify_without_hash_is_false() -> None:
    assert not verify_password("anything", None)
    assert not verify_password("anything", "not-a-valid-hash")


def test_token_round_trip() -> None:
    settings = make_settings()
    user_id = uuid.uuid4()
    assert decode_access_token(create_access_token(user_id, settings), settings) == user_id


def test_expired_token_is_rejected() -> None:
    settings = make_settings(access_token_ttl_minutes=1)
    token = create_access_token(
        uuid.uuid4(), settings, now=datetime.now(UTC) - timedelta(minutes=5)
    )
    with pytest.raises(AuthenticationError):
        decode_access_token(token, settings)


def test_token_signed_with_other_secret_is_rejected() -> None:
    token = create_access_token(uuid.uuid4(), make_settings(jwt_secret="a" * 40))
    with pytest.raises(AuthenticationError):
        decode_access_token(token, make_settings(jwt_secret="b" * 40))


def test_unsigned_alg_none_token_is_rejected() -> None:
    now = datetime.now(UTC)
    forged = jwt.encode(
        {
            "sub": str(uuid.uuid4()),
            "aud": "rag-platform-api",
            "iat": now,
            "exp": now + timedelta(hours=1),
        },
        key=None,
        algorithm="none",
    )
    with pytest.raises(AuthenticationError):
        decode_access_token(forged, make_settings())


def test_token_with_wrong_audience_is_rejected() -> None:
    settings = make_settings()
    now = datetime.now(UTC)
    token = jwt.encode(
        {
            "sub": str(uuid.uuid4()),
            "aud": "other-service",
            "iat": now,
            "exp": now + timedelta(hours=1),
        },
        settings.jwt_secret.get_secret_value(),
        algorithm="HS256",
    )
    with pytest.raises(AuthenticationError):
        decode_access_token(token, settings)


def test_garbage_token_is_rejected() -> None:
    with pytest.raises(AuthenticationError):
        decode_access_token("not.a.jwt", make_settings())
