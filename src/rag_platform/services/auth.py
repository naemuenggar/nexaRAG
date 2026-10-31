from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from rag_platform.auth.passwords import hash_password, verify_password
from rag_platform.auth.tokens import create_access_token
from rag_platform.core.config import Settings
from rag_platform.core.errors import AuthenticationError, ConflictError
from rag_platform.db.models import User
from rag_platform.repositories.users import UserRepository


def normalise_email(email: str) -> str:
    return email.strip().lower()


class AuthService:
    def __init__(self, session: AsyncSession, settings: Settings) -> None:
        self._session = session
        self._users = UserRepository(session)
        self._settings = settings

    async def register(self, email: str, password: str) -> User:
        email = normalise_email(email)
        if await self._users.get_by_email(email) is not None:
            raise ConflictError("An account with this email already exists", code="email_taken")
        user = User(email=email, hashed_password=hash_password(password))
        try:
            return await self._users.add(user)
        except IntegrityError as exc:  # concurrent registration with the same email
            raise ConflictError(
                "An account with this email already exists", code="email_taken"
            ) from exc

    async def login(self, email: str, password: str) -> str:
        user = await self._users.get_by_email(normalise_email(email))
        # Always run the hash check so unknown emails take as long as wrong passwords.
        valid = verify_password(password, user.hashed_password if user else None)
        if user is None or not valid or not user.is_active:
            raise AuthenticationError("Incorrect email or password", code="invalid_credentials")
        return create_access_token(user.id, self._settings)
