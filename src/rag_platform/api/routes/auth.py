from fastapi import APIRouter, status

from rag_platform.api.deps import AuthRateLimit, AuthServiceDep, CurrentUserDep, SettingsDep
from rag_platform.api.schemas import LoginRequest, RegisterRequest, TokenResponse, UserOut

router = APIRouter(prefix="/auth", tags=["auth"])


@router.post(
    "/register",
    response_model=UserOut,
    status_code=status.HTTP_201_CREATED,
    dependencies=[AuthRateLimit],
)
async def register(body: RegisterRequest, service: AuthServiceDep) -> UserOut:
    user = await service.register(body.email, body.password)
    return UserOut.model_validate(user)


@router.post("/login", response_model=TokenResponse, dependencies=[AuthRateLimit])
async def login(
    body: LoginRequest, service: AuthServiceDep, settings: SettingsDep
) -> TokenResponse:
    token = await service.login(body.email, body.password)
    return TokenResponse(access_token=token, expires_in=settings.access_token_ttl_minutes * 60)


@router.get("/me", response_model=UserOut)
async def me(user: CurrentUserDep) -> UserOut:
    return UserOut.model_validate(user)
