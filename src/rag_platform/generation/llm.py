"""LLM provider interface and HTTP implementations.

Only this module knows provider wire formats. Everything else sees ``LLMProvider``.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, Protocol

import httpx

if TYPE_CHECKING:
    from rag_platform.core.config import Settings

Role = Literal["system", "user", "assistant"]


@dataclass(frozen=True, slots=True)
class ChatMessage:
    role: Role
    content: str


@dataclass(frozen=True, slots=True)
class LLMResult:
    text: str
    model: str
    prompt_tokens: int
    completion_tokens: int


class LLMError(Exception):
    """The provider failed or returned something unusable. Never shown verbatim to users."""


class LLMProvider(Protocol):
    @property
    def model(self) -> str: ...

    async def complete(
        self, messages: Sequence[ChatMessage], *, max_tokens: int, temperature: float
    ) -> LLMResult: ...

    async def aclose(self) -> None: ...


class OpenAICompatibleProvider:
    """``POST {base_url}/chat/completions``: OpenAI, Azure OpenAI, vLLM, Ollama, LiteLLM, etc."""

    def __init__(
        self, *, base_url: str, api_key: str | None, model: str, timeout_seconds: float
    ) -> None:
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self._client = httpx.AsyncClient(
            base_url=base_url.rstrip("/"), headers=headers, timeout=timeout_seconds
        )
        self._model = model

    @property
    def model(self) -> str:
        return self._model

    async def complete(
        self, messages: Sequence[ChatMessage], *, max_tokens: int, temperature: float
    ) -> LLMResult:
        payload = {
            "model": self._model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        body = await _post_json(self._client, "/chat/completions", payload)
        try:
            text = body["choices"][0]["message"]["content"] or ""
            usage = body.get("usage") or {}
            return LLMResult(
                text=text,
                model=body.get("model", self._model),
                prompt_tokens=int(usage.get("prompt_tokens", 0)),
                completion_tokens=int(usage.get("completion_tokens", 0)),
            )
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            raise LLMError("unexpected response shape") from exc

    async def aclose(self) -> None:
        await self._client.aclose()


class AnthropicProvider:
    """Anthropic Messages API (``POST /v1/messages``)."""

    API_VERSION = "2023-06-01"

    def __init__(
        self, *, base_url: str, api_key: str | None, model: str, timeout_seconds: float
    ) -> None:
        headers = {"anthropic-version": self.API_VERSION}
        if api_key:
            headers["x-api-key"] = api_key
        self._client = httpx.AsyncClient(
            base_url=base_url.rstrip("/"), headers=headers, timeout=timeout_seconds
        )
        self._model = model

    @property
    def model(self) -> str:
        return self._model

    async def complete(
        self, messages: Sequence[ChatMessage], *, max_tokens: int, temperature: float
    ) -> LLMResult:
        system = "\n\n".join(m.content for m in messages if m.role == "system")
        payload: dict[str, Any] = {
            "model": self._model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": [
                {"role": m.role, "content": m.content} for m in messages if m.role != "system"
            ],
        }
        if system:
            payload["system"] = system
        body = await _post_json(self._client, "/v1/messages", payload)
        try:
            text = "".join(
                block.get("text", "") for block in body["content"] if block.get("type") == "text"
            )
            usage = body.get("usage") or {}
            return LLMResult(
                text=text,
                model=body.get("model", self._model),
                prompt_tokens=int(usage.get("input_tokens", 0)),
                completion_tokens=int(usage.get("output_tokens", 0)),
            )
        except (KeyError, TypeError, ValueError, AttributeError) as exc:
            raise LLMError("unexpected response shape") from exc

    async def aclose(self) -> None:
        await self._client.aclose()


async def _post_json(client: httpx.AsyncClient, path: str, payload: dict[str, Any]) -> Any:
    try:
        response = await client.post(path, json=payload)
    except httpx.HTTPError as exc:
        raise LLMError(f"request failed: {type(exc).__name__}") from exc
    if response.status_code >= 400:
        # The status is enough for logs; provider error bodies can echo the prompt.
        raise LLMError(f"provider returned HTTP {response.status_code}")
    try:
        return response.json()
    except ValueError as exc:
        raise LLMError("provider returned invalid JSON") from exc


def create_llm(settings: "Settings") -> LLMProvider | None:
    """The configured provider, or None when generation is disabled."""
    from rag_platform.core.config import LLMProviderName

    if settings.llm_provider is LLMProviderName.NONE:
        return None
    api_key = settings.llm_api_key.get_secret_value() if settings.llm_api_key else None
    provider_cls: type[OpenAICompatibleProvider] | type[AnthropicProvider] = (
        AnthropicProvider
        if settings.llm_provider is LLMProviderName.ANTHROPIC
        else OpenAICompatibleProvider
    )
    return provider_cls(
        base_url=settings.llm_base_url,
        api_key=api_key,
        model=settings.llm_model,
        timeout_seconds=settings.llm_timeout_seconds,
    )
