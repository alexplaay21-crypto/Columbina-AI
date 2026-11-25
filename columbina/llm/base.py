"""Провайдер-независимый интерфейс LLM.

Ядро (cognition, memory, initiative) работает только с этими типами. Чтобы сменить
провайдера, достаточно реализовать `LLMProvider.complete` для нового API.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, Union

Role = Literal["user", "assistant", "system"]


@dataclass
class TextPart:
    text: str


@dataclass
class ImagePart:
    data: bytes
    media_type: str = "image/jpeg"


@dataclass
class DocumentPart:
    """PDF-документ, который модель читает целиком."""

    data: bytes
    media_type: str = "application/pdf"
    title: str | None = None


Part = Union[TextPart, ImagePart, DocumentPart]


@dataclass
class ToolCall:
    id: str
    name: str
    input: dict[str, Any]


@dataclass
class ToolResult:
    tool_call_id: str
    content: str
    is_error: bool = False


@dataclass
class Message:
    role: Role
    parts: list[Part] = field(default_factory=list)
    # Для продолжения цикла инструментов: сырой ответ провайдера (с thinking-блоками),
    # который нужно передать обратно без изменений.
    provider_raw: Any = None
    tool_results: list[ToolResult] = field(default_factory=list)
    cache_breakpoint: bool = False

    @classmethod
    def text(cls, role: Role, text: str, cache_breakpoint: bool = False) -> "Message":
        return cls(role=role, parts=[TextPart(text)], cache_breakpoint=cache_breakpoint)

    def plain_text(self) -> str:
        return "\n".join(p.text for p in self.parts if isinstance(p, TextPart))


@dataclass
class SystemBlock:
    text: str
    cache: bool = False


@dataclass
class ToolSpec:
    name: str
    description: str
    input_schema: dict[str, Any]


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0

    def __add__(self, other: "Usage") -> "Usage":
        return Usage(
            self.input_tokens + other.input_tokens,
            self.output_tokens + other.output_tokens,
            self.cache_read_tokens + other.cache_read_tokens,
            self.cache_write_tokens + other.cache_write_tokens,
        )


Tier = Literal["main", "background"]


@dataclass
class LLMRequest:
    system: list[SystemBlock]
    messages: list[Message]
    tier: Tier = "main"
    max_tokens: int = 2000
    effort: str | None = None
    tools: list[ToolSpec] = field(default_factory=list)
    json_schema: dict[str, Any] | None = None  # структурированный вывод
    purpose: str = "chat"  # для учёта расходов


@dataclass
class LLMResponse:
    text: str
    tool_calls: list[ToolCall]
    usage: Usage
    stop_reason: str
    model: str
    raw_assistant: Any = None  # передаётся обратно в Message.provider_raw при цикле инструментов
    cost_usd: float = 0.0


class LLMError(Exception):
    """Ошибка провайдера, после которой стоит ответить «бледным» сообщением."""

    def __init__(self, message: str, retryable: bool = True):
        super().__init__(message)
        self.retryable = retryable


class LLMProvider(Protocol):
    async def complete(self, request: LLMRequest) -> LLMResponse: ...
