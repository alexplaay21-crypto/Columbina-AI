"""Детерминированный провайдер для тестов и офлайн-прогона без API-ключа."""

from __future__ import annotations

import json
from collections.abc import Callable

from .base import LLMRequest, LLMResponse, ToolCall, Usage


class FakeProvider:
    """Отвечает по сценарию: функция `script(request) -> str | LLMResponse`.

    Все запросы сохраняются в `self.requests`, чтобы тесты могли проверять,
    какой контекст был собран.
    """

    def __init__(self, script: Callable[[LLMRequest], object] | None = None):
        self.script = script or (lambda req: "Хорошо." if req.json_schema is None else "{}")
        self.requests: list[LLMRequest] = []

    async def complete(self, request: LLMRequest) -> LLMResponse:
        self.requests.append(request)
        out = self.script(request)
        if isinstance(out, LLMResponse):
            return out
        if isinstance(out, dict):
            out = json.dumps(out, ensure_ascii=False)
        text = str(out)
        chars = sum(len(b.text) for b in request.system) + sum(len(m.plain_text()) for m in request.messages)
        return LLMResponse(
            text=text,
            tool_calls=[],
            usage=Usage(input_tokens=chars // 3, output_tokens=len(text) // 3),
            stop_reason="end_turn",
            model="fake",
            raw_assistant=[{"type": "text", "text": text}],
        )


def tool_response(name: str, args: dict, call_id: str = "call_1") -> LLMResponse:
    return LLMResponse(
        text="",
        tool_calls=[ToolCall(id=call_id, name=name, input=args)],
        usage=Usage(),
        stop_reason="tool_use",
        model="fake",
        raw_assistant=[{"type": "tool_use", "id": call_id, "name": name, "input": args}],
    )
