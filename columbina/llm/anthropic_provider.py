"""Реализация LLMProvider на официальном Anthropic Python SDK."""

from __future__ import annotations

import base64
import logging
from typing import Any

import anthropic

from .base import (
    DocumentPart,
    ImagePart,
    LLMError,
    LLMRequest,
    LLMResponse,
    Message,
    TextPart,
    ToolCall,
    Usage,
)

log = logging.getLogger(__name__)

# $ за 1M токенов: (input, output, cache_read, cache_write_5m). Запись в 1h-кэш стоит 2× input.
PRICING: dict[str, tuple[float, float, float, float]] = {
    "claude-opus-5-5": (4.0, 20.0, 0.20, 5.0),
    "claude-sonnet-5-5": (2.0, 10.0, 0.10, 2.5),
    "claude-haiku-5-5": (0.10, 0.50, 0.01, 0.125),
    "claude-fable-5-2": (8.0, 40.0, 0.20, 10.0),
}

# Модели, для которых включаем серверный refusal-fallback (`fallbacks: "default"`).
FALLBACK_MODELS = {"claude-opus-5-5", "claude-sonnet-5-5", "claude-fable-5-2", "claude-fable-5-1", "claude-opus-5"}
FALLBACK_BETA = "server-side-fallback-2026-07-01"


def estimate_cost(model: str, usage: Usage) -> float:
    p_in, p_out, p_read, p_write = PRICING.get(model, PRICING["claude-opus-5-5"])
    return (
        usage.input_tokens * p_in
        + usage.output_tokens * p_out
        + usage.cache_read_tokens * p_read
        + usage.cache_write_tokens * p_write
    ) / 1_000_000


def _part_to_block(part: Any) -> dict[str, Any]:
    if isinstance(part, TextPart):
        return {"type": "text", "text": part.text}
    if isinstance(part, ImagePart):
        return {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": part.media_type,
                "data": base64.standard_b64encode(part.data).decode("ascii"),
            },
        }
    if isinstance(part, DocumentPart):
        block: dict[str, Any] = {
            "type": "document",
            "source": {
                "type": "base64",
                "media_type": part.media_type,
                "data": base64.standard_b64encode(part.data).decode("ascii"),
            },
        }
        if part.title:
            block["title"] = part.title[:200]
        return block
    raise TypeError(f"unknown part {type(part)!r}")


def _message_to_param(msg: Message) -> dict[str, Any]:
    if msg.role == "assistant" and msg.provider_raw is not None:
        return {"role": "assistant", "content": msg.provider_raw}
    if msg.tool_results:
        content: list[dict[str, Any]] = [
            {
                "type": "tool_result",
                "tool_use_id": r.tool_call_id,
                "content": r.content,
                **({"is_error": True} if r.is_error else {}),
            }
            for r in msg.tool_results
        ]
    elif msg.role == "system":
        # Mid-conversation system message: операторский канал, не часть кэша.
        return {"role": "system", "content": msg.plain_text()}
    else:
        # Документы и картинки — перед текстом.
        media = [p for p in msg.parts if not isinstance(p, TextPart)]
        texts = [p for p in msg.parts if isinstance(p, TextPart)]
        content = [_part_to_block(p) for p in media + texts]
    if msg.cache_breakpoint and content:
        content[-1] = {**content[-1], "cache_control": {"type": "ephemeral"}}
    return {"role": msg.role, "content": content}


class AnthropicProvider:
    def __init__(
        self,
        main_model: str,
        background_model: str,
        main_effort: str = "low",
        use_fallbacks: bool = True,
        client: anthropic.AsyncAnthropic | None = None,
    ):
        self.client = client or anthropic.AsyncAnthropic(max_retries=3, timeout=120.0)
        self.main_model = main_model
        self.background_model = background_model
        self.main_effort = main_effort
        self.use_fallbacks = use_fallbacks

    def model_for(self, tier: str) -> str:
        return self.main_model if tier == "main" else self.background_model

    def build_params(self, req: LLMRequest) -> dict[str, Any]:
        model = self.model_for(req.tier)
        system = []
        for b in req.system:
            block: dict[str, Any] = {"type": "text", "text": b.text}
            if b.cache:
                # 1 час: разговоры идут с паузами дольше 5 минут.
                block["cache_control"] = {"type": "ephemeral", "ttl": "1h"}
            system.append(block)

        output_config: dict[str, Any] = {}
        effort = req.effort or (self.main_effort if req.tier == "main" else "low")
        output_config["effort"] = effort
        if req.json_schema is not None:
            output_config["format"] = {"type": "json_schema", "schema": req.json_schema}

        params: dict[str, Any] = {
            "model": model,
            "max_tokens": req.max_tokens,
            "system": system,
            "messages": [_message_to_param(m) for m in req.messages],
            "output_config": output_config,
        }
        if req.tools:
            # Детерминированный порядок — чтобы не ломать кэш префикса.
            params["tools"] = [
                {"name": t.name, "description": t.description, "input_schema": t.input_schema}
                for t in sorted(req.tools, key=lambda t: t.name)
            ]
        betas: list[str] = []
        if self.use_fallbacks and model in FALLBACK_MODELS:
            betas.append(FALLBACK_BETA)
            params["fallbacks"] = "default"
        if betas:
            params["betas"] = betas
        return params

    async def complete(self, req: LLMRequest) -> LLMResponse:
        params = self.build_params(req)
        try:
            response = await self.client.beta.messages.create(**params)
        except anthropic.BadRequestError as e:
            log.error("LLM bad request: %s", e.message)
            raise LLMError(f"bad request: {e.message}", retryable=False) from e
        except (anthropic.AuthenticationError, anthropic.PermissionDeniedError, anthropic.NotFoundError) as e:
            raise LLMError(f"auth/model error: {e}", retryable=False) from e
        except anthropic.RateLimitError as e:
            raise LLMError("rate limited", retryable=True) from e
        except anthropic.APIStatusError as e:
            raise LLMError(f"api status {e.status_code}", retryable=e.status_code >= 500) from e
        except anthropic.APIConnectionError as e:
            raise LLMError("connection error", retryable=True) from e

        u = response.usage
        usage = Usage(
            input_tokens=u.input_tokens or 0,
            output_tokens=u.output_tokens or 0,
            cache_read_tokens=getattr(u, "cache_read_input_tokens", 0) or 0,
            cache_write_tokens=getattr(u, "cache_creation_input_tokens", 0) or 0,
        )
        texts: list[str] = []
        calls: list[ToolCall] = []
        for block in response.content:
            if block.type == "text":
                texts.append(block.text)
            elif block.type == "tool_use":
                calls.append(ToolCall(id=block.id, name=block.name, input=dict(block.input or {})))
        served_model = getattr(response, "model", None) or params["model"]
        raw = [b.model_dump(mode="json", exclude_none=True) for b in response.content]
        return LLMResponse(
            text="".join(texts).strip(),
            tool_calls=calls,
            usage=usage,
            stop_reason=response.stop_reason or "end_turn",
            model=served_model,
            raw_assistant=raw,
            cost_usd=estimate_cost(served_model, usage),
        )
