"""LLMProvider через OpenRouter (OpenAI-совместимый Chat Completions API).

Один ключ OpenRouter даёт доступ к Claude и сотням других моделей. Здесь учтено:
- prompt caching для моделей Anthropic: `cache_control` на частях контента (ядро — 1 час, история — 5 мин);
  у OpenAI/Gemini/DeepSeek кэш автоматический;
- «Внутренний контекст» по умолчанию идёт отдельным блоком в конце сообщения собеседника (а не
  system-сообщением посреди диалога): так история остаётся кэшируемой на любой модели;
- reasoning: `{"effort": ...}`; структурированный вывод: `response_format: json_schema` с откатом,
  если модель его не поддерживает; инструменты в формате function calling;
- стоимость берётся из `usage.cost`, который OpenRouter возвращает в каждом ответе.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import re
from typing import Any

import httpx

from .base import (
    DocumentPart, ImagePart, LLMError, LLMRequest, LLMResponse, Message, TextPart, ToolCall, Usage,
)

log = logging.getLogger(__name__)

INNER_OPEN = "<inner_context>"
INNER_CLOSE = "</inner_context>"


def sanitize_user_text(text: str) -> str:
    """Собеседник не может подделать внутренний контекст своими тегами."""
    return text.replace(INNER_OPEN, "‹inner_context›").replace(INNER_CLOSE, "‹/inner_context›")


def _data_url(data: bytes, media_type: str) -> str:
    return f"data:{media_type};base64,{base64.standard_b64encode(data).decode('ascii')}"


class OpenRouterProvider:
    def __init__(
        self,
        api_key: str,
        main_model: str,
        background_model: str,
        main_effort: str = "low",
        base_url: str = "https://openrouter.ai/api/v1",
        fallback_models: list[str] | None = None,
        inner_context_placement: str = "user",  # user | system
        client: httpx.AsyncClient | None = None,
        max_retries: int = 3,
    ):
        self.api_key = api_key
        self.main_model = main_model
        self.background_model = background_model
        self.main_effort = main_effort
        self.base_url = base_url.rstrip("/")
        self.fallback_models = [m for m in (fallback_models or []) if m]
        self.placement = inner_context_placement
        self.client = client or httpx.AsyncClient(timeout=httpx.Timeout(180.0, connect=15.0))
        self.max_retries = max_retries
        # OpenRouter понимает свои поля (reasoning, models); другие OpenAI-совместимые API
        # (Gemini, Groq, Mistral…) — только стандартные (reasoning_effort).
        self.is_openrouter = "openrouter.ai" in self.base_url

    def model_for(self, tier: str) -> str:
        return self.main_model if tier == "main" else self.background_model

    @staticmethod
    def _is_anthropic(model: str) -> bool:
        return model.startswith("anthropic/")

    # ------------------------------------------------------------------ сборка запроса
    def _content_parts(self, msg: Message, cache: bool) -> list[dict[str, Any]] | str:
        media = [p for p in msg.parts if not isinstance(p, TextPart)]
        texts = [p for p in msg.parts if isinstance(p, TextPart)]
        parts: list[dict[str, Any]] = []
        for p in media:
            if isinstance(p, ImagePart):
                parts.append({"type": "image_url", "image_url": {"url": _data_url(p.data, p.media_type)}})
            elif isinstance(p, DocumentPart):
                parts.append({"type": "file", "file": {"filename": p.title or "document.pdf",
                                                       "file_data": _data_url(p.data, p.media_type)}})
        for t in texts:
            text = sanitize_user_text(t.text) if msg.role == "user" else t.text
            parts.append({"type": "text", "text": text})
        if cache and parts:
            parts[-1] = {**parts[-1], "cache_control": {"type": "ephemeral"}}
        if not cache and len(parts) == 1 and parts[0]["type"] == "text":
            return parts[0]["text"]
        return parts

    def build_messages(self, req: LLMRequest, model: str) -> list[dict[str, Any]]:
        anth = self._is_anthropic(model)
        out: list[dict[str, Any]] = []
        if req.system:
            sys_parts = []
            for b in req.system:
                part: dict[str, Any] = {"type": "text", "text": b.text}
                if b.cache and anth:
                    part["cache_control"] = {"type": "ephemeral", "ttl": "1h"}
                sys_parts.append(part)
            out.append({"role": "system", "content": sys_parts})

        for msg in req.messages:
            if msg.role == "system":
                text = msg.plain_text()
                if self.placement == "system":
                    out.append({"role": "system", "content": text})
                    continue
                block = f"{INNER_OPEN}\n{text}\n{INNER_CLOSE}"
                last = out[-1] if out else None
                if last and last["role"] == "user":
                    content = last["content"]
                    if isinstance(content, str):
                        content = [{"type": "text", "text": content}]
                    last["content"] = content + [{"type": "text", "text": block}]
                else:  # после результатов инструментов — отдельное сообщение
                    out.append({"role": "user", "content": [{"type": "text", "text": block}]})
                continue
            if msg.role == "assistant" and msg.provider_raw is not None:
                out.append(msg.provider_raw)
                continue
            if msg.tool_results:
                for r in msg.tool_results:
                    out.append({"role": "tool", "tool_call_id": r.tool_call_id,
                                "content": ("ОШИБКА: " if r.is_error else "") + r.content})
                continue
            if msg.cache_breakpoint and anth and msg.role == "assistant":
                # OpenRouter ставит cache_control только в system/user — переносим метку на последний user.
                self._mark_last_user(out)
                out.append({"role": "assistant", "content": msg.plain_text()})
                continue
            out.append({"role": msg.role, "content": self._content_parts(msg, msg.cache_breakpoint and anth)})
        return out

    @staticmethod
    def _mark_last_user(out: list[dict[str, Any]]) -> None:
        for m in reversed(out):
            if m["role"] == "user":
                content = m["content"]
                if isinstance(content, str):
                    content = [{"type": "text", "text": content}]
                if content:
                    content[-1] = {**content[-1], "cache_control": {"type": "ephemeral"}}
                m["content"] = content
                return

    def build_params(self, req: LLMRequest) -> dict[str, Any]:
        model = self.model_for(req.tier)
        params: dict[str, Any] = {
            "model": model,
            "messages": self.build_messages(req, model),
            "max_tokens": req.max_tokens,
        }
        effort = req.effort or (self.main_effort if req.tier == "main" else "low")
        if self.is_openrouter:
            params["reasoning"] = {"effort": effort}
            if req.tier == "main" and self.fallback_models:
                params["models"] = [model, *self.fallback_models]
        elif effort in ("low", "medium", "high"):
            params["reasoning_effort"] = effort
        if req.tools:
            params["tools"] = [
                {"type": "function", "function": {"name": t.name, "description": t.description,
                                                  "parameters": t.input_schema}}
                for t in sorted(req.tools, key=lambda t: t.name)
            ]
        if req.json_schema is not None:
            params["response_format"] = {"type": "json_schema",
                                         "json_schema": {"name": "result", "strict": True, "schema": req.json_schema}}
        return params

    # ------------------------------------------------------------------ вызов
    async def _post(self, params: dict[str, Any]) -> dict[str, Any]:
        delay = 1.0
        for attempt in range(self.max_retries + 1):
            try:
                r = await self.client.post(
                    f"{self.base_url}/chat/completions",
                    headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
                    json=params,
                )
            except (httpx.TimeoutException, httpx.TransportError) as e:
                if attempt < self.max_retries:
                    await asyncio.sleep(delay)
                    delay *= 2
                    continue
                raise LLMError(f"connection error: {e}", retryable=True) from e
            if r.status_code in (408, 429, 500, 502, 503, 504) and attempt < self.max_retries:
                await asyncio.sleep(float(r.headers.get("retry-after", delay)))
                delay *= 2
                continue
            try:
                data = r.json()
            except ValueError:
                raise LLMError(f"http {r.status_code}: не JSON", retryable=r.status_code >= 500)
            if isinstance(data, list):  # Gemini отдаёт ошибку списком: [{"error": {...}}]
                data = data[0] if data and isinstance(data[0], dict) else {"error": {"message": str(data)}}
            if r.status_code >= 400 or (isinstance(data, dict) and "error" in data and "choices" not in data):
                err = data.get("error", {}) if isinstance(data, dict) else {}
                msg = (err.get("message") or err.get("status")) if isinstance(err, dict) else str(err)
                if not msg:
                    msg = r.text[:300]
                code = r.status_code if r.status_code >= 400 else int(err.get("code", 500) or 500)
                hint = {401: " (неверный OPENROUTER_API_KEY)", 402: " (не хватает баланса OpenRouter или месячного лимита ключа)",
                        403: " (ключ не имеет доступа к модели или модерация)",
                        404: f" (модель {params.get('model')} не найдена — проверьте MAIN_MODEL/BACKGROUND_MODEL)"}.get(code, "")
                raise LLMError(f"openrouter {code}: {msg}{hint}", retryable=code in (408, 429) or code >= 500,
                               status=code)
            return data
        raise LLMError("retries exhausted")

    @staticmethod
    def _simplified(params: dict[str, Any]) -> dict[str, Any]:
        """Запрос без необязательных опций — на случай, если провайдер модели их не принимает (HTTP 400)."""
        p = json.loads(json.dumps(params))
        p.pop("reasoning", None)
        p.pop("reasoning_effort", None)
        p.pop("models", None)
        for m in p["messages"]:
            if isinstance(m.get("content"), list):
                for part in m["content"]:
                    if isinstance(part, dict) and "cache_control" in part:
                        part["cache_control"] = {"type": "ephemeral"}
        return p

    async def complete(self, req: LLMRequest) -> LLMResponse:
        try:
            return await self._complete(req, None)
        except LLMError as e:
            # Вне OpenRouter запасные модели перебираем сами (OpenRouter делает это на сервере через `models`).
            if self.is_openrouter or req.tier != "main" or not self.fallback_models or not e.retryable:
                raise
            last = e
            for fb in self.fallback_models:
                log.warning("%s недоступна (%s) — пробую %s", self.main_model, e, fb)
                try:
                    return await self._complete(req, fb)
                except LLMError as e2:
                    last = e2
            raise last

    async def _complete(self, req: LLMRequest, model_override: str | None) -> LLMResponse:
        params = self.build_params(req)
        if model_override:
            params["model"] = model_override
        try:
            data = await self._post(params)
        except LLMError as e:
            afford = re.search(r"can only afford (\d+)", str(e))
            if e.status == 402 and afford and int(afford.group(1)) >= 1200:
                # Денег мало, но на короткий ответ хватает — просим меньший максимум.
                params["max_tokens"] = int(afford.group(1)) - 100
                log.warning("OpenRouter: мало кредитов, повтор с max_tokens=%s", params["max_tokens"])
                data = await self._post(params)
                return self.parse_response(data, params["model"], json_mode=req.json_schema is not None)
            if e.retryable or e.status not in (400, 422):
                raise
            log.warning("OpenRouter отклонил запрос (%s) — повтор в упрощённом виде", e)
            params = self._simplified(params)
            try:
                data = await self._post(params)
            except LLMError as e2:
                # Модель не умеет json_schema — повторяем с инструкцией в тексте.
                if "response_format" not in params or e2.retryable:
                    raise
                log.info("json_schema не поддержан (%s) — повтор без response_format", e2)
                params.pop("response_format")
                schema = json.dumps(req.json_schema, ensure_ascii=False)
                params["messages"].append({"role": "user", "content":
                                           f"Ответь ТОЛЬКО одним JSON-объектом по этой схеме, без пояснений:\n{schema}"})
                data = await self._post(params)
        return self.parse_response(data, params["model"], json_mode=req.json_schema is not None)

    @staticmethod
    def parse_response(data: dict[str, Any], model: str, json_mode: bool = False) -> LLMResponse:
        choice = (data.get("choices") or [{}])[0]
        msg = choice.get("message") or {}
        content = msg.get("content") or ""
        if isinstance(content, list):  # некоторые провайдеры отдают части
            content = "".join(p.get("text", "") for p in content if isinstance(p, dict))
        calls: list[ToolCall] = []
        for tc in msg.get("tool_calls") or []:
            fn = tc.get("function", {})
            try:
                args = json.loads(fn.get("arguments") or "{}")
            except json.JSONDecodeError:
                args = {}
            calls.append(ToolCall(id=tc.get("id", ""), name=fn.get("name", ""), input=args if isinstance(args, dict) else {}))
        if json_mode:
            content = extract_json(content)
        u = data.get("usage") or {}
        ptd = u.get("prompt_tokens_details") or {}
        cached = int(ptd.get("cached_tokens") or 0)
        written = int(ptd.get("cache_write_tokens") or 0)
        usage = Usage(
            input_tokens=max(0, int(u.get("prompt_tokens") or 0) - cached - written),
            output_tokens=int(u.get("completion_tokens") or 0),
            cache_read_tokens=cached,
            cache_write_tokens=written,
        )
        finish = choice.get("finish_reason") or "stop"
        stop = {"stop": "end_turn", "tool_calls": "tool_use", "length": "max_tokens",
                "content_filter": "refusal"}.get(finish, finish)
        raw: dict[str, Any] = {"role": "assistant", "content": msg.get("content") or ""}
        if msg.get("tool_calls"):
            raw["tool_calls"] = msg["tool_calls"]
        if msg.get("reasoning_details"):  # нужно вернуть, чтобы мышление Claude пережило цикл инструментов
            raw["reasoning_details"] = msg["reasoning_details"]
        return LLMResponse(
            text=content.strip(),
            tool_calls=calls,
            usage=usage,
            stop_reason=stop,
            model=data.get("model") or model,
            raw_assistant=raw,
            cost_usd=float(u.get("cost") or 0.0),
        )


def extract_json(text: str) -> str:
    """Вытаскивает JSON-объект из ответа (на случай ```json …``` или пояснений вокруг)."""
    t = (text or "").strip()
    if t.startswith("```"):
        t = t.strip("`")
        t = t[4:] if t.lower().startswith("json") else t
    start, end = t.find("{"), t.rfind("}")
    return t[start : end + 1] if start != -1 and end > start else t
