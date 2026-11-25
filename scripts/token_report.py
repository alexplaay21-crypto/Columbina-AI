"""Оценка расхода токенов на один ход и на день — без API (эвристика) или точно (--exact, нужен ключ).

    python scripts/token_report.py            # оценка по символам
    python scripts/token_report.py --exact    # messages.count_tokens через API
"""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from columbina.cognition.context_builder import estimate_tokens  # noqa: E402
from columbina.cognition.engine import CognitiveEngine  # noqa: E402
from columbina.config import ROOT, Settings  # noqa: E402
from columbina.llm.anthropic_provider import PRICING  # noqa: E402
from columbina.llm.fake import FakeProvider  # noqa: E402
from columbina.memory.embeddings import LocalHashEmbedder  # noqa: E402
from columbina.memory.store import MemoryStore  # noqa: E402
from columbina.perception.model import Percept  # noqa: E402
from columbina.persona.loader import load_persona  # noqa: E402


async def build_sample():
    tmp = Path(tempfile.mkdtemp())
    s = dataclasses.replace(Settings(), db_path=tmp / "x.sqlite3", daily_budget_usd=0)
    store = await MemoryStore(s.db_path).open()
    fake = FakeProvider(lambda r: "Хм. А если по-другому? Расскажи, что было дальше." if r.tier == "main" else "{}")
    persona = load_persona(ROOT / "data" / "persona")
    e = CognitiveEngine(s, persona, store, fake, LocalHashEmbedder())
    await e.setup()
    for i in range(10):
        await store.add_episode(1, f"Обсуждали работу и кофе, эпизод {i}: собеседник рассказывал про проект", 3)
        await store.upsert_fact(1, f"факт {i}", "работает дизайнером, любит горы")
    msgs = [
        "Привет! Как ты там?", "Сегодня был странный день на работе, начальник опять устроил разнос",
        "Да ладно, переживу. А ты что делала?", "Смешно. А духи правда тают?", "Слушай, а что ты любишь есть?",
        "Я вот вчера пекла пирог с ягодами", "Ахах, да, подгорел немного", "Ладно, пойду спать скоро",
    ] * 2  # ~16 ходов → ~32 сообщения в истории
    for m in msgs:
        await e.respond(1, 1, [Percept("text", True, "text", "", text=m, history=m)])
    await e.drain()
    await store.close()
    return fake, persona


async def main(exact: bool) -> None:
    fake, persona = await build_sample()
    req = [r for r in fake.requests if r.tier == "main"][-1]
    system_tok = estimate_tokens(persona.core_prompt)
    inner = req.messages[-1].plain_text()
    inner_tok = estimate_tokens(inner)
    hist_tok = sum(estimate_tokens(m.plain_text()) for m in req.messages[:-2])
    new_tok = estimate_tokens(req.messages[-2].plain_text())
    tools_tok = 150
    if exact:
        import anthropic

        from columbina.llm.anthropic_provider import AnthropicProvider

        p = AnthropicProvider("claude-opus-5-5", "claude-haiku-5-5")
        params = p.build_params(req)
        client = anthropic.Anthropic()
        total = client.messages.count_tokens(model=params["model"], system=params["system"],
                                             messages=params["messages"], tools=params.get("tools", [])).input_tokens
        print(f"Точно (count_tokens): {total} входных токенов на ход")
    print("Оценка входа основного запроса (токены):")
    print(f"  system (ядро, кэш 1ч)        ~{system_tok}")
    print(f"  tools                        ~{tools_tok}")
    print(f"  история ({len(req.messages) - 2} сообщ., кэш 5м) ~{hist_tok}")
    print(f"  новое сообщение              ~{new_tok}")
    print(f"  внутренний контекст          ~{inner_tok}")

    out_tok = 350  # ответ ~80 токенов + адаптивное мышление на effort=low
    for model in ("claude-opus-5-5", "claude-sonnet-5-5", "claude-haiku-5-5"):
        p_in, p_out, p_read, p_write = PRICING[model]
        cached = system_tok + tools_tok + hist_tok * 0.85
        fresh = new_tok + inner_tok + hist_tok * 0.15
        per_turn = (cached * p_read + fresh * p_in + hist_tok * 0.15 * p_write + out_tok * p_out) / 1e6
        # фон: рефлексия ~1.5k вход + 250 выход на Haiku 5.5 на каждый ход, резюме раз в ~8 ходов
        bg = (1500 * PRICING["claude-haiku-5-5"][0] + 250 * PRICING["claude-haiku-5-5"][1]) / 1e6
        day = 60 * (per_turn + bg) + 2 * per_turn  # 60 сообщений + 2 инициативы
        print(f"\n{model}: ~${per_turn:.4f} за ход, фон ~${bg:.5f}; день (60 сообщений) ≈ ${day:.2f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--exact", action="store_true")
    asyncio.run(main(ap.parse_args().exact))
