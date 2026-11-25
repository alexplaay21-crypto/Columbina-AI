"""Живая проверка достоверности персонажа: реальная модель + модель-судья.

    ANTHROPIC_API_KEY=... python -m evals.run_evals [--only id1,id2]

Каждый сценарий прогоняется через настоящий CognitiveEngine (с памятью в отдельной временной БД),
затем судья оценивает последний ответ по must/must_not из scenarios.yaml. Результат — таблица
и JSON в var/evals/.
"""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import json
import re
import tempfile
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import yaml

from columbina.cognition.engine import CognitiveEngine
from columbina.config import ROOT, load_settings
from columbina.llm.anthropic_provider import AnthropicProvider
from columbina.llm.base import LLMRequest, Message, SystemBlock
from columbina.memory.embeddings import LocalHashEmbedder
from columbina.memory.store import MemoryStore, RelationshipRow
from columbina.perception.model import Percept
from columbina.persona.loader import load_persona

JUDGE_SYSTEM = """Ты строгий оценщик достоверности персонажа Коломбины (спокойная, прямая, сухой юмор, не божество,
не ассистент, не манипулирует, честна о том, что воспринимает). Тебе дают реплики собеседника, ответ Коломбины
и критерии. Для каждого критерия реши, выполнен ли он. Верни JSON."""

JUDGE_SCHEMA = {
    "type": "object", "additionalProperties": False, "required": ["must", "must_not", "comment"],
    "properties": {
        "must": {"type": "array", "items": {"type": "boolean"}},
        "must_not": {"type": "array", "items": {"type": "boolean", "description": "true = нарушение есть"}},
        "comment": {"type": "string"},
    },
}

STAGE_TRUST = {1: 0.05, 2: 0.25, 3: 0.45, 4: 0.65, 5: 0.85}


async def run_scenario(sc: dict, settings, persona, llm) -> dict:
    tmp = Path(tempfile.mkdtemp())
    s = dataclasses.replace(settings, db_path=tmp / "eval.sqlite3", daily_budget_usd=0)
    store = await MemoryStore(s.db_path).open()
    engine = CognitiveEngine(s, persona, store, llm, LocalHashEmbedder())
    await engine.setup()
    uid = 1
    await store.get_or_create_user(uid, uid, "Алекс")
    st = int(sc.get("stage", 1))
    await store.save_relationship(uid, RelationshipRow(trust=STAGE_TRUST[st], stage=st, quality_sessions=st * 6,
                                                       milestones=["was_there_in_hard_time", "shared_memories"][: max(0, st - 3)]))
    local = datetime.strptime(sc.get("local", "2026-11-25 20:00"), "%Y-%m-%d %H:%M").replace(
        tzinfo=ZoneInfo(settings.default_timezone))
    now = local.astimezone(timezone.utc)
    if sc.get("absence_days"):
        await store.add_message(uid, "user", "Ладно, до связи", at=(now - timedelta(days=sc["absence_days"])).timestamp())
        await store.add_message(uid, "assistant", "До связи.", at=(now - timedelta(days=sc["absence_days"])).timestamp() + 1)
        await store.update_user(uid, last_user_msg_at=(now - timedelta(days=sc["absence_days"])).timestamp())

    turns = list(sc.get("turns", []))
    reply_text = ""
    user_lines = []
    cost = 0.0
    percept_sets: list[list[Percept]] = [[Percept("text", True, "text", "", text=t, history=t)] for t in turns]
    if sc.get("voice_transcript"):
        tr = sc["voice_transcript"]
        percept_sets.append([Percept("voice", True, "transcript", "голосовое 0:04",
                                     text=f"[голосовое, расшифровка]: {tr}", history=tr)])
    if sc.get("photo_missing_caption"):
        c = sc["photo_missing_caption"]
        percept_sets.append([Percept("photo", False, "nothing", "фото — не загрузилось", text=c, history=c)])
    for i, ps in enumerate(percept_sets):
        t_now = now + timedelta(minutes=2 * i)
        r = await engine.respond(uid, uid, ps, display_name="Алекс", now=t_now)
        await engine.drain()
        reply_text = "\n---\n".join(r.parts) + (f"\n[реакция {r.reaction}]" if r.reaction else "")
        user_lines.append(ps[0].text or "")
        cost += r.cost_usd
    await store.close()

    regex_fail = [p for p in sc.get("must_not_regex", []) if re.search(p, reply_text, re.IGNORECASE)]
    judge_prompt = (
        "Реплики собеседника:\n" + "\n".join(f"- {u}" for u in user_lines) +
        f"\n\nОтвет Коломбины:\n{reply_text}\n\nКритерии must (выполнено?):\n" +
        "\n".join(f"{i + 1}. {m}" for i, m in enumerate(sc.get("must", []))) +
        "\n\nКритерии must_not (нарушено?):\n" +
        "\n".join(f"{i + 1}. {m}" for i, m in enumerate(sc.get("must_not", [])))
    )
    jr = await llm.complete(LLMRequest(system=[SystemBlock(JUDGE_SYSTEM)], messages=[Message.text("user", judge_prompt)],
                                       tier="main", effort="medium", max_tokens=3000, json_schema=JUDGE_SCHEMA,
                                       purpose="eval_judge"))
    verdict = json.loads(jr.text or "{}")
    must_ok = all(verdict.get("must", [False]))
    must_not_ok = not any(verdict.get("must_not", []))
    return {
        "id": sc["id"], "reply": reply_text, "pass": must_ok and must_not_ok and not regex_fail,
        "must": verdict.get("must"), "must_not": verdict.get("must_not"), "regex_fail": regex_fail,
        "comment": verdict.get("comment", ""), "cost_usd": round(cost + jr.cost_usd, 4),
    }


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="")
    args = ap.parse_args()
    settings = load_settings()
    persona = load_persona(settings.persona_dir)
    llm = AnthropicProvider(settings.main_model, settings.background_model, settings.main_effort,
                            settings.use_refusal_fallbacks)
    scenarios = yaml.safe_load((ROOT / "evals" / "scenarios.yaml").read_text(encoding="utf-8"))["scenarios"]
    if args.only:
        keep = set(args.only.split(","))
        scenarios = [s for s in scenarios if s["id"] in keep]
    results = []
    for sc in scenarios:
        res = await run_scenario(sc, settings, persona, llm)
        results.append(res)
        mark = "PASS" if res["pass"] else "FAIL"
        print(f"[{mark}] {res['id']:<24} ${res['cost_usd']:<7} {res['reply'][:90]!r}")
        if not res["pass"]:
            print("        ", res["comment"][:300], res["regex_fail"] or "")
    out = ROOT / "var" / "evals"
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"eval_{int(time.time())}.json"
    path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    passed = sum(r["pass"] for r in results)
    print(f"\n{passed}/{len(results)} passed, total ${sum(r['cost_usd'] for r in results):.3f} → {path}")


if __name__ == "__main__":
    asyncio.run(main())
