from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from columbina.abilities.engine import AbilitiesEngine, AbilityContext
from columbina.abilities.moon import _find_next, moon_info
from columbina.llm.anthropic_provider import AnthropicProvider, estimate_cost
from columbina.llm.base import ImagePart, LLMRequest, Message, SystemBlock, ToolSpec, Usage
from columbina.media.links import UnsafeURL, extract_urls, ip_is_public, resolve_public, validate_url


# ---------------- луна ----------------
@pytest.mark.parametrize("kind,when", [
    ("full", datetime(2024, 12, 15, 9, 2, tzinfo=timezone.utc)),
    ("new", datetime(2025, 1, 29, 12, 36, tzinfo=timezone.utc)),
    ("new", datetime(2024, 4, 8, 18, 21, tzinfo=timezone.utc)),
    ("full", datetime(2025, 3, 14, 6, 55, tzinfo=timezone.utc)),
])
def test_moon_phases_accurate(kind, when):
    got = _find_next(when - timedelta(days=4), 180.0 if kind == "full" else 0.0)
    assert abs((got - when).total_seconds()) < 30 * 60


def test_moon_today_flag():
    m = moon_info(datetime(2024, 12, 15, 20, 0, tzinfo=timezone.utc))
    assert m.is_full_today and m.illumination > 0.95


# ---------------- способности ----------------
def test_abilities_triggers(persona):
    ae = AbilitiesEngine(persona.abilities)
    ids = [a.id for a in ae.active(AbilityContext("Напомни мне завтра купить хлеб", {"reminder_request"}, 1))]
    assert "time_reflection" in ids
    ids = [a.id for a in ae.active(AbilityContext("не спится совсем", {"user_cant_sleep", "night"}, 2))]
    assert "pain_relief" in ids
    # путь на луну до стадии 5 — только если о луне спросили напрямую
    assert "moon_path" not in [a.id for a in ae.active(AbilityContext("привет", {"full_moon"}, 2))]
    assert ae.active(AbilityContext("как дела", set(), 1)) == []


# ---------------- SSRF ----------------
@pytest.mark.parametrize("ip", ["127.0.0.1", "10.0.0.5", "192.168.1.1", "172.16.0.1", "169.254.169.254",
                                "100.64.0.1", "0.0.0.0", "::1", "fd00::1", "fe80::1", "::ffff:127.0.0.1", "224.0.0.1"])
def test_private_ips_blocked(ip):
    assert not ip_is_public(ip)


def test_public_ip_allowed():
    assert ip_is_public("93.184.216.34")


@pytest.mark.parametrize("url", ["file:///etc/passwd", "ftp://x.com", "http://user:pw@x.com/", "http://x.com:22/",
                                 "gopher://x", "http://x.com:8080/"])
def test_bad_urls(url):
    with pytest.raises(UnsafeURL):
        validate_url(url)


@pytest.mark.asyncio
@pytest.mark.parametrize("host", ["localhost", "127.0.0.1", "metadata.google.internal", "printer.local", "10.1.2.3"])
async def test_resolve_blocks(host):
    with pytest.raises(UnsafeURL):
        await resolve_public(host, 80)


def test_extract_urls():
    assert extract_urls("глянь https://example.com/a?b=1, и http://x.org).") == ["https://example.com/a?b=1", "http://x.org"]


# ---------------- провайдер ----------------
def test_anthropic_params_shape():
    p = AnthropicProvider("claude-opus-5-5", "claude-haiku-5-5", client=object())  # type: ignore[arg-type]
    req = LLMRequest(
        system=[SystemBlock("core", cache=True)],
        messages=[Message.text("user", "a"), Message.text("assistant", "b", cache_breakpoint=True),
                  Message(role="user", parts=[ImagePart(b"\xff\xd8", "image/jpeg")]), Message.text("system", "ctx")],
        tools=[ToolSpec("z_tool", "d", {"type": "object"}), ToolSpec("a_tool", "d", {"type": "object"})],
    )
    params = p.build_params(req)
    assert params["model"] == "claude-opus-5-5"
    assert params["system"][0]["cache_control"]["ttl"] == "1h"
    assert params["messages"][1]["content"][-1]["cache_control"] == {"type": "ephemeral"}
    assert params["messages"][2]["content"][0]["type"] == "image"
    assert params["messages"][3] == {"role": "system", "content": "ctx"}
    assert [t["name"] for t in params["tools"]] == ["a_tool", "z_tool"]  # детерминированный порядок
    assert params["fallbacks"] == "default" and params["betas"] == ["server-side-fallback-2026-07-01"]
    assert params["output_config"]["effort"] == "low"
    assert "thinking" not in params and "temperature" not in params

    bg = p.build_params(LLMRequest(system=[], messages=[Message.text("user", "x")], tier="background",
                                   json_schema={"type": "object"}))
    assert bg["model"] == "claude-haiku-5-5" and "fallbacks" not in bg
    assert bg["output_config"]["format"]["type"] == "json_schema"


def test_cost_estimate():
    u = Usage(input_tokens=1_000_000, output_tokens=0, cache_read_tokens=0, cache_write_tokens=0)
    assert estimate_cost("claude-opus-5-5", u) == pytest.approx(4.0)
    assert estimate_cost("claude-haiku-5-5", Usage(output_tokens=1_000_000)) == pytest.approx(0.5)
