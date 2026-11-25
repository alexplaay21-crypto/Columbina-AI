"""Чтение страниц: Jina Reader (r.jina.ai, ключ необязателен) с запасным локальным SafeFetcher.

Адрес всегда проверяется на SSRF до отправки куда-либо (даже в Jina): внутренние адреса не читаются.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

import httpx

from ..media.links import SafeFetcher, UnsafeURL, ip_is_public, resolve_public, validate_url

log = logging.getLogger(__name__)
MD_LINK = re.compile(r"(?<!!)\[([^\]]{2,80})\]\((https?://[^)\s]+)\)")
MD_TITLE = re.compile(r"^Title:\s*(.+)$", re.M)


def _is_ip(host: str) -> bool:
    import ipaddress

    try:
        ipaddress.ip_address(host)
        return True
    except ValueError:
        return host.lower() in {"localhost"}


@dataclass
class Page:
    url: str
    title: str
    text: str
    links: list[tuple[str, str]] = field(default_factory=list)
    truncated: bool = False
    via: str = ""


class WebReader:
    def __init__(self, client: httpx.AsyncClient, fetcher: SafeFetcher, jina_key: str = "", max_chars: int = 6000,
                 use_jina: bool = True):
        self.client = client
        self.fetcher = fetcher
        self.jina_key = jina_key
        self.max_chars = max_chars
        self.use_jina = use_jina

    async def _check(self, url: str) -> None:
        _, host, port = validate_url(url)
        await resolve_public(host, port)

    async def read(self, url: str) -> Page:
        await self._check(url)  # UnsafeURL — наружу
        if self.use_jina:
            try:
                return await self._jina(url)
            except Exception as e:  # noqa: BLE001
                log.info("jina failed for %s: %s — читаю сама", url, e)
        p = await self.fetcher.fetch(url)
        return Page(p.url, p.title, p.text[: self.max_chars], [], p.truncated, "direct")

    async def fetch(self, url: str) -> Page:  # совместимость с SafeFetcher (ссылки в сообщениях)
        return await self.read(url)

    async def _jina(self, url: str) -> Page:
        headers = {"Accept": "text/plain", "X-Return-Format": "markdown", "X-With-Links-Summary": "true",
                   "X-Remove-Selector": "header, nav, footer, aside, .navbox, .mw-jump-link, #mw-navigation, "
                                        ".vector-header-container, .cookie, [role=navigation]"}
        if self.jina_key:
            headers["Authorization"] = f"Bearer {self.jina_key}"
        r = await self.client.get("https://r.jina.ai/" + url, headers=headers)
        r.raise_for_status()
        raw = r.text[:400_000]
        title_m = MD_TITLE.search(raw[:2000])
        body = raw.split("Markdown Content:", 1)[-1].strip()
        links = []
        for t, u in MD_LINK.findall(raw):
            try:
                _, host, _ = validate_url(u)
                if _is_ip(host) and not ip_is_public(host):
                    continue
            except (UnsafeURL, ValueError):
                continue
            if (t, u) not in links:
                links.append((t.strip(), u))
        text = MD_LINK.sub(r"\1", body)
        text = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", text)  # картинки
        text = re.sub(r"\n{3,}", "\n\n", text).strip()
        return Page(url, title_m.group(1).strip() if title_m else "", text[: self.max_chars], links[:15],
                    len(text) > self.max_chars, "jina")
