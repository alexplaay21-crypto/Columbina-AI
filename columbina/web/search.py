"""Поиск в интернете: Tavily (с ключом), DuckDuckGo и Wikipedia (без ключей). Цепочка с запасными вариантами."""

from __future__ import annotations

import html
import logging
import re
from dataclasses import dataclass
from typing import Protocol
from urllib.parse import parse_qs, unquote, urlparse

import httpx

log = logging.getLogger(__name__)
UA = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"


@dataclass
class SearchResult:
    title: str
    url: str
    snippet: str
    content: str = ""  # расширенный текст, если поисковик его дал (Tavily)
    source: str = ""   # tavily | duckduckgo | wikipedia


class SearchProvider(Protocol):
    name: str

    async def search(self, query: str, max_results: int = 5) -> list[SearchResult]: ...


class TavilySearch:
    name = "tavily"
    URL = "https://api.tavily.com/search"

    def __init__(self, api_key: str, client: httpx.AsyncClient):
        self.api_key = api_key
        self.client = client

    async def search(self, query: str, max_results: int = 5) -> list[SearchResult]:
        r = await self.client.post(self.URL, headers={"Authorization": f"Bearer {self.api_key}"}, json={
            "query": query, "max_results": max_results, "search_depth": "basic",
            "include_answer": False, "include_raw_content": False,
        })
        r.raise_for_status()
        out = []
        for it in r.json().get("results", [])[:max_results]:
            content = (it.get("content") or "").strip()
            out.append(SearchResult(it.get("title") or it.get("url", ""), it.get("url", ""), content[:300],
                                    content[:1500], self.name))
        return out


_DDG_LINK = re.compile(r'<a[^>]+class="result__a"[^>]+href="(?P<href>[^"]+)"[^>]*>(?P<title>.*?)</a>', re.S)
_DDG_SNIP = re.compile(r'class="result__snippet"[^>]*>(?P<snip>.*?)</a>', re.S)
_TAG = re.compile(r"<[^>]+>")


def _clean(s: str | None) -> str:
    return html.unescape(_TAG.sub("", s or "")).strip()


def _ddg_url(href: str) -> str:
    href = html.unescape(href)
    if href.startswith("//"):
        href = "https:" + href
    p = urlparse(href)
    if "duckduckgo.com" in p.netloc and p.path.startswith("/l/"):
        q = parse_qs(p.query).get("uddg")
        if q:
            return unquote(q[0])
    return href


class DuckDuckGoSearch:
    name = "duckduckgo"
    URL = "https://html.duckduckgo.com/html/"

    def __init__(self, client: httpx.AsyncClient, region: str = "ru-ru"):
        self.client = client
        self.region = region

    LITE = "https://lite.duckduckgo.com/lite/"

    async def search(self, query: str, max_results: int = 5) -> list[SearchResult]:
        try:
            return await self._html(query, max_results)
        except RuntimeError:
            return await self._lite(query, max_results)

    async def _lite(self, query: str, max_results: int) -> list[SearchResult]:
        r = await self.client.post(self.LITE, data={"q": query, "kl": self.region},
                                   headers={"User-Agent": UA, "Referer": "https://lite.duckduckgo.com/"})
        r.raise_for_status()
        out = []
        for m in re.finditer(r'<a[^>]+href="([^"]+)"[^>]*class=.result-link.[^>]*>(.*?)</a>', r.text, re.S):
            url = _ddg_url(m.group(1))
            if url.startswith("http") and "duckduckgo.com" not in url:
                out.append(SearchResult(_clean(m.group(2)), url, "", "", self.name))
            if len(out) >= max_results:
                break
        if not out:
            raise RuntimeError("DuckDuckGo недоступен (капча)")
        return out

    async def _html(self, query: str, max_results: int) -> list[SearchResult]:
        r = await self.client.post(self.URL, data={"q": query, "kl": self.region},
                                   headers={"User-Agent": UA, "Referer": "https://html.duckduckgo.com/"})
        r.raise_for_status()
        out: list[SearchResult] = []
        links = list(_DDG_LINK.finditer(r.text))
        for i, m in enumerate(links):
            url = _ddg_url(m.group("href"))
            if "duckduckgo.com/y.js" in url or not url.startswith("http"):
                continue  # реклама
            block_end = links[i + 1].start() if i + 1 < len(links) else len(r.text)
            sm = _DDG_SNIP.search(r.text, m.end(), block_end)
            snip = _clean(sm.group("snip")) if sm else ""
            title = " ".join(_clean(m.group("title")).split())
            out.append(SearchResult(title, url, snip[:300], "", self.name))
            if len(out) >= max_results:
                break
        if not out and "anomaly" in r.text.lower():
            raise RuntimeError("DuckDuckGo попросил капчу")
        return out


class JinaSearch:
    """s.jina.ai — нужен бесплатный ключ jina.ai (JINA_API_KEY)."""
    name = "jina"

    def __init__(self, api_key: str, client: httpx.AsyncClient):
        self.api_key, self.client = api_key, client

    async def search(self, query: str, max_results: int = 5) -> list[SearchResult]:
        r = await self.client.get("https://s.jina.ai/", params={"q": query}, headers={
            "Authorization": f"Bearer {self.api_key}", "Accept": "application/json", "X-Respond-With": "no-content"})
        r.raise_for_status()
        return [SearchResult(it.get("title", ""), it.get("url", ""), (it.get("description") or "")[:300],
                             (it.get("content") or "")[:1500], self.name)
                for it in (r.json().get("data") or [])[:max_results] if it.get("url")]


class BraveSearch:
    """Brave Search API — бесплатный тариф ~2000 запросов в месяц (BRAVE_API_KEY)."""
    name = "brave"

    def __init__(self, api_key: str, client: httpx.AsyncClient):
        self.api_key, self.client = api_key, client

    async def search(self, query: str, max_results: int = 5) -> list[SearchResult]:
        r = await self.client.get("https://api.search.brave.com/res/v1/web/search",
                                  params={"q": query, "count": max_results},
                                  headers={"X-Subscription-Token": self.api_key, "Accept": "application/json"})
        r.raise_for_status()
        return [SearchResult(_clean(it.get("title")), it.get("url", ""), _clean(it.get("description"))[:300], "",
                             self.name) for it in (r.json().get("web", {}).get("results") or [])[:max_results]]


class SearxngSearch:
    """Свой экземпляр SearXNG (SEARXNG_URL) — бесплатный метапоиск без ключей."""
    name = "searxng"

    def __init__(self, base_url: str, client: httpx.AsyncClient):
        self.base_url, self.client = base_url.rstrip("/"), client

    async def search(self, query: str, max_results: int = 5) -> list[SearchResult]:
        r = await self.client.get(f"{self.base_url}/search", params={"q": query, "format": "json"})
        r.raise_for_status()
        return [SearchResult(it.get("title", ""), it.get("url", ""), (it.get("content") or "")[:300], "", self.name)
                for it in (r.json().get("results") or [])[:max_results] if it.get("url")]


class WikipediaSearch:
    name = "wikipedia"

    def __init__(self, client: httpx.AsyncClient, lang: str = "ru"):
        self.client = client
        self.lang = lang

    async def search(self, query: str, max_results: int = 3) -> list[SearchResult]:
        r = await self.client.get(f"https://{self.lang}.wikipedia.org/w/api.php", params={
            "action": "query", "list": "search", "srsearch": query, "format": "json",
            "srlimit": max_results, "utf8": 1,
        }, headers={"User-Agent": "PersonalTelegramBot/1.0 (+https://www.mediawiki.org/wiki/API:Etiquette) httpx"})
        r.raise_for_status()
        out = []
        for it in r.json().get("query", {}).get("search", [])[:max_results]:
            title = it["title"]
            url = f"https://{self.lang}.wikipedia.org/wiki/" + title.replace(" ", "_")
            out.append(SearchResult(title, url, _clean(it.get("snippet"))[:300], "", self.name))
        return out


class WebSearch:
    """Пробует поисковики по порядку; Wikipedia добавляется к справочным запросам как второй источник."""

    def __init__(self, providers: list[SearchProvider], wikipedia: WikipediaSearch | None = None):
        self.providers = providers
        self.wikipedia = wikipedia

    async def search(self, query: str, max_results: int = 5) -> tuple[list[SearchResult], list[str]]:
        errors: list[str] = []
        results: list[SearchResult] = []
        for p in self.providers:
            try:
                results = await p.search(query, max_results)
                if results:
                    break
            except Exception as e:  # noqa: BLE001 — любой сбой поисковика = пробуем следующий
                errors.append(f"{p.name}: {type(e).__name__}")
                log.info("search %s failed: %s", p.name, e)
        if self.wikipedia and len(results) < max_results:
            try:
                seen = {r.url for r in results}
                results += [w for w in await self.wikipedia.search(query, 2) if w.url not in seen]
            except Exception as e:  # noqa: BLE001
                errors.append(f"wikipedia: {type(e).__name__}")
        return results[: max_results + 2], errors
