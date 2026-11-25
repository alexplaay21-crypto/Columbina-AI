"""Безопасное получение содержимого ссылок (защита от SSRF).

- только http/https, только порты 80/443 (и явный 8080/8443 не разрешаем);
- DNS-резолв и проверка каждого IP: запрещены private, loopback, link-local (169.254.x — метаданные облака),
  multicast, reserved, unspecified, CGNAT (100.64/10), IPv6 ULA/site-local и IPv4-mapped варианты;
- соединение идёт на проверенный IP (защита от DNS rebinding), Host/SNI — исходное имя;
- редиректы обрабатываются вручную, каждый адрес проверяется заново (максимум 3);
- лимиты: размер ответа, время, типы содержимого (text/html, text/plain, application/json, xhtml).
"""

from __future__ import annotations

import asyncio
import ipaddress
import re
import socket
from dataclasses import dataclass
from html.parser import HTMLParser
from urllib.parse import urljoin, urlsplit

import httpx

URL_RE = re.compile(r"https?://[^\s<>\"'«»]+", re.IGNORECASE)
ALLOWED_PORTS = {80, 443}
ALLOWED_TYPES = ("text/html", "text/plain", "application/json", "application/xhtml+xml")
MAX_BYTES = 1_500_000
MAX_REDIRECTS = 3
BLOCKED_HOSTS = {"localhost", "localhost.localdomain", "metadata.google.internal", "metadata"}
CGNAT = ipaddress.ip_network("100.64.0.0/10")


class UnsafeURL(Exception):
    pass


def extract_urls(text: str, limit: int = 2) -> list[str]:
    urls = [u.rstrip(".,;:!?)»") for u in URL_RE.findall(text or "")]
    seen: list[str] = []
    for u in urls:
        if u not in seen:
            seen.append(u)
    return seen[:limit]


def ip_is_public(ip: str) -> bool:
    addr = ipaddress.ip_address(ip)
    if isinstance(addr, ipaddress.IPv6Address) and addr.ipv4_mapped:
        addr = addr.ipv4_mapped
    if isinstance(addr, ipaddress.IPv6Address) and addr.sixtofour:
        addr = addr.sixtofour
    if (
        addr.is_private or addr.is_loopback or addr.is_link_local or addr.is_multicast
        or addr.is_reserved or addr.is_unspecified
    ):
        return False
    if isinstance(addr, ipaddress.IPv4Address) and addr in CGNAT:
        return False
    return addr.is_global


async def resolve_public(host: str, port: int) -> str:
    """Возвращает первый IP хоста, если ВСЕ его адреса публичные; иначе UnsafeURL."""
    if host.lower().rstrip(".") in BLOCKED_HOSTS or host.lower().endswith((".local", ".internal", ".localhost")):
        raise UnsafeURL("blocked host")
    try:
        ipaddress.ip_address(host)
        ips = [host]
    except ValueError:
        loop = asyncio.get_running_loop()
        try:
            infos = await loop.getaddrinfo(host, port, type=socket.SOCK_STREAM)
        except socket.gaierror as e:
            raise UnsafeURL("dns failure") from e
        ips = sorted({i[4][0] for i in infos})
    if not ips or not all(ip_is_public(ip) for ip in ips):
        raise UnsafeURL("non-public address")
    return ips[0]


def validate_url(url: str) -> tuple[str, str, int]:
    parts = urlsplit(url)
    if parts.scheme not in ("http", "https"):
        raise UnsafeURL("scheme")
    if parts.username or parts.password:
        raise UnsafeURL("credentials in url")
    host = parts.hostname
    if not host:
        raise UnsafeURL("no host")
    port = parts.port or (443 if parts.scheme == "https" else 80)
    if port not in ALLOWED_PORTS:
        raise UnsafeURL("port")
    return parts.scheme, host, port


class _TextExtractor(HTMLParser):
    SKIP = {"script", "style", "noscript", "svg", "nav", "footer", "header", "form", "aside", "iframe"}

    def __init__(self):
        super().__init__()
        self.out: list[str] = []
        self.title = ""
        self._skip = 0
        self._in_title = False

    def handle_starttag(self, tag, attrs):
        if tag in self.SKIP:
            self._skip += 1
        if tag == "title":
            self._in_title = True
        if tag in ("p", "br", "div", "li", "h1", "h2", "h3", "h4", "tr"):
            self.out.append("\n")

    def handle_endtag(self, tag):
        if tag in self.SKIP and self._skip:
            self._skip -= 1
        if tag == "title":
            self._in_title = False

    def handle_data(self, data):
        if self._in_title:
            self.title += data
        elif not self._skip:
            self.out.append(data)


def html_to_text(html: str) -> tuple[str, str]:
    p = _TextExtractor()
    try:
        p.feed(html)
    except Exception:
        pass
    text = re.sub(r"[ \t\r\f\v]+", " ", "".join(p.out))
    text = re.sub(r"\n\s*\n+", "\n\n", text).strip()
    return p.title.strip(), text


@dataclass
class FetchedPage:
    url: str
    title: str
    text: str
    truncated: bool


class SafeFetcher:
    def __init__(self, timeout: float = 10.0, max_chars: int = 6000, verify: bool | str = True):
        self.timeout = timeout
        self.max_chars = max_chars
        self.verify = verify  # путь к CA-бандлу, если сервер за корпоративным прокси

    async def fetch(self, url: str) -> FetchedPage:
        current = url
        for _ in range(MAX_REDIRECTS + 1):
            scheme, host, port = validate_url(current)
            ip = await resolve_public(host, port)
            # Подключаемся к проверенному IP, сохраняя Host и SNI.
            target = httpx.URL(current).copy_with(host=ip)
            ext = {"sni_hostname": host} if scheme == "https" else {}
            async with httpx.AsyncClient(timeout=self.timeout, follow_redirects=False, trust_env=False,
                                         verify=self.verify) as client:
                req = client.build_request(
                    "GET", target,
                    headers={"Host": host, "User-Agent": "Mozilla/5.0 (compatible; LinkReader/1.0)",
                             "Accept": "text/html,text/plain;q=0.9,*/*;q=0.1"},
                    extensions=ext,
                )
                resp = await client.send(req, stream=True)
                try:
                    if resp.status_code in (301, 302, 303, 307, 308):
                        loc = resp.headers.get("location")
                        if not loc:
                            raise UnsafeURL("redirect without location")
                        current = urljoin(current, loc)
                        continue
                    if resp.status_code >= 400:
                        raise UnsafeURL(f"http {resp.status_code}")
                    ctype = resp.headers.get("content-type", "").split(";")[0].strip().lower()
                    if ctype and not ctype.startswith(ALLOWED_TYPES):
                        raise UnsafeURL(f"content-type {ctype}")
                    chunks: list[bytes] = []
                    size = 0
                    async for chunk in resp.aiter_bytes():
                        size += len(chunk)
                        if size > MAX_BYTES:
                            break
                        chunks.append(chunk)
                    body = b"".join(chunks)
                    enc = resp.encoding or "utf-8"
                finally:
                    await resp.aclose()
            raw = body.decode(enc, errors="replace")
            if ctype in ("text/html", "application/xhtml+xml") or "<html" in raw[:500].lower():
                title, text = html_to_text(raw)
            else:
                title, text = "", raw
            truncated = len(text) > self.max_chars
            return FetchedPage(url=current, title=title[:200], text=text[: self.max_chars], truncated=truncated)
        raise UnsafeURL("too many redirects")
