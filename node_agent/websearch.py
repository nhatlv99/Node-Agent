"""Tier WS — Web search for Node Agent Assistant (LIVE-first).

Pluggable search layer. The Node Assistant is a customer-support agent for a
cloud whose data (GPU pricing, MaaS model list, promos, specs) changes in real
time, so every GreenNode question must hit a LIVE source — the local KB is only
a cache, never the source of truth.

Backends (auto-selected, in priority order):
  1. SearXNG     — self-hosted meta-search (set NODE_AGENT_SEARXNG_URL).
                   Best option: aggregates Google/Bing/DDG, returns JSON.
  2. DuckDuckGo  — HTML endpoint (no key, stdlib). Fallback when SearXNG is
                   not up yet so the pipeline still runs live end-to-end.

Both return a uniform list[SearchResult]. Swapping in SearXNG later is just an
env var — the reasoning/verify tiers never change.

Domain bias: GreenNode questions are scoped to the official sources first
(greennode.ai, helpdesk.greennode.ai, vngcloud.vn) via a `site:` preference,
then general web as backup.

stdlib-only (urllib + html.parser + json + re) — verifiable via py_compile and
a live query, no install gate.
"""

from __future__ import annotations

import dataclasses
import html
import json
import os
import re
import urllib.parse
import urllib.request
from html.parser import HTMLParser
from typing import Optional

_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
)

# Official GreenNode sources — preferred / trusted for the support domain.
OFFICIAL_DOMAINS = (
    "greennode.ai",
    "helpdesk.greennode.ai",
    "docs.vngcloud.vn",
    "vngcloud.vn",
)


@dataclasses.dataclass
class SearchResult:
    title: str
    url: str
    snippet: str
    engine: str = ""

    @property
    def is_official(self) -> bool:
        return any(d in self.url for d in OFFICIAL_DOMAINS)


def _fetch(url: str, *, timeout: int = 15, data: bytes | None = None,
           headers: dict | None = None) -> str:
    h = {"User-Agent": _UA, "Accept": "text/html,application/json,*/*"}
    if headers:
        h.update(headers)
    req = urllib.request.Request(url, data=data, headers=h,
                                 method="POST" if data else "GET")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read().decode("utf-8", errors="replace")


# ── Backend 1: SearXNG ───────────────────────────────────────────────────────
def _searxng_url() -> str:
    return os.environ.get("NODE_AGENT_SEARXNG_URL", "").rstrip("/")


def searxng_available() -> bool:
    base = _searxng_url()
    if not base:
        return False
    try:
        _fetch(f"{base}/healthz", timeout=4)
        return True
    except Exception:
        try:
            _fetch(base, timeout=4)
            return True
        except Exception:
            return False


def _search_searxng(query: str, limit: int) -> list[SearchResult]:
    base = _searxng_url()
    qs = urllib.parse.urlencode({"q": query, "format": "json", "language": "vi"})
    raw = _fetch(f"{base}/search?{qs}", timeout=20)
    data = json.loads(raw)
    out: list[SearchResult] = []
    for r in data.get("results", [])[: limit * 2]:
        out.append(SearchResult(
            title=(r.get("title") or "").strip(),
            url=(r.get("url") or "").strip(),
            snippet=(r.get("content") or "").strip(),
            engine="searxng:" + ",".join(r.get("engines", []) or []),
        ))
    return out


# ── Backend 2: DuckDuckGo HTML (fallback, no key) ────────────────────────────
class _DDGParser(HTMLParser):
    """Parse DuckDuckGo HTML endpoint result rows."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.results: list[SearchResult] = []
        self._cur_href: Optional[str] = None
        self._mode: Optional[str] = None  # 'title' | 'snippet'
        self._title = ""
        self._snippet = ""

    def handle_starttag(self, tag: str, attrs: list) -> None:
        a = dict(attrs)
        cls = a.get("class", "")
        if tag == "a" and "result__a" in cls:
            self._mode = "title"
            self._cur_href = a.get("href")
            self._title = ""
        elif tag == "a" and "result__snippet" in cls:
            self._mode = "snippet"
            self._snippet = ""
        elif tag in ("td", "div") and "result__snippet" in cls:
            self._mode = "snippet"
            self._snippet = ""

    def handle_endtag(self, tag: str) -> None:
        if tag == "a" and self._mode == "title":
            self._mode = None
        elif self._mode == "snippet" and tag in ("a", "td", "div"):
            # close out a result once we have a snippet
            if self._cur_href:
                self.results.append(SearchResult(
                    title=self._title.strip(),
                    url=_ddg_clean(self._cur_href),
                    snippet=self._snippet.strip(),
                    engine="duckduckgo",
                ))
                self._cur_href = None
            self._mode = None

    def handle_data(self, data: str) -> None:
        if self._mode == "title":
            self._title += data
        elif self._mode == "snippet":
            self._snippet += data


def _ddg_clean(href: str) -> str:
    """DDG wraps targets as /l/?uddg=<encoded>. Unwrap to the real URL."""
    if "uddg=" in href:
        q = urllib.parse.urlparse(href).query
        params = urllib.parse.parse_qs(q)
        if "uddg" in params:
            return urllib.parse.unquote(params["uddg"][0])
    if href.startswith("//"):
        return "https:" + href
    return href


_DDG_ENDPOINTS = (
    "https://html.duckduckgo.com/html/",
    "https://lite.duckduckgo.com/lite/",  # lighter mirror, different rate-limit bucket
)


def _search_ddg(query: str, limit: int) -> list[SearchResult]:
    """Query DuckDuckGo HTML. Retries across endpoints with a short backoff
    because the free HTML endpoint rate-limits intermittently (returns an empty
    body) — a single attempt made the agent loop randomly starve. We try the
    main endpoint then the lite mirror, with a small delay, before giving up."""
    import time as _t

    last_results: list[SearchResult] = []
    for attempt, endpoint in enumerate(_DDG_ENDPOINTS):
        try:
            data = urllib.parse.urlencode({"q": query}).encode()
            raw = _fetch(endpoint, data=data, timeout=20)
        except Exception:
            _t.sleep(0.6)
            continue
        p = _DDGParser()
        p.feed(raw)
        seen, out = set(), []
        for r in p.results:
            if r.url and r.url not in seen and r.title:
                seen.add(r.url)
                out.append(r)
            if len(out) >= limit * 2:
                break
        if out:
            return out
        last_results = out
        _t.sleep(0.6)  # rate-limited empty body → brief pause then next endpoint
    return last_results


# ── Public API ───────────────────────────────────────────────────────────────
def web_search(query: str, *, limit: int = 6, official_first: bool = True
               ) -> list[SearchResult]:
    """Run a live search via the best available backend.

    `official_first` re-ranks GreenNode official-domain hits to the top so the
    support agent trusts greennode.ai / helpdesk over third-party blogs.
    """
    backend = "searxng" if searxng_available() else "ddg"
    try:
        results = _search_searxng(query, limit) if backend == "searxng" \
            else _search_ddg(query, limit)
    except Exception:
        # last-ditch: try the other backend
        try:
            results = _search_ddg(query, limit)
        except Exception:
            return []

    if official_first:
        results.sort(key=lambda r: 0 if r.is_official else 1)
    # dedup by url, trim to limit
    seen, out = set(), []
    for r in results:
        if r.url not in seen:
            seen.add(r.url)
            out.append(r)
        if len(out) >= limit:
            break
    return out


# Official entry points the agent can always fall back to crawling when the
# search backend returns nothing (DDG rate-limited). These are KB landing /
# search pages on the official help centers — the agentic loop can fetch_url a
# query-targeted search URL and follow links, instead of starving on empty.
_OFFICIAL_SEEDS = (
    ("GreenNode/VNG Cloud Help Center", "https://helpdesk.vngcloud.vn/portal/en/kb"),
    ("VNG Cloud Docs", "https://docs.vngcloud.vn/"),
    ("GreenNode Help Center", "https://helpdesk.greennode.ai/portal/en/kb"),
)


def _seed_results(query: str, limit: int) -> list[SearchResult]:
    """Last-resort official entry points when live search returns nothing.

    Includes a query-targeted help-center search URL first (so a single
    fetch_url lands on relevant articles even with no search backend), then the
    KB roots. Every URL is on an official domain so the support agent stays
    grounded. The agentic loop then crawls/renders these live.
    """
    q = urllib.parse.quote(query)
    seeds = [
        SearchResult(
            title=f"Tìm trong Help Center: {query[:50]}",
            url=f"https://helpdesk.vngcloud.vn/portal/en/kb/search/{q}",
            snippet="", engine="seed-search",
        )
    ]
    for title, url in _OFFICIAL_SEEDS:
        seeds.append(SearchResult(title=title, url=url, snippet="", engine="seed"))
    return seeds[:limit]


def greennode_search(query: str, *, limit: int = 6) -> list[SearchResult]:
    """GreenNode-scoped search: bias the query toward official sources.

    Runs a site-restricted pass first, then a general pass, and merges
    (official hits first). This is what the Node Assistant calls per question.

    If BOTH passes come back empty (search backend down / rate-limited), we
    return official KB entry points (_seed_results) so the agentic loop always
    has an official page to crawl/render instead of giving up empty-handed.
    """
    site_q = f"{query} (site:greennode.ai OR site:helpdesk.greennode.ai OR site:helpdesk.vngcloud.vn OR site:docs.vngcloud.vn OR site:vngcloud.vn)"
    official = web_search(site_q, limit=limit, official_first=True)
    if len(official) >= limit:
        return official[:limit]
    general = web_search(query, limit=limit, official_first=True)
    seen = {r.url for r in official}
    merged = official + [r for r in general if r.url not in seen]
    if not merged:
        # search backend gave us nothing → seed official entry points to crawl.
        return _seed_results(query, limit)
    return merged[:limit]


if __name__ == "__main__":
    import sys

    q = " ".join(sys.argv[1:]) or "GreenNode GPU H100 pricing"
    print(f"SearXNG available: {searxng_available()} (url={_searxng_url() or '—'})")
    print(f"Query: {q}\n")
    for i, r in enumerate(greennode_search(q, limit=6), 1):
        flag = "★" if r.is_official else " "
        print(f"{flag}[{i}] {r.title[:70]}")
        print(f"     {r.url}")
        print(f"     {r.snippet[:120]}  ({r.engine})\n")
