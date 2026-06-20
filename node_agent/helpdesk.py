"""Tier HELPDESK — render & extract Zoho Desk help-center articles.

helpdesk.greennode.ai is a Zoho Desk portal: a React SPA whose article bodies
are rendered CLIENT-SIDE. Verified (2026-06-14):
  - article HTML is an empty shell (visible text ~0 chars) for plain crawlers
  - the public REST API (/api/v1/articles, /api/v1/articles/{id}) returns 401
    for every guest auth shape we tried (orgId, portalId, cookie jar, CSRF)
So neither stdlib crawl nor the API can read content. The ONLY robust path is
to render the page in a real browser and read the hydrated DOM.

This module uses Playwright (Chromium) when available. It degrades honestly:
  - playwright not installed  -> render_article() returns ("", "") and
    is_available() is False, so the agentic loop keeps the article as a
    LINK-ONLY source (cite the official URL) instead of crashing.

INSTALL (user runs these — terminal is gatekept; agent writes, user installs):
    cd "/mnt/e/Node Agent Src"
    uv pip install playwright
    .venv/bin/python -m playwright install chromium

Then the loop auto-detects Playwright and starts rendering helpdesk articles.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


def is_available() -> bool:
    """True if Playwright + a browser are importable in this interpreter."""
    try:
        import playwright  # noqa: F401
        return True
    except Exception:
        return False


@dataclass
class Article:
    url: str
    title: str
    text: str


# Article URLs look like /portal/<lang>/kb/articles/<slug>. The KB landing and
# category pages list them; an article page renders the body into a known
# Zoho container once JS runs.
_ARTICLE_RE = re.compile(r"/portal/[a-z-]+/kb/articles/[^\"'\s]+")
_BODY_SELECTORS = (
    "div.article-content",      # Zoho elegant theme body
    "div#articleContent",
    "article",
    "main",
)


def _render(url: str, *, timeout_ms: int = 12000, wait_selector: str | None = None) -> str:
    """Return rendered HTML of a page via headless Chromium.

    Uses `domcontentloaded` (NOT `networkidle`): Zoho-Desk SPAs run analytics /
    polling that never let the network go idle, so `networkidle` always waited
    out the full timeout (~20s/page → 58s for a multi-page gather). We instead
    load the DOM, then wait briefly for the article container to hydrate, which
    cuts a render from ~20s to ~3-5s.
    """
    from playwright.sync_api import sync_playwright

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        try:
            page = browser.new_page(
                user_agent=(
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                    "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
                )
            )
            page.goto(url, wait_until="domcontentloaded", timeout=timeout_ms)
            if wait_selector:
                try:
                    page.wait_for_selector(wait_selector, timeout=6000)
                except Exception:
                    pass
            else:
                page.wait_for_timeout(1200)  # brief settle for JS hydration
            return page.content()
        finally:
            browser.close()


# Process-lifetime cache: rendering the same article twice in one session (the
# loop + a critique re-fetch) is wasteful. Cache by URL → (title, text).
_RENDER_CACHE: dict[str, tuple[str, str]] = {}


def render_article(url: str) -> tuple[str, str]:
    """Render one Zoho article → (title, clean_text). ('','') if unavailable."""
    if not is_available():
        return "", ""
    if url in _RENDER_CACHE:
        return _RENDER_CACHE[url]
    try:
        html = _render(url, wait_selector="div.article-content, article")
    except Exception:
        return "", ""
    from .ingest import extract_text
    title, text = extract_text(html.encode("utf-8"))
    _RENDER_CACHE[url] = (title, text)
    return title, text


# Cache of search/landing page → list of article URLs found on it.
_LINKS_CACHE: dict[str, list[str]] = {}


def find_article_links(url: str, limit: int = 8) -> list[str]:
    """Render a help-center search/landing page and return the ARTICLE urls it
    links to (from the live DOM <a href>, which extract_text strips away).

    This is what lets the agent drill from a search-results page into the actual
    article: the rendered text has no hrefs, so we read anchors off the DOM.
    """
    if not is_available():
        return []
    if url in _LINKS_CACHE:
        return _LINKS_CACHE[url]
    from playwright.sync_api import sync_playwright
    out: list[str] = []
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            try:
                page = browser.new_page()
                page.goto(url, wait_until="domcontentloaded", timeout=12000)
                try:
                    page.wait_for_selector("a[href*='/kb/articles/']", timeout=6000)
                except Exception:
                    page.wait_for_timeout(1500)
                hrefs = page.eval_on_selector_all(
                    "a[href*='/kb/articles/']",
                    "els => els.map(e => e.href)",
                )
            finally:
                browser.close()
        seen = set()
        for h in hrefs or []:
            if h and h not in seen:
                seen.add(h)
                out.append(h)
            if len(out) >= limit:
                break
    except Exception:
        out = []
    _LINKS_CACHE[url] = out
    return out


def list_article_urls(
    portal_home: str = "https://helpdesk.greennode.ai/portal/en/kb",
) -> list[str]:
    """Discover article URLs by rendering the KB landing/category pages."""
    if not is_available():
        return []
    try:
        html = _render(portal_home, wait_selector="a[href*='/kb/articles/']")
    except Exception:
        return []
    base = "https://helpdesk.greennode.ai"
    urls, seen = [], set()
    for m in _ARTICLE_RE.findall(html):
        full = base + m if m.startswith("/") else m
        if full not in seen:
            seen.add(full)
            urls.append(full)
    return urls


if __name__ == "__main__":
    import sys

    print(f"Playwright available: {is_available()}")
    if not is_available():
        print("Install: uv pip install playwright && python -m playwright install chromium")
        sys.exit(0)
    test = sys.argv[1] if len(sys.argv) > 1 else (
        "https://helpdesk.greennode.ai/portal/en/kb/articles/"
        "c%C3%A1c-b%C6%B0%E1%BB%9Bc-thay-%C4%91%E1%BB%95i-t%C3%A0i-kho%E1%BA%A3n-qu%E1%BA%A3n-tr%E1%BB%8B"
    )
    t, body = render_article(test)
    print(f"title: {t}\nchars: {len(body)}\n---\n{body[:600]}")
