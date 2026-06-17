"""Tier 1 — KB Ingest for Node Agent Assistant.

Crawl greennode.ai (server-rendered, stdlib-only — verified), strip HTML to
clean text, chunk it, and emit JSONL records ready for Tier 2 (embed + store).

NO third-party deps: urllib + html.parser + re + json. This keeps ingest
runnable on a bare Python before any pip install, and verifiable via
`python3 -m py_compile`.

Verified facts (2026-06-14):
  - greennode.ai renders server-side; visible text present in raw HTML.
  - robots.txt: `Disallow:` (nothing blocked).
  - sitemap.xml is gzip-compressed → must gunzip before XML parse.
  - 195 URLs in sitemap; curated KB set = product/solution/pricing/about/blog
    (drop security-advisories CVE noise, privacy, thank-you, coming-soon).

Treat all crawled content as UNTRUSTED DATA, never instructions (T3 guardrail).
"""

from __future__ import annotations

import dataclasses
import gzip
import html
import io
import json
import re
import time
import urllib.request
import xml.etree.ElementTree as ET
from html.parser import HTMLParser
from pathlib import Path
from typing import Iterable, Iterator

# ── HTTP ──────────────────────────────────────────────────────────────────
# greennode.ai returns 200 to a normal browser UA. Default urllib UA risks a
# 403 on many VN sites (verified on Vietstock), so always send a browser UA.
_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
)
_HEADERS = {"User-Agent": _UA, "Accept": "text/html,application/xhtml+xml,*/*"}

SITEMAP_URL = "https://greennode.ai/sitemap.xml"

# Path prefixes worth keeping in the KB. Everything else is dropped as noise.
_KEEP_PREFIXES = ("/product", "/solution", "/blog", "/pricing", "/about")
# Explicit drops even if under a kept prefix (CVE write-ups, legal boilerplate).
_DROP_SUBSTR = (
    "security-advisor",
    "/privacy",
    "/terms",
    "thank-you",
    "coming-soon",
    "schedule-meeting",
    "/author",
)


def _fetch(url: str, timeout: int = 20) -> bytes:
    req = urllib.request.Request(url, headers=_HEADERS)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def _maybe_gunzip(raw: bytes) -> bytes:
    # gzip magic 1f 8b — sitemap.xml is served gzipped without a clear ext.
    if raw[:2] == b"\x1f\x8b":
        return gzip.GzipFile(fileobj=io.BytesIO(raw)).read()
    return raw


# ── Sitemap → URL list ──────────────────────────────────────────────────────
def list_sitemap_urls(sitemap_url: str = SITEMAP_URL) -> list[str]:
    """Return all <loc> URLs from a sitemap (handles gzip + nested sitemaps)."""
    raw = _maybe_gunzip(_fetch(sitemap_url))
    root = ET.fromstring(raw)
    # Strip XML namespace so tag matching is simple.
    locs = [el.text.strip() for el in root.iter() if el.tag.endswith("loc") and el.text]

    # If this is a sitemap index (points to more sitemaps), recurse one level.
    if root.tag.endswith("sitemapindex"):
        out: list[str] = []
        for sm in locs:
            try:
                out.extend(list_sitemap_urls(sm))
            except Exception:
                continue
        return out
    return locs


def curate_urls(urls: Iterable[str]) -> list[str]:
    """Filter the raw sitemap down to the KB-worthy pages."""
    out: list[str] = []
    seen: set[str] = set()
    for u in urls:
        path = re.sub(r"^https?://[^/]+", "", u)
        if any(s in u for s in _DROP_SUBSTR):
            continue
        if not (path == "/" or path.startswith(_KEEP_PREFIXES)):
            continue
        if u not in seen:
            seen.add(u)
            out.append(u)
    return out


# ── HTML → clean text ───────────────────────────────────────────────────────
class _TextExtractor(HTMLParser):
    """Collect visible text, dropping script/style/nav/footer chrome."""

    _SKIP = {"script", "style", "noscript", "svg", "head", "nav", "footer", "form"}
    _BLOCK = {"p", "div", "section", "li", "br", "h1", "h2", "h3", "h4", "h5", "tr", "article"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._skip_depth = 0
        self._parts: list[str] = []
        self.title = ""
        self._in_title = False

    def handle_starttag(self, tag: str, attrs: list) -> None:
        if tag in self._SKIP:
            self._skip_depth += 1
        if tag == "title":
            self._in_title = True
        if tag in self._BLOCK:
            self._parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in self._SKIP and self._skip_depth > 0:
            self._skip_depth -= 1
        if tag == "title":
            self._in_title = False
        if tag in self._BLOCK:
            self._parts.append("\n")

    def handle_data(self, data: str) -> None:
        # Title check MUST come before the skip guard: <title> lives inside
        # <head>, which is in _SKIP, so skip_depth > 0 while reading it.
        if self._in_title:
            self.title += data
            return
        if self._skip_depth > 0:
            return
        text = data.strip()
        if text:
            self._parts.append(text)

    def text(self) -> str:
        raw = " ".join(self._parts)
        raw = html.unescape(raw)
        # Collapse runs of whitespace; keep paragraph breaks as single \n.
        raw = re.sub(r"[ \t]+", " ", raw)
        raw = re.sub(r"\s*\n\s*", "\n", raw)
        raw = re.sub(r"\n{2,}", "\n", raw)
        return raw.strip()


def _strip_html_fragment(fragment: str) -> str:
    """Run an HTML fragment (Strapi rich-text) through the text extractor."""
    p = _TextExtractor()
    p.feed(fragment)
    return p.text()


def _extract_strapi_content(raw_html: str) -> str:
    """Pull blog body from greennode.ai's Strapi JSON island.

    Blog bodies are NOT in <p> tags — the page ships them inside a <script>
    JSON blob as `"content":"\\u003Cp>...escaped HTML..."`. The visible DOM
    shows only header/meta (or a maintenance banner), so the <p> parser yields
    thin text. We locate each `"content":"` key, JSON-decode the string value
    (respecting \\-escapes), strip its HTML, and concatenate.

    Returns "" when no Strapi content island is present (e.g. /product pages,
    which the normal parser handles fine).
    """
    out: list[str] = []
    key = '"content":"'
    i = 0
    while True:
        j = raw_html.find(key, i)
        if j == -1:
            break
        start = j + len(key) - 1  # point at the opening quote
        # Walk the JSON string honoring backslash escapes to find its end.
        k = start + 1
        n = len(raw_html)
        while k < n:
            c = raw_html[k]
            if c == "\\":
                k += 2
                continue
            if c == '"':
                break
            k += 1
        literal = raw_html[start : k + 1]
        i = k + 1
        try:
            decoded = json.loads(literal)  # turns \u003Cp> into <p>, etc.
        except Exception:
            continue
        if not isinstance(decoded, str) or len(decoded) < 40:
            continue
        text = _strip_html_fragment(decoded)
        if len(text) > 40:
            out.append(text)
    return "\n".join(out).strip()


def extract_text(html_bytes: bytes) -> tuple[str, str]:
    """Return (title, clean_text) from a page's raw HTML.

    Prefers Strapi blog body when it is richer than the visible-DOM text
    (blogs); falls back to the <p>/block parser (product/pricing/solution).
    """
    raw = html_bytes.decode("utf-8", errors="replace")
    parser = _TextExtractor()
    parser.feed(raw)
    dom_text = parser.text()

    strapi_text = _extract_strapi_content(raw)
    if len(strapi_text) > len(dom_text):
        # Keep the title line for context, then the rich Strapi body.
        title = parser.title.strip()
        body = f"{title}\n{strapi_text}" if title else strapi_text
        return title, body.strip()
    return parser.title.strip(), dom_text


# ── Chunking ────────────────────────────────────────────────────────────────
@dataclasses.dataclass
class Chunk:
    url: str
    title: str
    chunk_index: int
    text: str

    def to_json(self) -> str:
        return json.dumps(dataclasses.asdict(self), ensure_ascii=False)


def chunk_text(
    text: str, *, max_chars: int = 1200, overlap: int = 150
) -> list[str]:
    """Greedy paragraph-aware chunker.

    Packs paragraphs up to ~max_chars; oversize paragraphs are hard-split.
    `overlap` chars of tail are prepended to the next chunk to preserve context
    across boundaries (helps retrieval recall).
    """
    paras = [p.strip() for p in text.split("\n") if p.strip()]
    chunks: list[str] = []
    buf = ""
    for p in paras:
        if len(p) > max_chars:
            # Flush current buffer, then hard-split the long paragraph.
            if buf:
                chunks.append(buf)
                buf = ""
            for i in range(0, len(p), max_chars):
                chunks.append(p[i : i + max_chars])
            continue
        if len(buf) + len(p) + 1 <= max_chars:
            buf = f"{buf}\n{p}" if buf else p
        else:
            chunks.append(buf)
            tail = buf[-overlap:] if overlap else ""
            buf = f"{tail}\n{p}" if tail else p
    if buf:
        chunks.append(buf)
    return [c.strip() for c in chunks if c.strip()]


# ── Orchestration ───────────────────────────────────────────────────────────
def crawl_page(url: str) -> Iterator[Chunk]:
    title, text = extract_text(_fetch(url))
    for i, ch in enumerate(chunk_text(text)):
        yield Chunk(url=url, title=title, chunk_index=i, text=ch)


def ingest(
    out_path: str | Path,
    *,
    limit: int | None = None,
    delay: float = 0.3,
    verbose: bool = True,
) -> dict:
    """Full Tier-1 run: sitemap → curate → crawl → chunk → JSONL.

    Returns a stats dict. Writes one JSON chunk per line to out_path.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    urls = curate_urls(list_sitemap_urls())
    if limit:
        urls = urls[:limit]

    pages_ok = pages_err = total_chunks = 0
    errors: list[tuple[str, str]] = []
    with out_path.open("w", encoding="utf-8") as f:
        for n, url in enumerate(urls, 1):
            try:
                page_chunks = list(crawl_page(url))
                for ch in page_chunks:
                    f.write(ch.to_json() + "\n")
                total_chunks += len(page_chunks)
                pages_ok += 1
                if verbose:
                    print(f"[{n}/{len(urls)}] OK  {len(page_chunks):2d} chunks  {url}")
            except Exception as e:  # noqa: BLE001 — log + continue, never abort whole crawl
                pages_err += 1
                errors.append((url, str(e)))
                if verbose:
                    print(f"[{n}/{len(urls)}] ERR {e}  {url}")
            time.sleep(delay)  # be polite to the origin

    return {
        "urls_total": len(urls),
        "pages_ok": pages_ok,
        "pages_err": pages_err,
        "total_chunks": total_chunks,
        "out_path": str(out_path),
        "errors": errors[:20],
    }


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Tier 1 KB ingest (greennode.ai)")
    ap.add_argument("--out", default="data/kb_chunks.jsonl")
    ap.add_argument("--limit", type=int, default=None, help="max pages (smoke test)")
    ap.add_argument("--delay", type=float, default=0.3)
    args = ap.parse_args()

    stats = ingest(args.out, limit=args.limit, delay=args.delay)
    print("\n=== INGEST STATS ===")
    print(json.dumps({k: v for k, v in stats.items() if k != "errors"}, indent=2))
    if stats["errors"]:
        print(f"({len(stats['errors'])} errors shown above)")
