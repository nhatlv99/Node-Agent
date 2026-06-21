"""Tier 2 — Retrieval for Node Agent Assistant.

Pure-Python BM25 over the Tier-1 JSONL chunks. Zero heavy deps (no numpy, no
torch, no sqlite-vec) so it runs on a bare interpreter and is verifiable with
`python3 -m py_compile` + a live query. Embeddings (multilingual-e5) are a
drop-in upgrade later: keep the `Retriever.search(query, k)` signature stable
and swap the scorer behind it.

BM25 was chosen over embeddings for the MVP because:
  - The KB is small (~111 pages / a few hundred chunks) — lexical recall is fine.
  - GreenNode product/pricing queries are keyword-heavy (H100, VKS, MaaS, IDP,
    "bảng giá", "GPU") where exact-term matching is a strength, not a weakness.
  - No install gate, no model download, instant verify.

Tokenizer is Unicode-aware (keeps Vietnamese diacritics) and lowercases. It is
deliberately simple: word chars across scripts via \\w with re.UNICODE.
"""

from __future__ import annotations

import dataclasses
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Iterable

_TOKEN_RE = re.compile(r"\w+", re.UNICODE)
# Detects a real money figure (used to surface rate tables for pricing intent).
_PRICE_RE = re.compile(r"\$\s?\d|\d[\d.,]*\s*(?:VND|VNĐ|đồng|đ|USD|/giờ|/hour|/h\b|/tháng|/month)", re.I)


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


# Query-side stopwords. NOT removed from doc indexing (BM25 idf already
# discounts ubiquitous terms there) — only stripped from the QUERY so VN
# function words ("có", "không", "thế nào") and the omnipresent brand token
# ("greennode") don't dilute the real content terms. Tuned for VN+EN queries.
_STOPWORDS: frozenset[str] = frozenset(
    """
    có không là gì thế nào ra sao được khi mà của và với cho về như thì
    bao nhiêu ở trong trên hỗ trợ giúp tôi mình cần muốn xin hãy vui lòng
    này đó các những một hay hoặc nên cùng theo từ đến tại sẽ đang đã
    a an the is are do does what how why when which to of for on in with
    can could please help me my i you greennode node
    """.split()
)


def filter_stopwords(terms: list[str]) -> list[str]:
    """Drop query stopwords, but never return empty (keep all if all stop)."""
    kept = [t for t in terms if t not in _STOPWORDS and len(t) > 1]
    return kept or terms


# ── Bilingual query expansion ────────────────────────────────────────────────
# BM25 is purely lexical, so a Vietnamese query ("hóa đơn", "bảng giá") never
# matches English KB chunks ("invoice", "pricing") even when the right chunk
# exists. Until e5 semantic embeddings land (the proper fix, needs torch), we
# bridge the gap with a domain term map: each query token that hits a key adds
# its cross-lingual / synonym variants to the bag of query terms. Tuned for the
# GreenNode KB vocabulary. Keys/values are lowercased single tokens.
_EXPANSIONS: dict[str, tuple[str, ...]] = {
    # Vietnamese → English (and back)
    "hóa": ("invoice", "invoices", "billing"),
    "đơn": ("invoice", "invoices", "order"),
    "bảng": ("pricing", "price", "cost"),
    "giá": ("pricing", "price", "cost", "rate"),
    "thuê": ("rent", "pricing", "ondemand"),
    "tài": ("document", "documents", "file"),
    "liệu": ("document", "documents", "file"),
    "lưu": ("storage", "store", "backup"),
    "trữ": ("storage", "store", "backup"),
    "máy": ("server", "instance", "vm"),
    "chủ": ("server", "instance", "host"),
    "mạng": ("network", "networking"),
    "bảo": ("security", "secure", "backup"),
    "mật": ("security", "secure", "encryption"),
    "huấn": ("training", "train"),
    "luyện": ("training", "train", "finetune"),
    "suy": ("inference", "serving"),
    "luận": ("inference", "serving"),
    "triển": ("deploy", "deployment"),
    "khai": ("deploy", "deployment"),
    "tốc": ("performance", "speed", "fast"),
    "độ": ("performance", "speed", "latency"),
    "doanh": ("enterprise", "business"),
    "nghiệp": ("enterprise", "business", "company"),
    # Acronym / brand normalisation (bidirectional)
    "vks": ("kubernetes", "k8s", "managed"),
    "kubernetes": ("vks", "k8s", "container"),
    "k8s": ("kubernetes", "vks"),
    "maas": ("model", "service", "models"),
    "idp": ("document", "ocr", "invoice", "processing"),
    "ocr": ("idp", "document", "text"),
    "gpu": ("h100", "h200", "nvidia", "compute"),
    "h100": ("gpu", "h200", "nvidia", "hgx"),
    "h200": ("gpu", "h100", "nvidia"),
    "finetune": ("training", "finetuning", "train"),
    "rag": ("retrieval", "embedding", "knowledge"),
    "vm": ("instance", "server", "compute"),
    # Portfolio / overview bridges: broad "có những dịch vụ/sản phẩm gì" questions
    # use abstract VN words that never appear on the (English) product pages. Map
    # them to the concrete product-line tokens so BM25 can surface the catalog
    # pages (gpu-compute, cpu-instances, object-storage, ai-platform, maas, idp...).
    "dịch": ("service", "services", "platform", "cloud", "gpu", "storage"),
    "vụ": ("service", "services", "platform"),
    "sản": ("product", "platform", "gpu", "cpu", "storage", "kubernetes"),
    "phẩm": ("product", "products", "platform"),
    "portfolio": ("product", "platform", "gpu", "cpu", "storage", "kubernetes", "model", "document"),
    "danh": ("product", "catalog", "list"),
    "mục": ("product", "catalog"),
    "nhóm": ("product", "category", "platform"),
    "tổng": ("overview", "platform", "product"),
    "quan": ("overview", "platform"),
    # Common English synonyms
    "price": ("pricing", "cost", "rate"),
    "pricing": ("price", "cost", "rate"),
    "invoice": ("idp", "billing", "document"),
    "deploy": ("deployment", "provision"),
    "train": ("training", "finetune"),
    "storage": ("object", "block", "backup"),
}


def expand_terms(q_terms: list[str]) -> list[str]:
    """Augment query terms with domain bilingual variants (deduped)."""
    out: list[str] = list(q_terms)
    seen = set(q_terms)
    for t in q_terms:
        for variant in _EXPANSIONS.get(t, ()):  # noqa: B007
            if variant not in seen:
                seen.add(variant)
                out.append(variant)
    return out


@dataclasses.dataclass
class Doc:
    url: str
    title: str
    chunk_index: int
    text: str
    tokens: list[str] = dataclasses.field(default_factory=list, repr=False)


@dataclasses.dataclass
class Hit:
    score: float
    url: str
    title: str
    chunk_index: int
    text: str


class BM25Retriever:
    """Classic Okapi BM25 (k1=1.5, b=0.75) over in-memory chunks."""

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.docs: list[Doc] = []
        self.df: Counter[str] = Counter()
        self.idf: dict[str, float] = {}
        self.avgdl: float = 0.0

    # ── build ──────────────────────────────────────────────────────────────
    def add(self, docs: Iterable[Doc]) -> None:
        for d in docs:
            d.tokens = tokenize(d.text)
            self.docs.append(d)

    def finalize(self) -> "BM25Retriever":
        n = len(self.docs)
        if n == 0:
            raise ValueError("BM25Retriever: no documents added")
        self.df.clear()
        total_len = 0
        for d in self.docs:
            total_len += len(d.tokens)
            for term in set(d.tokens):
                self.df[term] += 1
        self.avgdl = total_len / n
        # BM25 idf with +0.5 smoothing; floor at a small positive so common
        # terms (present in >half the docs) never go negative and cancel hits.
        self.idf = {
            term: max(0.05, math.log((n - df + 0.5) / (df + 0.5) + 1.0))
            for term, df in self.df.items()
        }
        return self

    # ── query ────────────────────────────────────────────────────────────────
    def search(self, query: str, k: int = 5, *, expand: bool = True, diversify: bool = False, prefer_price: bool = False) -> list[Hit]:
        base = filter_stopwords(tokenize(query))
        # Original terms get full weight; bilingual/synonym expansions get a
        # down-weight so they boost recall without overpowering exact matches.
        weights: dict[str, float] = {t: 1.0 for t in base}
        if expand:
            for variant in expand_terms(base):
                weights.setdefault(variant, 0.4)
        hits: list[Hit] = []
        for d in self.docs:
            score = self._score(weights, d)
            if score > 0:
                # Authoritative-source boost: greennode.ai /product/ and /solution/
                # pages are the canonical catalog; blog/SEO pages (95% of the corpus)
                # otherwise dominate BM25 on broad "portfolio/overview" queries and
                # bury the real product pages. Multiply official pages up so the
                # writer sees the actual product groups, not marketing articles.
                u = d.url
                if "/product/" in u or "/solution" in u:
                    score *= 2.2
                # Pricing intent: the chunks that actually carry per-hour/per-month
                # rates ($2.99, /giờ, VND) are dense tables of numbers with few
                # keywords, so BM25 ranks the wordy /product/h100 marketing page
                # (no price) above the /pricing rate table. When the question is a
                # pricing one, boost chunks that contain a real currency-number so
                # the writer gets actual figures instead of an empty price column.
                if prefer_price and _PRICE_RE.search(d.text):
                    score *= 3.0
                hits.append(
                    Hit(score, d.url, d.title, d.chunk_index, d.text)
                )
        hits.sort(key=lambda h: h.score, reverse=True)
        # Per-URL diversity cap: keep at most MAX_PER_URL chunks from any single
        # page so a broad "what services" question spreads across distinct product
        # pages (gpu / cpu / storage / maas / idp...) instead of returning 6 chunks
        # of the same ai-platform page. Overflow chunks are kept as a tail so we can
        # still backfill to k when the corpus lacks enough distinct sources.
        # diversify=True (broad "what services/portfolio" questions) caps each page
        # to 1 chunk so the top-k spreads across ALL product pages; otherwise 2.
        MAX_PER_URL = 1 if diversify else 2
        per: Counter[str] = Counter()
        primary: list[Hit] = []
        overflow: list[Hit] = []
        for h in hits:
            if per[h.url] < MAX_PER_URL:
                per[h.url] += 1
                primary.append(h)
            else:
                overflow.append(h)
        return (primary + overflow)[:k]

    def _score(self, q_weights: dict[str, float], d: Doc) -> float:
        if not d.tokens:
            return 0.0
        tf = Counter(d.tokens)
        dl = len(d.tokens)
        score = 0.0
        for term, w in q_weights.items():
            f = tf.get(term, 0)
            if f == 0:
                continue
            idf = self.idf.get(term, 0.0)
            denom = f + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
            score += w * idf * (f * (self.k1 + 1)) / denom
        return score


# ── loading ──────────────────────────────────────────────────────────────────
def load_chunks(jsonl_path: str | Path) -> list[Doc]:
    # Dedupe on (url, normalised-text). The corpus carries ~700 exact-duplicate
    # chunks (26% of 2716); without this the BM25 top-k returns the SAME chunk
    # twice, wasting evidence slots and starving the writer of distinct product
    # groups on broad "portfolio/overview" questions.
    docs: list[Doc] = []
    seen: set[tuple[str, str]] = set()
    dropped = 0
    with Path(jsonl_path).open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            text = r["text"]
            key = (r["url"], " ".join(text.split()))
            if key in seen:
                dropped += 1
                continue
            seen.add(key)
            docs.append(
                Doc(
                    url=r["url"],
                    title=r.get("title", ""),
                    chunk_index=r.get("chunk_index", 0),
                    text=text,
                )
            )
    if dropped:
        print(f"[retrieve] deduped {dropped} duplicate chunks -> {len(docs)} unique")
    return docs


def build_retriever(jsonl_path: str | Path) -> BM25Retriever:
    r = BM25Retriever()
    r.add(load_chunks(jsonl_path))
    return r.finalize()


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Tier 2 BM25 retrieval test")
    ap.add_argument("--kb", default="data/kb_chunks.jsonl")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("query", nargs="+")
    args = ap.parse_args()

    retr = build_retriever(args.kb)
    print(f"KB: {len(retr.docs)} chunks, avgdl={retr.avgdl:.1f}\n")
    for h in retr.search(" ".join(args.query), k=args.k):
        snippet = h.text[:160].replace("\n", " ")
        print(f"[{h.score:5.2f}] {h.title[:50]}")
        print(f"        {h.url}")
        print(f"        {snippet}\n")
