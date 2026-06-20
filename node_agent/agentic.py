"""Tier AGENTIC — LIVE search → collect → assess → verify → (re-search) → report.

This is the agentic loop for the Node Assistant (Nhật's spec, 2026-06-14):

    question
      → SEARCH    (websearch.greennode_search — SearXNG, official-first)
      → COLLECT   (crawl each result URL → clean text via ingest.extract_text)
      → ASSESS    (score reliability/coverage of collected evidence)
      → VERIFY    (enough trustworthy evidence to answer?)
          NO  → refine the query and SEARCH again (up to MAX_ROUNDS)
          YES → REPORT (LLM grounds an answer on the live evidence, cites [n],
                        tables stay tables — enforced by the system prompt)

Philosophy: GreenNode data (pricing, MaaS models, promos) is real-time, so the
loop always hits LIVE sources. The local KB is only a warm-start cache, never a
reason to refuse. The loop only gives up AFTER an honest live attempt found
nothing — and even then it points the customer at the official source.

The search backend is pluggable (websearch.py): SearXNG when up, else fallback.
Collection + assessment are stdlib + the existing crawler, so this tier is
verifiable offline by pointing COLLECT straight at known greennode.ai URLs
(no search engine needed) — see __main__.
"""

from __future__ import annotations

import dataclasses
import re
from typing import Optional

from .ingest import _fetch, extract_text
from .websearch import SearchResult, greennode_search
from .reason import build_context_block, _sources_from_hits
from .retrieve import Hit


# How hard the loop tries before reporting what it has.
MAX_ROUNDS = 3
# Min trustworthy evidence chars to consider the question "answerable".
MIN_EVIDENCE_CHARS = 600
# Per-page crawl cap (chars) so one long page can't dominate the context.
PER_PAGE_CHARS = 2400


@dataclasses.dataclass
class Evidence:
    """One collected, crawled source backing an answer."""
    url: str
    title: str
    text: str
    engine: str = ""
    official: bool = False

    def as_hit(self) -> Hit:
        # Reuse the Hit shape so build_context_block / _sources_from_hits work.
        return Hit(score=1.0, url=self.url, title=self.title,
                   chunk_index=0, text=self.text)


@dataclasses.dataclass
class LoopTrace:
    """Audit trail of what the loop did — surfaced to the dashboard."""
    rounds: int = 0
    queries: list[str] = dataclasses.field(default_factory=list)
    collected: int = 0
    official_count: int = 0
    evidence_chars: int = 0
    verified: bool = False
    notes: list[str] = dataclasses.field(default_factory=list)


# ── COLLECT ──────────────────────────────────────────────────────────────────
def collect(results: list[SearchResult], *, max_pages: int = 5,
            max_renders: int = 2) -> list[Evidence]:
    """Crawl each search hit into clean evidence text. Official domains first.

    `max_renders` caps the number of help-center pages we Playwright-render in
    one call (each render is ~3-5s). Without the cap a wide search queued ~5
    renders → 50s+ latency. Past the cap, remaining helpdesk hits stay link-only.
    """
    ordered = sorted(results, key=lambda r: 0 if r.is_official else 1)
    out: list[Evidence] = []
    renders_done = 0
    for r in ordered[:max_pages]:
        try:
            title, text = extract_text(_fetch(r.url, timeout=15))
        except Exception:
            text, title = r.snippet, r.title
        text = (text or "").strip()
        is_helpdesk = ("helpdesk.greennode.ai" in r.url or "helpdesk.vngcloud.vn" in r.url)
        if is_helpdesk and len(text) < 200 and renders_done < max_renders:
            try:
                from .helpdesk import render_article
                h_title, h_text = render_article(r.url)
                renders_done += 1
                if len(h_text) > len(text):
                    title, text = (h_title or title), h_text
            except Exception:
                pass
        text = (text or r.snippet or "").strip()[:PER_PAGE_CHARS]
        if not text:
            continue
        out.append(Evidence(
            url=r.url, title=title or r.title, text=text,
            engine=r.engine, official=r.is_official,
        ))
    return out


# ── ASSESS ───────────────────────────────────────────────────────────────────
def assess(question: str, evidence: list[Evidence]) -> tuple[float, list[str]]:
    """Heuristic reliability/coverage score in 0..1 + human-readable notes.

    Deliberately LLM-free so it's deterministic and cheap. Signals:
      - at least one OFFICIAL source present (greennode.ai / helpdesk / vngcloud)
      - total trustworthy evidence volume (chars)
      - query-term coverage: do the collected pages actually mention the asked
        terms (so a search that drifted off-topic scores low)
    """
    notes: list[str] = []
    if not evidence:
        return 0.0, ["no evidence collected"]

    official = [e for e in evidence if e.official]
    total_chars = sum(len(e.text) for e in evidence)
    off_chars = sum(len(e.text) for e in official)

    # query-term coverage over official text (fall back to all text)
    corpus = " ".join(e.text for e in (official or evidence)).lower()
    terms = [t for t in re.findall(r"\w+", question.lower()) if len(t) > 2]
    hit = sum(1 for t in set(terms) if t in corpus)
    coverage = hit / max(1, len(set(terms)))

    score = 0.0
    if official:
        score += 0.45
        notes.append(f"{len(official)} nguồn chính thức")
    else:
        notes.append("CHƯA có nguồn chính thức")
    score += min(0.30, (off_chars or total_chars) / MIN_EVIDENCE_CHARS * 0.30)
    score += 0.25 * coverage
    notes.append(f"coverage thuật ngữ {coverage:.0%}")
    notes.append(f"{total_chars} ký tự bằng chứng")
    return min(1.0, score), notes


def refine_query(question: str, round_i: int, notes: list[str]) -> str:
    """Tighten the query for a re-search round based on what was weak."""
    base = question.strip()
    if round_i == 1:
        # push harder onto official docs / help center
        return f"{base} GreenNode bảng giá thông số chính thức"
    if round_i == 2:
        return f"{base} site:helpdesk.greennode.ai OR site:greennode.ai"
    return base


# ── KB-seeded searcher (bridge until SearXNG is up) ──────────────────────────
def make_kb_searcher(retr):
    """Build a searcher that uses the local KB as a URL INDEX, not as the answer.

    The KB knows ~111 greennode.ai URLs. BM25 picks the URLs most relevant to
    the question; the loop then crawls those URLs LIVE (fresh content at ask
    time, not the stale cached chunk). This gives a real live-first path with
    zero dependency on an external search engine. When SearXNG is up, swap this
    for websearch.greennode_search to also reach beyond greennode.ai.
    """
    def _search(query: str, limit: int = 6) -> list[SearchResult]:
        hits = retr.search(query, k=limit * 4)
        seen: set[str] = set()
        out: list[SearchResult] = []
        for h in hits:
            if h.url in seen:
                continue
            seen.add(h.url)
            out.append(SearchResult(
                title=h.title or h.url, url=h.url, snippet=h.text[:160],
                engine="kb-seed",
            ))
            if len(out) >= limit:
                break
        return out
    return _search


# ── Hybrid searcher: LIVE WEB first (SearXNG→DDG), KB only as a rescue ────────
def make_hybrid_searcher(retr):
    """Searcher that ALWAYS tries the live web first, KB only when web is empty.

    The old logic was: `greennode_search if searxng_available() else kb`. That
    was wrong — when SearXNG is down it fell straight to a KB-only search and
    NEVER touched the web, even though `greennode_search` has its own DuckDuckGo
    fallback (websearch.web_search → DDG). So a question whose answer lives in
    live docs (helpdesk.greennode.ai, vngcloud docs) but NOT in the cached blog
    KB would always come back "không có dữ liệu".

    This searcher fixes that:
      1. greennode_search(query)  → SearXNG if up, else DuckDuckGo (auto).
      2. if that returns nothing  → make_kb_searcher (cached URL index) as a
         last-resort so we degrade to "something" instead of empty-handed.
    Either way the agentic loop then CRAWLS the chosen URLs live at ask time.
    """
    kb = make_kb_searcher(retr)

    def _search(query: str, limit: int = 6) -> list[SearchResult]:
        try:
            web = greennode_search(query, limit=limit)
        except Exception:
            web = []
        if web:
            return web
        # web genuinely returned nothing (offline / blocked) → KB rescue.
        return kb(query, limit=limit)

    return _search


# ── Multi-intent split ───────────────────────────────────────────────────────
# Customer questions often bundle several asks ("cho em (1) bảng giá (2) reset
# mật khẩu (3) có hỗ trợ K8s không"). One blended query retrieves shallow
# evidence for each part (low coverage). Split into sub-questions, search each
# separately, then merge the evidence so every intent gets its own sources.
_SPLIT_RE = re.compile(
    r"\(\s*\d+\s*\)|"          # (1) (2) (3)
    r"(?:^|\s)\d+[\.\)]\s|"    # 1.  2)
    r"\bvà\b|;|\n|"            # 'và', semicolon, newline
    r"\?\s+",                  # end of one question before another
)


def split_intents(question: str, *, max_parts: int = 4) -> list[str]:
    """Break a multi-part question into sub-questions. Returns [question] if single."""
    # Keep the head (often "cho em 3 thứ:") out of the parts.
    body = question
    if ":" in question and re.search(r"\d", question.split(":", 1)[0]):
        body = question.split(":", 1)[1]
    parts = [p.strip(" .,:-–") for p in _SPLIT_RE.split(body) if p and len(p.strip()) > 8]
    # Drop near-duplicates / too-short fragments; cap.
    seen, out = set(), []
    for p in parts:
        key = p.lower()[:40]
        if key not in seen:
            seen.add(key)
            out.append(p)
    out = out[:max_parts]
    # Only treat as multi-intent if we got 2+ solid parts; else single.
    return out if len(out) >= 2 else [question]


def _gather_evidence(question: str, searcher, trace: "LoopTrace",
                     min_score: float, max_rounds: int) -> tuple[list[Evidence], float]:
    """Search→collect→assess one (sub-)question, re-searching while weak."""
    best_ev: list[Evidence] = []
    best_score = 0.0
    query = question
    for round_i in range(max_rounds):
        trace.rounds += 1
        trace.queries.append(query)
        results = searcher(query, limit=6)
        evidence = collect(results)
        score, notes = assess(question, evidence)
        if score > best_score:
            best_score, best_ev = score, evidence
        trace.notes.append(f"«{question[:30]}» vòng {round_i+1}: score={score:.2f} · " + " · ".join(notes))
        if score >= min_score:
            break
        query = refine_query(question, round_i + 1, notes)
    return best_ev, best_score


# ── Self-verify (before sending to the customer) ─────────────────────────────
def self_verify(question: str, answer: str, intents: list[str],
                provider, *, model=None) -> tuple[bool, list[str]]:
    """Ask the model to CHECK its own draft before it reaches the customer.

    Deterministic pre-checks first (cheap), then one LLM self-critique that
    returns OK or a list of concrete gaps. We only act on hard gaps:
      - an intent the answer never addressed (multi-part coverage)
      - a specific claim (price/number) with no citation near it
    Returns (ok, issues). The loop uses issues to trigger a corrective rewrite.
    """
    issues: list[str] = []
    low = answer.lower()

    # 1. Coverage: every sub-intent should leave a lexical trace in the answer.
    if len(intents) >= 2:
        for it in intents:
            kw = [w for w in re.findall(r"\w+", it.lower()) if len(w) > 3]
            if kw and sum(1 for w in set(kw) if w in low) / len(set(kw)) < 0.25:
                issues.append(f"chưa trả lời ý: «{it[:40]}»")

    # 2. Truncation heuristic: ends mid-sentence / mid-heading.
    tail = answer.rstrip()[-1:] if answer.strip() else ""
    if tail and tail not in ".!?)]}…\"'" and not answer.rstrip().endswith("```"):
        issues.append("câu trả lời có vẻ bị cắt cụt (kết thúc giữa chừng)")

    if provider is None:
        return (not issues), issues

    # 3. LLM self-critique grounded on the same evidence.
    critique_msgs = [
        {"role": "system", "content":
         "Bạn là bộ KIỂM TRA chất lượng. Đọc CÂU HỎI và CÂU TRẢ LỜI. "
         "Chỉ ra lỗi NGHIÊM TRỌNG nếu có: (a) bỏ sót ý khách hỏi, (b) số liệu/giá "
         "không kèm trích dẫn [n], (c) bịa thông tin, (d) trả lời cụt. "
         "Nếu ổn, trả về đúng 1 từ: OK. Nếu có lỗi, liệt kê ngắn gọn mỗi lỗi 1 dòng."},
        {"role": "user", "content": f"CÂU HỎI:\n{question}\n\nCÂU TRẢ LỜI:\n{answer}"},
    ]
    try:
        crit = provider.chat(critique_msgs, temperature=0.0, max_tokens=256, model=model).text.strip()
        if crit and not re.match(r"^\s*OK\b", crit, re.IGNORECASE):
            issues.append("self-critique: " + crit.replace("\n", " ")[:200])
    except Exception:
        pass
    return (not issues), issues


# ── LOOP ─────────────────────────────────────────────────────────────────────
@dataclasses.dataclass
class AgenticResult:
    answer: str
    sources: list
    model: str
    trace: LoopTrace
    verified: bool


def run_loop(
    question: str,
    provider=None,
    *,
    system_prompt: str,
    min_score: float = 0.55,
    max_rounds: int = MAX_ROUNDS,
    temperature: float = 0.2,
    model: Optional[str] = None,
    searcher=greennode_search,
    history: Optional[list] = None,
    memory_preamble: str = "",
) -> AgenticResult:
    """Full live agentic loop. `searcher` is injectable for offline tests.

    `history` (prior turns) + `memory_preamble` (recalled long-term facts) are
    threaded into the final REPORT prompt so follow-ups keep context and a
    returning customer is recognised — without loosening the citation gate.
    """
    from .reason import INSUFFICIENT_INFO
    from .quality import verify, verify_urls

    trace = LoopTrace()

    # SPLIT: bundled questions → sub-intents, each searched on its own so every
    # part gets dedicated evidence (fixes shallow coverage on multi-part asks).
    intents = split_intents(question)
    if len(intents) >= 2:
        trace.notes.append("tách %d ý: " % len(intents) + " | ".join(i[:30] for i in intents))

    # GATHER evidence per intent, then merge (dedup by url, official first).
    merged: list[Evidence] = []
    seen_urls: set[str] = set()
    scores: list[float] = []
    for sub in intents:
        ev, sc = _gather_evidence(sub, searcher, trace, min_score, max_rounds)
        scores.append(sc)
        for e in ev:
            if e.url not in seen_urls:
                seen_urls.add(e.url)
                merged.append(e)
    best_ev = sorted(merged, key=lambda e: 0 if e.official else 1)
    best_score = sum(scores) / len(scores) if scores else 0.0

    trace.collected = len(best_ev)
    trace.official_count = sum(1 for e in best_ev if e.official)
    trace.evidence_chars = sum(len(e.text) for e in best_ev)
    trace.verified = best_score >= min_score

    hits = [e.as_hit() for e in best_ev]
    sources = _sources_from_hits(hits)

    # No provider → offline mode: return assembled evidence (for verification).
    if provider is None:
        return AgenticResult(
            answer=build_context_block(hits) or "(không thu thập được bằng chứng)",
            sources=sources, model="(no-llm)", trace=trace, verified=trace.verified,
        )

    # No evidence at all → honest, helpful fallback (never a bare refusal).
    if not best_ev:
        return AgenticResult(
            answer=(
                "Em đã tra cứu trực tiếp nhưng chưa lấy được dữ liệu xác thực cho "
                "câu hỏi này. Anh/Chị vui lòng xem trực tiếp tại trang chính thức "
                "https://greennode.ai hoặc https://helpdesk.greennode.ai, hoặc liên hệ "
                "info@greennode.vn để có thông tin chính xác và mới nhất."
            ),
            sources=[], model="(no-evidence)", trace=trace, verified=False,
        )

    # REPORT — ground the answer on live evidence; reuse the citation gate.
    from .reason import build_messages
    messages = build_messages(question, hits, system_prompt,
                              history=history, memory_preamble=memory_preamble)
    # Dynamic budget: multi-part / table answers need more room (small orchestrator model truncated
    # a 3-part answer at 1024). Scale with #intents and evidence volume.
    max_tokens = 1024
    if len(intents) >= 2:
        max_tokens = min(4096, 1024 + 700 * len(intents))
    res = provider.chat(messages, temperature=temperature, model=model, max_tokens=max_tokens)

    source_urls = {s.url for s in sources}
    verdict = verify(res.text, n_sources=len(sources))
    foreign = verify_urls(res.text, source_urls)
    if not verdict.ok or foreign:
        retry = messages + [
            {"role": "assistant", "content": res.text},
            {"role": "user", "content": (
                "Câu trả lời vừa rồi sai quy tắc trích dẫn (dùng [n] ngoài phạm vi "
                "nguồn, hoặc chèn URL không có trong NGỮ CẢNH). Viết lại CHỈ dùng "
                "nguồn [1..%d], không bịa URL." % len(sources)
            )},
        ]
        res = provider.chat(retry, temperature=0.0, model=model, max_tokens=max_tokens)

    # SELF-VERIFY before sending to the customer: check coverage of every intent,
    # truncation, and an LLM self-critique. On hard gaps, do ONE corrective
    # rewrite that must fix the listed issues using only the same sources.
    ok, issues = self_verify(question, res.text, intents, provider, model=model)
    trace.notes.append("self-verify: " + ("OK" if ok else "; ".join(issues)[:200]))
    if not ok:
        fix = messages + [
            {"role": "assistant", "content": res.text},
            {"role": "user", "content": (
                "Bản trả lời trên CHƯA đạt vì: " + "; ".join(issues) + ". "
                "Viết lại HOÀN CHỈNH, trả lời ĐỦ MỌI ý khách hỏi, mỗi số liệu/giá "
                "kèm trích dẫn [n] từ nguồn đã cho, KHÔNG bịa, KHÔNG bỏ dở giữa "
                "chừng, KHÔNG xin lỗi hay viết 'viết lại'. Bảng phải giữ dạng bảng."
            )},
        ]
        try:
            res2 = provider.chat(fix, temperature=0.0, model=model, max_tokens=max_tokens)
            # accept the rewrite only if it still passes the citation gate
            v2 = verify(res2.text, n_sources=len(sources))
            if v2.ok and not verify_urls(res2.text, source_urls) and res2.text.strip():
                res = res2
                trace.notes.append("self-verify: đã viết lại bản hoàn chỉnh")
        except Exception:
            pass

    return AgenticResult(
        answer=res.text, sources=sources, model=res.model,
        trace=trace, verified=trace.verified,
    )


if __name__ == "__main__":
    import sys

    # Offline verify: bypass the (not-yet-up) search engine by stubbing the
    # searcher with known greennode.ai URLs, so COLLECT→ASSESS→loop is exercised
    # for real against live pages without needing SearXNG.
    def _stub_search(query, limit=6):
        urls = [
            ("https://greennode.ai/pricing", "Pricing | GreenNode"),
            ("https://greennode.ai/product/h100", "NVIDIA H100 | GreenNode"),
            ("https://greennode.ai/product/ai-platform", "AI Platform | GreenNode"),
        ]
        return [SearchResult(title=t, url=u, snippet="", engine="stub") for u, t in urls]

    q = " ".join(sys.argv[1:]) or "Bảng giá GPU H100 của GreenNode"
    r = run_loop(q, provider=None, system_prompt="", searcher=_stub_search)
    print(f"verified={r.verified} rounds={r.trace.rounds} "
          f"collected={r.trace.collected} official={r.trace.official_count} "
          f"chars={r.trace.evidence_chars}")
    for n in r.trace.notes:
        print("  -", n)
    print("\n--- evidence preview ---")
    print(r.answer[:600])
