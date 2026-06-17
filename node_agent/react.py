"""Tier 3.6 — AGENTIC ReAct LOOP for Node Agent Assistant.

The old pipeline was *fixed*: code chose the searcher, THINK ran once, queries
were refined by a hardcoded heuristic string. That is NOT an agent — the model
never decided what to look up. This module makes the THINKER seat drive a real
REASON → ACT → OBSERVE → REFLECT loop where the MODEL chooses the tool and
writes the query each round, until it judges the evidence sufficient (or a hard
round ceiling is hit).

Why JSON-action emulation (not native function-calling): the VNG MaaS / gateway
models are plain OpenAI-compatible chat endpoints with no guaranteed tool-call
schema. So each round we ask the thinker to emit ONE strict JSON action; we
parse it, run the tool, summarize the result, and feed it back. This runs on
ANY chat model (gemma/qwen/minimax/haiku) — no SDK, no tool-call API needed.

TOOLS the model may call (kept tiny + safe — read-only, GreenNode-scoped):
  web_search(query)   live web (SearXNG→DDG auto), official domains first
  fetch_url(url)      crawl ONE url → clean text (Playwright-renders Zoho SPAs)
  search_kb(query)    local cached KB (BM25) — offline rescue
  finish()            model declares it has enough evidence to answer

Each round is traced via the `emit` callback so the dashboard shows the agent
THINKING and ACTING live (not just the final pipeline stages).

Everything degrades honestly: a tool error becomes an observation the model can
react to, never a crash. If the model emits garbage JSON we fall back to a
web_search on the original question so the loop still makes progress.
"""

from __future__ import annotations

import dataclasses
import json
import re
from typing import Callable, Optional

from .agentic import Evidence, collect
from .websearch import SearchResult, greennode_search


# Hard ceiling on agent rounds (anh: "2-3 vòng"). Each round = 1 thinker call
# + at most 1 tool call, so worst case is bounded + cheap.
MAX_AGENT_ROUNDS = 2
# Per-observation summary length so the running context can't blow up.
OBS_SUMMARY_CHARS = 600
# Min distinct evidence pages before the loop will let the model finish early.
MIN_EVIDENCE_PAGES = 2


@dataclasses.dataclass
class AgentStep:
    round: int
    thought: str          # the model's reasoning for this round
    action: str           # web_search | fetch_url | search_kb | finish
    action_input: str     # query or url
    observation: str      # summarized tool result (or note)
    prompt_tokens: int = 0
    completion_tokens: int = 0


@dataclasses.dataclass
class ReactResult:
    evidence: list[Evidence]          # all evidence pages gathered, deduped
    steps: list[AgentStep]            # the full ReAct trace
    rounds: int
    finished_reason: str              # "model_finished" | "max_rounds" | "enough"


# ── Tool registry ────────────────────────────────────────────────────────────
def _make_tools(retr) -> dict[str, Callable]:
    """Build the read-only tool set the agent may call.

    Each tool returns a list[Evidence] (possibly empty). Crawling/rendering is
    reused from agentic.collect so Zoho-SPA help-center pages get Playwright-
    rendered exactly like the non-agentic path.
    """
    from .agentic import make_kb_searcher

    kb_search = make_kb_searcher(retr)

    def web_search(query: str) -> list[Evidence]:
        try:
            results = greennode_search(query, limit=6)
        except Exception:
            results = []
        # Don't Playwright-render here: search hits on Zoho help centers are
        # mostly search/landing pages whose render is wasted (~5s each). We keep
        # them as link-only evidence; AUTO-DRILL in the loop renders the ONE
        # relevant article. This keeps a round to a single render, not 4-5.
        return collect(results, max_renders=0)

    def fetch_url(url: str) -> list[Evidence]:
        # Wrap a single URL as a SearchResult so collect() does the crawl +
        # Playwright render + cleaning + char-cap uniformly (1 render max).
        sr = SearchResult(title=url, url=url, snippet="", engine="fetch_url")
        return collect([sr], max_pages=1, max_renders=1)

    def search_kb(query: str) -> list[Evidence]:
        return collect(kb_search(query, limit=6))

    return {"web_search": web_search, "fetch_url": fetch_url, "search_kb": search_kb}


# ── Action parsing (robust to messy model output) ────────────────────────────
_ACTIONS = ("web_search", "fetch_url", "search_kb", "finish")


def _parse_action(text: str) -> tuple[str, str, str]:
    """Extract (thought, action, action_input) from the model's JSON reply.

    Tolerant: finds the first {...} block, accepts loose keys, defaults to a
    web_search on parse failure so the loop always progresses.
    """
    thought = action = action_input = ""
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        try:
            d = json.loads(m.group(0))
            thought = str(d.get("thought") or d.get("reasoning") or "")[:400]
            action = str(d.get("action") or d.get("tool") or "").strip()
            action_input = str(
                d.get("action_input") or d.get("input") or d.get("query")
                or d.get("url") or ""
            ).strip()
        except Exception:
            pass
    if action not in _ACTIONS:
        # Heuristic recovery: look for an action name mentioned in the text.
        for a in _ACTIONS:
            if a in text:
                action = a
                break
    if action not in _ACTIONS:
        action = "web_search"  # safe default keeps the loop moving
    return thought, action, action_input


# A real Zoho-Desk ARTICLE url has /kb/articles/<slug>. Search/landing pages
# (/kb, /kb/search/...) only list links — their body is NOT the answer, so the
# loop must not treat them as "enough" and must drill into an article instead.
_ARTICLE_URL_RE = re.compile(r"/kb/articles/", re.IGNORECASE)
_ARTICLE_LINK_RE = re.compile(r"https?://[^\s\"'<>]*?/kb/articles/[^\s\"'<>]+", re.IGNORECASE)


def _is_article_url(url: str) -> bool:
    return bool(_ARTICLE_URL_RE.search(url or ""))


def _extract_article_links(retr_or_html: str) -> list[str]:
    """Pull article URLs out of a rendered help-center page's text/HTML."""
    seen, out = set(), []
    for m in _ARTICLE_LINK_RE.findall(retr_or_html or ""):
        u = m.rstrip(".,)\"'")
        if u not in seen:
            seen.add(u)
            out.append(u)
    return out


def _summarize_evidence(evs: list[Evidence]) -> str:
    """Compact observation the model reads next round (keeps context small)."""
    if not evs:
        return "(không thu được nội dung)"
    parts = []
    for e in evs[:4]:
        tag = "★official" if e.official else "web"
        kind = "ARTICLE" if _is_article_url(e.url) else "trang-danh-mục/tìm-kiếm"
        parts.append(f"[{tag}·{kind}] {e.title[:70]} — {e.url}\n{e.text[:OBS_SUMMARY_CHARS]}")
    return "\n\n".join(parts)


# ── The loop ─────────────────────────────────────────────────────────────────
_SYS = (
    "Bạn là TÁC NHÂN nghiên cứu của trợ lý hỗ trợ khách hàng GreenNode/VNG Cloud. "
    "Nhiệm vụ: THU THẬP đủ bằng chứng từ nguồn CHÍNH THỨC để trả lời câu hỏi, KHÔNG tự trả lời ở bước này.\n\n"
    "Mỗi lượt, suy nghĩ rồi chọn ĐÚNG MỘT hành động, trả về DUY NHẤT một JSON (không thêm chữ nào ngoài JSON):\n"
    '{\"thought\":\"<suy luận ngắn: đang thiếu gì, cần tra gì>\",'
    '\"action\":\"web_search|fetch_url|search_kb|finish\",'
    '\"action_input\":\"<truy vấn tìm kiếm HOẶC url để tải HOẶC rỗng nếu finish>\"}\n\n'
    "QUY TẮC:\n"
    "- web_search: tìm trên web (ưu tiên greennode.ai, helpdesk.vngcloud.vn, docs.vngcloud.vn).\n"
    "- fetch_url: khi đã thấy 1 URL hứa hẹn trong observation, TẢI nó để đọc nội dung chi tiết (rất quan trọng với trang helpdesk/Zoho chỉ có tiêu đề).\n"
    "- search_kb: tra kho nội bộ khi web không có.\n"
    "- finish: CHỈ khi đã có nội dung CHI TIẾT đủ để trả lời (không phải chỉ tiêu đề).\n"
    "- Nếu observation chỉ có tiêu đề/đường link mà chưa có bước/chi tiết → PHẢI fetch_url trang đó, ĐỪNG finish.\n"
    "- Tối đa vài lượt, ưu tiên hiệu quả."
)


def run_react(
    question: str,
    retr,
    provider,
    *,
    model: str,
    max_rounds: int = MAX_AGENT_ROUNDS,
    emit: Optional[Callable] = None,
    budget: Optional["LoopBudget"] = None,
) -> ReactResult:
    """Drive the model-controlled ReAct evidence-gathering loop.

    `emit(kind, **data)` optional sink for live tracing (REASON/ACT/OBSERVE).
    `budget` optional LoopBudget from orchestrator — when passed, the loop
    checks `budget.can_gather()` before each REASON call and stops early
    if the budget is exhausted (§3.2 hard ceiling).
    Returns the gathered evidence + the full step trace.
    """
    tools = _make_tools(retr)
    steps: list[AgentStep] = []
    evidence: list[Evidence] = []
    seen_urls: set[str] = set()
    transcript: list[dict] = [{"role": "system", "content": _SYS},
                              {"role": "user", "content": f"CÂU HỎI:\n{question}"}]
    finished_reason = "max_rounds"
    searched_queries: set[str] = set()
    first_official_url = {"u": ""}

    def _emit(kind: str, **data):
        if emit:
            try:
                emit(kind, **data)
            except Exception:
                pass

    for rnd in range(1, max_rounds + 1):
        # Unified hard ceiling (§3.2): stop before a REASON call if the shared
        # loop budget is exhausted. The orchestrator will ESCALATE safely.
        if budget is not None and not budget.can_gather():
            finished_reason = "budget_exhausted"
            break

        # ── REASON: model decides next action ────────────────────────────────
        try:
            # Seat THINKER = minimax-m2.5, a reasoning model whose thinking can't
            # be disabled (it spends ~200 tokens reasoning before the visible JSON
            # action). At 320 tokens it ran out mid-thought → truncated/empty
            # action. Give it room so the ReAct action survives the hidden CoT.
            res = provider.chat(transcript, temperature=0.1, max_tokens=1200, model=model)
            raw = res.text.strip()
            pt, ct = res.prompt_tokens, res.completion_tokens
            if budget is not None:
                budget.bump("gather")
        except Exception as e:
            raw = '{"action":"web_search","action_input":"%s"}' % question.replace('"', "")
            pt = ct = 0
        thought, action, action_input = _parse_action(raw)
        if action != "finish" and not action_input:
            action_input = question  # never fire a tool with an empty query

        # ANTI-LOOP: the model sometimes repeats the SAME web_search every round
        # (landing pages give it nothing new), burning the budget. If it tries a
        # query we've already run, force a fetch_url into the best official URL
        # we've seen so far so the round makes real progress.
        if action == "web_search" and action_input.strip().lower() in searched_queries:
            if first_official_url["u"]:
                action = "fetch_url"
                action_input = first_official_url["u"]
        if action == "web_search":
            searched_queries.add(action_input.strip().lower())

        # ── ACT + OBSERVE ────────────────────────────────────────────────────
        if action == "finish":
            observation = "Tác nhân tuyên bố đã đủ bằng chứng."
            steps.append(AgentStep(rnd, thought, action, "", observation, pt, ct))
            _emit("react", round=rnd, thought=thought, action=action,
                  action_input="", observation=observation, pt=pt, ct=ct)
            finished_reason = "model_finished"
            break

        tool = tools.get(action, tools["web_search"])
        try:
            new_ev = tool(action_input)
        except Exception as ex:
            new_ev = []
            obs_err = f"(lỗi tool {action}: {ex})"
        else:
            obs_err = ""
        # merge fresh evidence (dedup by url, official first)
        for e in new_ev:
            if e.url not in seen_urls:
                seen_urls.add(e.url)
                evidence.append(e)
            if e.official and not first_official_url["u"]:
                first_official_url["u"] = e.url

        # ── AUTO-DRILL: if we landed on a help-center search/landing page (not
        # a real article), pull article links out of it and fetch the most
        # relevant one immediately — don't make the model spend a round, and
        # don't let it "finish" on a list page that has no real steps.
        drilled_note = ""
        landed_non_article = any(
            e.official and not _is_article_url(e.url) for e in new_ev
        )
        have_article = any(_is_article_url(e.url) for e in evidence)
        if landed_non_article and not have_article and rnd < max_rounds:
            links = []
            # Read real <a href> article links off the rendered search/landing
            # page DOM (extract_text strips hrefs, so scraping the text fails —
            # we ask Playwright for the anchors directly).
            try:
                from .helpdesk import find_article_links
                for e in new_ev:
                    if "helpdesk." in e.url and "/kb/articles/" not in e.url:
                        links += find_article_links(e.url)
                        if links:
                            break  # one landing page's links is enough; don't render more
            except Exception:
                pass
            # also try any article URLs that happen to be in the text
            for e in new_ev:
                links += _extract_article_links(e.text)
            # rank article links by overlap with the question terms
            qterms = [w for w in re.findall(r"\w+", question.lower()) if len(w) > 3]
            def _score(u):
                ul = u.lower()
                return sum(1 for w in set(qterms) if w in ul)
            links = sorted(set(links), key=_score, reverse=True)
            if links:
                target = links[0]
                art = tools["fetch_url"](target)
                for e in art:
                    if e.url not in seen_urls:
                        seen_urls.add(e.url)
                        evidence.append(e)
                if art:
                    have_article = any(_is_article_url(e.url) for e in evidence)
                    drilled_note = (f"\n\n[AUTO-DRILL] đã tự mở bài viết: {target}\n"
                                    + _summarize_evidence(art))

        observation = (obs_err or _summarize_evidence(new_ev)) + drilled_note

        steps.append(AgentStep(rnd, thought, action, action_input, observation, pt, ct))
        _emit("react", round=rnd, thought=thought, action=action,
              action_input=action_input, observation=observation, pt=pt, ct=ct)

        # feed the observation back so the next REASON sees it
        transcript.append({"role": "assistant", "content": raw})
        transcript.append({"role": "user", "content":
                           f"OBSERVATION (vòng {rnd}):\n{observation}\n\n"
                           "Dựa trên đây, chọn hành động tiếp theo (JSON). "
                           "CHỈ finish khi đã có bài viết (ARTICLE) với bước/chi "
                           "tiết thật; nếu mới chỉ có trang danh mục/tìm kiếm thì "
                           "fetch_url vào bài viết cụ thể."})

        # ── REFLECT (deterministic guard): stop only when we hold a REAL
        # article (kb/articles/...) with substantial body — a search/landing
        # page no longer counts as "enough".
        article_ev = [e for e in evidence if _is_article_url(e.url) and len(e.text) > 300]
        if article_ev:
            finished_reason = "enough"
            break

    evidence = sorted(evidence, key=lambda e: 0 if e.official else 1)
    return ReactResult(evidence=evidence, steps=steps, rounds=len(steps),
                       finished_reason=finished_reason)


if __name__ == "__main__":
    # Offline-ish smoke: needs network for web_search; stub provider picks tools.
    import sys

    class _StubProvider:
        """Cycles: search → fetch first url → finish. No real LLM needed."""
        def __init__(self):
            self.calls = 0
            self.last_url = ""

        def chat(self, msgs, **kw):
            self.calls += 1
            text = msgs[-1]["content"]
            # grab a url from the last observation to fetch
            m = re.search(r"https?://\S+", text)
            class R:
                prompt_tokens = 10
                completion_tokens = 10
            if self.calls == 1:
                R.text = json.dumps({"thought": "tìm trên web", "action": "web_search",
                                     "action_input": sys.argv[1] if len(sys.argv) > 1
                                     else "thay đổi IP private vServer VNG Cloud"})
            elif self.calls == 2 and m:
                R.text = json.dumps({"thought": "đọc chi tiết", "action": "fetch_url",
                                     "action_input": m.group(0)})
            else:
                R.text = json.dumps({"thought": "đủ rồi", "action": "finish", "action_input": ""})
            return R()

    from .retrieve import build_retriever
    import os
    kb = os.environ.get("NODE_AGENT_KB", "data/kb_chunks.jsonl")
    retr = build_retriever(kb)
    r = run_react(sys.argv[1] if len(sys.argv) > 1 else "thay đổi IP private cho vServer",
                  retr, _StubProvider(), model="stub",
                  emit=lambda k, **d: print(f"  [{d.get('round')}] {d.get('action')}({d.get('action_input','')[:50]}) → {d.get('observation','')[:80]}"))
    print(f"\nfinished={r.finished_reason} rounds={r.rounds} evidence={len(r.evidence)}")
    for e in r.evidence[:5]:
        print(f"  {'★' if e.official else ' '} {e.title[:60]} ({len(e.text)} chars)")
