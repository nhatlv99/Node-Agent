"""Tier 5 — Serve layer for Node Agent Assistant.

FastAPI app that exposes the T0-T4 pipeline over HTTP for the web dashboard.
Runs in-process with the `node_agent` brain (single shared hermes-fork venv).

Design:
  - The BM25 retriever is built ONCE at startup (KB load is the slow part) and
    reused across requests — `app.state.retriever`.
  - The LLM provider is OPTIONAL: if Tier-0 env (NODE_AGENT_BASE_URL/MODEL) is
    set, live answers; otherwise the API still serves retrieve-only results so
    the dashboard demos without a key (the offline-verify mode from Tier 3).
  - Every response carries a `live` flag (true = LLM answered, false = retrieve
    -only / mock) so the UI can show a LIVE vs MOCK badge — same degradation
    pattern as the vn-trading-agent dashboard.
  - Bearer auth: a single shared token from NODE_AGENT_DASH_TOKEN (default
    'demo-key-change-me'). Stdlib compare, no DB. Public endpoints: /, /api/health.

Endpoints:
  GET  /                 dashboard HTML (static)
  GET  /api/health       liveness + KB stats + whether LLM is wired
  POST /api/ask          {question} -> grounded answer + sources + quality
  GET  /api/kb           KB stats (chunks, pages, domains)

Run (dev):
  cd "<workspace>"
  hermes-fork/.venv/bin/python -m uvicorn node_agent.api:app --port 8000
"""

from __future__ import annotations

import hmac
import os
import time
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import (
    FileResponse,
    HTMLResponse,
    JSONResponse,
    StreamingResponse,
)
from pydantic import BaseModel

from .reason import answer, build_retriever
from .retrieve import BM25Retriever

# ── config ───────────────────────────────────────────────────────────────────
_WS = Path(__file__).resolve().parent.parent
KB_PATH = os.environ.get("NODE_AGENT_KB", str(_WS / "data" / "kb_chunks.jsonl"))
DASH_TOKEN = os.environ.get("NODE_AGENT_DASH_TOKEN", "demo-key-change-me")
_STATIC = Path(__file__).resolve().parent / "static"


def _maybe_provider():
    """Return a live Provider if Tier-0 env is configured, else None."""
    if os.environ.get("NODE_AGENT_BASE_URL") and os.environ.get("NODE_AGENT_MODEL"):
        try:
            from .provider import Provider

            return Provider()
        except Exception:
            return None
    return None


app = FastAPI(title="Node Agent Assistant", version="0.1.0")


@app.on_event("startup")
def _startup() -> None:
    app.state.retriever = build_retriever(KB_PATH)
    app.state.kb_path = KB_PATH
    # Memory: short-term conversation history + durable cross-session facts.
    from .memory import ConversationMemory, LongTermMemory
    app.state.convo = ConversationMemory(db_path=str(_WS / "data" / "convo.db"))
    app.state.ltm = LongTermMemory(str(_WS / "data" / "memory.db"))


# ── auth ─────────────────────────────────────────────────────────────────────
def _check_auth(request: Request) -> None:
    auth = request.headers.get("authorization", "")
    token = auth[7:] if auth.lower().startswith("bearer ") else ""
    # constant-time compare to avoid token-guessing via timing.
    if not hmac.compare_digest(token, DASH_TOKEN):
        raise HTTPException(status_code=401, detail="unauthorized")


# ── models ───────────────────────────────────────────────────────────────────
class AskRequest(BaseModel):
    question: str
    k: int = 5
    mode: str = "node_assistant"
    model: str | None = None
    live: bool = True  # node_assistant: live agentic loop (real-time, default)
    session_id: str = "default"  # conversation/memory key (per browser session)
    harness: bool = True  # node_assistant: drive the 3-seat multi-model harness


# ── endpoints ────────────────────────────────────────────────────────────────
@app.get("/api/health")
def health() -> dict:
    retr: BM25Retriever = app.state.retriever
    prov = _maybe_provider()
    return {
        "status": "ok",
        "kb_chunks": len(retr.docs),
        "llm_wired": prov is not None,
        "model_name": getattr(prov, "model", "") if prov else "",
        "ts": int(time.time()),
    }


@app.get("/api/kb")
def kb_stats(request: Request) -> dict:
    _check_auth(request)
    retr: BM25Retriever = app.state.retriever
    pages: set[str] = set()
    domains: dict[str, int] = {}
    for d in retr.docs:
        pages.add(d.url)
        seg = d.url.split("greennode.ai/")[-1].split("/")[0] or "home"
        domains[seg] = domains.get(seg, 0) + 1
    return {
        "chunks": len(retr.docs),
        "pages": len(pages),
        "domains": dict(sorted(domains.items(), key=lambda x: -x[1])),
    }


@app.get("/api/models")
def list_models() -> dict:
    """The dashboard exposes ONE virtual model 'Node Agent' (the 3-seat harness),
    not the raw seat models — the user picks a system, not a model."""
    from .modes import DASHBOARD_MODELS, MODES, NODE_AGENT_MODEL_ID, DEFAULT_MODE

    return {
        "models": DASHBOARD_MODELS,
        "default_model": NODE_AGENT_MODEL_ID,
        "modes": [{"id": k, "label": v["label"]} for k, v in MODES.items()],
        "default_mode": DEFAULT_MODE,
    }


@app.post("/api/ask")
def ask(req: AskRequest, request: Request) -> dict:
    _check_auth(request)
    from .modes import get_mode, is_allowed_model, DEFAULT_MODE

    q = (req.question or "").strip()
    if not q:
        raise HTTPException(status_code=400, detail="empty question")

    # Validate mode + model against the registry (reject anything off-list).
    mode = req.mode if req.mode in {"node_assistant", "trading_agent"} else DEFAULT_MODE
    from .modes import NODE_AGENT_MODEL_ID
    # "node-agent" is the virtual harness id, not a real seat model. On the
    # single-model fallback path it must resolve to None (server default), so the
    # harness/seat wiring decides the models — never the virtual id.
    model = req.model if (req.model and is_allowed_model(req.model)) else None
    if model == NODE_AGENT_MODEL_ID:
        model = None

    retr: BM25Retriever = app.state.retriever
    provider = _maybe_provider()
    sid = (req.session_id or "default").strip()[:80]

    # PREFETCH memory (node_assistant only): prior turns + recalled long-term facts.
    history = None
    preamble = ""
    if mode == "node_assistant":
        from .memory import build_memory_preamble
        history = app.state.convo.history(sid)
        preamble = build_memory_preamble(app.state.ltm.recall(sid))

    # ── HARNESS PATH: node_assistant drives the 3-seat multi-model orchestrator
    # (triage → think → write → critique loop) with full Kanban tracing. The
    # single-model `answer()` path stays as a fallback (harness=False / trading).
    if mode == "node_assistant" and req.harness:
        from .modes import get_mode
        from .orchestrator import run as orch_run

        sysprompt = get_mode(mode)["prompt"]
        r = orch_run(q, retr, provider, system_prompt=sysprompt, k=req.k,
                     history=history, memory_preamble=preamble, session_id=sid)
        if mode == "node_assistant":
            from .memory import extract_facts
            app.state.convo.add(sid, "user", q)
            app.state.convo.add(sid, "assistant", r.answer)
            for kk, vv in extract_facts(q).items():
                app.state.ltm.remember(sid, kk, vv)
        return {
            "question": q,
            "answer": r.answer,
            "domain": r.domain,
            "mode": mode,
            "model": r.seats.get("writer", ""),
            "live": provider is not None,
            "used_context": bool(r.sources),
            "quality_score": 1.0 if r.verified else 0.5,
            "quality_reasons": [s["detail"] for s in r.trace.get("steps", []) if s.get("detail")],
            "harness": True,
            "run_id": r.run_id,
            "seats": r.seats,
            "triage": {
                "intent": r.triage.intent, "domain": r.triage.domain,
                "need_thinking": r.triage.need_thinking,
                "answer_shape": r.triage.answer_shape,
                "max_sentences": r.triage.max_sentences,
            },
            "rounds": r.rounds,
            "trace": r.trace,
            "sources": [
                {"n": s.n, "title": s.title, "url": s.url, "score": round(s.score, 2)}
                for s in r.sources
            ],
        }

    res = answer(q, retr, provider, k=req.k, mode=mode, model=model, live=req.live,
                 history=history, memory_preamble=preamble)

    # SYNC memory after the turn: record dialogue + persist coarse durable facts.
    if mode == "node_assistant":
        from .memory import extract_facts
        app.state.convo.add(sid, "user", q)
        app.state.convo.add(sid, "assistant", res.answer)
        for kk, vv in extract_facts(q).items():
            app.state.ltm.remember(sid, kk, vv)

    return {
        "question": q,
        "answer": res.answer,
        "domain": res.domain,
        "mode": mode,
        "model": res.model,
        "live": provider is not None and res.model not in ("(no-llm)", "(no-context)"),
        "used_context": res.used_context,
        "quality_score": res.quality_score,
        "quality_reasons": res.quality_reasons,
        "harness": False,
        "sources": [
            {"n": s.n, "title": s.title, "url": s.url, "score": round(s.score, 2)}
            for s in res.sources
        ],
    }


@app.get("/api/trace")
def trace_recent(request: Request, limit: int = 20, session_id: str | None = None) -> dict:
    """Most-recent harness runs (live Kanban board feed), newest first.

    Pass `session_id` to scope the history to one browser session.
    """
    _check_auth(request)
    from . import trace as tracing
    return {"runs": tracing.recent(limit, session_id=session_id), "lanes": list(tracing.LANES)}


@app.get("/api/baseline")
def trace_baseline(request: Request, session_id: str | None = None) -> dict:
    """Aggregate per-seat baseline (token/latency/pass-rate).

    Pass `session_id` to scope the baseline to one browser session.
    """
    _check_auth(request)
    from . import trace as tracing
    return tracing.baseline(session_id=session_id)


@app.get("/api/roles")
def roles_mapping(request: Request) -> dict:
    """Current role→model wiring so the dashboard shows which model drives each seat."""
    _check_auth(request)
    from . import roles
    mp = roles.current_mapping()
    return {
        "roles": {r: {"model": rm.model, "source": rm.source} for r, rm in mp.items()},
        "prod_targets": roles.PROD_TARGETS,
    }


@app.get("/api/sessions")
def list_sessions(request: Request, limit: int = 50) -> dict:
    """List recent chat sessions (newest first) for the left-rail history.

    Each item: session_id, title (first user turn), turn count, last activity.
    Backed by convo.db so old conversations survive a server restart.
    """
    _check_auth(request)
    return {"sessions": app.state.convo.list_sessions(limit=limit)}


@app.get("/api/session/{session_id}")
def load_session(request: Request, session_id: str) -> dict:
    """Full transcript of one session, for loading an old conversation."""
    _check_auth(request)
    return {
        "session_id": session_id,
        "turns": app.state.convo.full_transcript(session_id),
    }


@app.delete("/api/session/{session_id}")
def delete_session(request: Request, session_id: str) -> dict:
    """Delete a session's transcript + its long-term facts."""
    _check_auth(request)
    app.state.convo.clear(session_id)
    try:
        app.state.ltm.forget(session_id)
    except Exception:
        pass
    return {"deleted": session_id}


@app.get("/api/ask_stream")
def ask_stream(request: Request, q: str, token: str = "", session_id: str = "default",
               k: int = 6) -> StreamingResponse:
    """Realtime per-turn pipeline as Server-Sent Events.

    Runs the 3-seat harness in a worker THREAD; the orchestrator's `emit`
    callback pushes each step (with the seat's real output) onto a queue the
    moment it finishes. This generator drains the queue and streams SSE frames
    to the browser, so the user watches triage → think → write → critique
    unfold live, seeing each model's actual result — not a post-hoc summary.

    Auth via query param (EventSource can't set headers). Same shared token.
    """
    # constant-time token check (query param, since SSE has no custom headers).
    if not hmac.compare_digest(token, DASH_TOKEN):
        raise HTTPException(status_code=401, detail="unauthorized")

    question = (q or "").strip()
    if not question:
        raise HTTPException(status_code=400, detail="empty question")

    import json as _json
    import queue
    import threading

    from .modes import get_mode
    from .orchestrator import run as orch_run

    retr: BM25Retriever = app.state.retriever
    provider = _maybe_provider()
    sid = (session_id or "default").strip()[:80]

    from .memory import build_memory_preamble
    history = app.state.convo.history(sid)
    preamble = build_memory_preamble(app.state.ltm.recall(sid))
    sysprompt = get_mode("node_assistant")["prompt"]

    evq: "queue.Queue[dict | None]" = queue.Queue()

    def _emit(ev: dict) -> None:
        evq.put(ev)

    def _worker() -> None:
        try:
            r = orch_run(question, retr, provider, system_prompt=sysprompt, k=k,
                         session_id=sid,
                         history=history, memory_preamble=preamble, emit=_emit)
            # persist memory after the turn (same as the JSON path)
            from .memory import extract_facts
            app.state.convo.add(sid, "user", question)
            app.state.convo.add(sid, "assistant", r.answer)
            for kk, vv in extract_facts(question).items():
                app.state.ltm.remember(sid, kk, vv)
        except Exception as e:  # surface the failure to the client, then close
            evq.put({"type": "error", "detail": str(e)[:200]})
        finally:
            evq.put(None)  # sentinel: stream complete

    threading.Thread(target=_worker, daemon=True).start()

    def _gen():
        # initial comment frame opens the stream promptly for the browser.
        yield ": stream open\n\n"
        while True:
            ev = evq.get()
            if ev is None:
                yield "event: end\ndata: {}\n\n"
                break
            yield f"data: {_json.dumps(ev, ensure_ascii=False)}\n\n"

    return StreamingResponse(_gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache",
                                      "X-Accel-Buffering": "no"})


# Static vendor assets (ECharts lib, etc.). Served from static/ only, with a
# strict whitelist + path-traversal guard so nothing outside static/ leaks.
_STATIC_TYPES = {".js": "application/javascript", ".css": "text/css",
                 ".map": "application/json", ".svg": "image/svg+xml",
                 ".woff2": "font/woff2", ".png": "image/png"}


@app.get("/static/{name:path}")
def static_asset(name: str) -> FileResponse:
    # resolve and confirm the path stays inside _STATIC (no ../ escape).
    target = (_STATIC / name).resolve()
    try:
        target.relative_to(_STATIC.resolve())
    except ValueError:
        raise HTTPException(status_code=404, detail="not found")
    if not target.is_file() or target.suffix not in _STATIC_TYPES:
        raise HTTPException(status_code=404, detail="not found")
    return FileResponse(target, media_type=_STATIC_TYPES[target.suffix])


@app.get("/echarts.min.js")
def echarts_lib() -> FileResponse:
    f = _STATIC / "echarts.min.js"
    if not f.is_file():
        raise HTTPException(status_code=404, detail="not found")
    return FileResponse(f, media_type="application/javascript")


@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    html = _STATIC / "index.html"
    # no-store so the browser NEVER serves a stale dashboard from cache — the
    # single-file app changes often and a cached copy hides new stream/chart code.
    headers = {"Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
               "Pragma": "no-cache", "Expires": "0"}
    if html.exists():
        return HTMLResponse(html.read_text(encoding="utf-8"), headers=headers)
    return HTMLResponse("<h1>Node Agent Assistant</h1><p>dashboard not built</p>",
                        headers=headers)
