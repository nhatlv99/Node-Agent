"""Tier MON — Run trace + baseline board for the multi-model harness.

Every customer request becomes ONE *run* made of ordered *steps*. Each step is
one seat (orchestrator/thinker/writer) doing one job (triage/think/write/
critique). We record which ROLE ran, which concrete MODEL backed it, latency,
token usage, the round number, and a verdict. This is the data the dashboard
Kanban panel renders so anh can SEE the three models working together against a
baseline (token/latency/round/pass-rate per seat).

Storage:
  - In-memory ring (RECENT) for realtime polling (the live Kanban board).
  - SQLite append (data/trace.db) for history + baseline aggregates across runs.

Both are best-effort: a tracing failure must NEVER break an answer, so every
write is wrapped and swallowed. stdlib-only (sqlite3 + time + json).

Lanes (Kanban columns), in flow order:
    TRIAGE → THINK → WRITE → CRITIQUE → DONE        (ESCALATE = gave up)
"""

from __future__ import annotations

import dataclasses
import json
import os
import sqlite3
import threading
import time
import uuid
from collections import deque
from pathlib import Path
from typing import Optional

# Lanes the board renders, in flow order.
LANES = ("TRIAGE", "THINK", "WRITE", "CRITIQUE", "DONE", "ESCALATE")

_DB_PATH = os.environ.get(
    "NODE_AGENT_TRACE_DB",
    str(Path(__file__).resolve().parent.parent / "data" / "trace.db"),
)
# Keep the last N runs hot in memory for the realtime board.
_RING_MAX = 60
RECENT: "deque[dict]" = deque(maxlen=_RING_MAX)
_LOCK = threading.Lock()


@dataclasses.dataclass
class Step:
    lane: str          # one of LANES
    role: str          # logical role (orchestrator/thinker/writer) or "system"
    model: str         # concrete model id that ran (or "" for LM-free steps)
    status: str        # "ok" | "fail" | "skip" | "info"
    detail: str = ""   # short human note for the card
    round: int = 0     # critique round (0 = first pass)
    latency_ms: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    ts: float = dataclasses.field(default_factory=time.time)

    def as_dict(self) -> dict:
        d = dataclasses.asdict(self)
        d["ts"] = round(self.ts, 3)
        return d


class RunTrace:
    """Collects steps for a single request and persists on finish().

    Usage:
        tr = RunTrace(question, mode)
        with tr.step("TRIAGE", "orchestrator", model) as s:
            ...; s.note("need_thinking=True"); s.tokens(p, c)
        tr.finish(verified=True, answer_len=512)
    """

    def __init__(self, question: str, mode: str = "node_assistant",
                 session_id: str = "default") -> None:
        self.run_id = uuid.uuid4().hex[:12]
        self.question = (question or "")[:500]
        self.mode = mode
        self.session_id = (session_id or "default")[:80]
        self.started = time.time()
        self.steps: list[Step] = []
        self.verified = False
        self.answer_len = 0
        self.final_lane = "DONE"

    # context-manager per step so latency is measured automatically.
    def step(self, lane: str, role: str, model: str = ""):
        return _StepCtx(self, lane, role, model)

    def add(self, step: Step) -> None:
        self.steps.append(step)

    def finish(self, *, verified: bool, answer_len: int,
               final_lane: str = "DONE") -> dict:
        self.verified = verified
        self.answer_len = answer_len
        self.final_lane = final_lane
        snap = self.snapshot()
        with _LOCK:
            RECENT.appendleft(snap)
        _persist(snap)
        return snap

    def snapshot(self) -> dict:
        total_ms = int((time.time() - self.started) * 1000)
        seats = sorted({s.role for s in self.steps if s.role not in ("system", "")})
        rounds = max([s.round for s in self.steps], default=0)
        ptok = sum(s.prompt_tokens for s in self.steps)
        ctok = sum(s.completion_tokens for s in self.steps)
        return {
            "run_id": self.run_id,
            "session_id": self.session_id,
            "question": self.question,
            "mode": self.mode,
            "ts": round(self.started, 3),
            "total_ms": total_ms,
            "verified": self.verified,
            "answer_len": self.answer_len,
            "final_lane": self.final_lane,
            "seats": seats,
            "rounds": rounds,
            "prompt_tokens": ptok,
            "completion_tokens": ctok,
            "steps": [s.as_dict() for s in self.steps],
        }


class _StepCtx:
    def __init__(self, run: RunTrace, lane: str, role: str, model: str) -> None:
        self._run = run
        self._step = Step(lane=lane, role=role, model=model, status="ok")
        self._t0 = 0.0

    def __enter__(self) -> "_StepCtx":
        self._t0 = time.time()
        return self

    def note(self, detail: str) -> "_StepCtx":
        self._step.detail = (detail or "")[:300]
        return self

    def tokens(self, prompt: int, completion: int) -> "_StepCtx":
        self._step.prompt_tokens = int(prompt or 0)
        self._step.completion_tokens = int(completion or 0)
        return self

    def round(self, r: int) -> "_StepCtx":
        self._step.round = int(r)
        return self

    def status(self, st: str) -> "_StepCtx":
        self._step.status = st
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        self._step.latency_ms = int((time.time() - self._t0) * 1000)
        if exc_type is not None:
            self._step.status = "fail"
            self._step.detail = (self._step.detail + f" · EXC {exc}")[:300]
        self._run.add(self._step)
        return False  # never swallow real exceptions from the wrapped body


# ── SQLite persistence (best-effort) ─────────────────────────────────────────
def _connect() -> Optional[sqlite3.Connection]:
    try:
        Path(_DB_PATH).parent.mkdir(parents=True, exist_ok=True)
        cx = sqlite3.connect(_DB_PATH, timeout=2.0)
        cx.execute(
            "CREATE TABLE IF NOT EXISTS runs ("
            "run_id TEXT PRIMARY KEY, ts REAL, mode TEXT, question TEXT, "
            "total_ms INTEGER, verified INTEGER, answer_len INTEGER, "
            "final_lane TEXT, rounds INTEGER, prompt_tokens INTEGER, "
            "completion_tokens INTEGER, steps_json TEXT, "
            "session_id TEXT DEFAULT 'default')"
        )
        # Migrate older DBs that predate the session_id column (best-effort).
        cols = {r[1] for r in cx.execute("PRAGMA table_info(runs)").fetchall()}
        if "session_id" not in cols:
            cx.execute("ALTER TABLE runs ADD COLUMN session_id TEXT DEFAULT 'default'")
        cx.commit()
        return cx
    except Exception:
        return None


def _persist(snap: dict) -> None:
    cx = _connect()
    if cx is None:
        return
    try:
        # Named columns (not positional) so schema changes can't silently misalign.
        cx.execute(
            "INSERT OR REPLACE INTO runs "
            "(run_id, session_id, ts, mode, question, total_ms, verified, "
            "answer_len, final_lane, rounds, prompt_tokens, completion_tokens, "
            "steps_json) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                snap["run_id"], snap.get("session_id", "default"), snap["ts"],
                snap["mode"], snap["question"], snap["total_ms"],
                int(snap["verified"]), snap["answer_len"], snap["final_lane"],
                snap["rounds"], snap["prompt_tokens"], snap["completion_tokens"],
                json.dumps(snap["steps"], ensure_ascii=False),
            ),
        )
        cx.commit()
    except Exception:
        pass
    finally:
        cx.close()


def recent(limit: int = 30, session_id: str | None = None) -> list[dict]:
    """Most-recent runs (in-memory ring), newest first — for the live board.

    When `session_id` is given, only that session's runs are returned (the
    per-session history view). Otherwise all sessions are shown.
    """
    with _LOCK:
        runs = list(RECENT)
    if session_id:
        runs = [r for r in runs if r.get("session_id") == session_id]
    return runs[:limit]


def baseline(session_id: str | None = None) -> dict:
    """Aggregate baseline across persisted runs: per-seat token/latency/usage
    and overall pass-rate / thinking-rate / avg critique rounds.

    This is the 'baseline' anh asked for: swap a model at one seat, re-run
    traffic, compare these numbers. When `session_id` is given, the aggregate
    is scoped to that session only (per-session baseline).
    """
    cx = _connect()
    if cx is None:
        return {"runs": 0, "seats": {}, "overall": {}}
    try:
        if session_id:
            rows = cx.execute(
                "SELECT verified, rounds, prompt_tokens, completion_tokens, "
                "total_ms, steps_json, final_lane FROM runs WHERE session_id = ?",
                (session_id,),
            ).fetchall()
        else:
            rows = cx.execute(
                "SELECT verified, rounds, prompt_tokens, completion_tokens, "
                "total_ms, steps_json, final_lane FROM runs"
            ).fetchall()
    except Exception:
        rows = []
    finally:
        cx.close()

    n = len(rows)
    if not n:
        return {"runs": 0, "seats": {}, "overall": {}}

    seat_agg: dict[str, dict] = {}
    think_runs = 0
    verified_runs = 0
    escalated = 0
    total_rounds = 0
    total_ms = 0
    for verified, rounds, _pt, _ct, ms, steps_json, final_lane in rows:
        verified_runs += int(verified or 0)
        total_rounds += int(rounds or 0)
        total_ms += int(ms or 0)
        if final_lane == "ESCALATE":
            escalated += 1
        try:
            steps = json.loads(steps_json or "[]")
        except Exception:
            steps = []
        used_thinker = any(s.get("role") == "thinker" and s.get("status") != "skip"
                           for s in steps)
        think_runs += int(used_thinker)
        for s in steps:
            role = s.get("role") or "system"
            a = seat_agg.setdefault(role, {
                "calls": 0, "ms": 0, "prompt_tokens": 0,
                "completion_tokens": 0, "fail": 0, "skip": 0,
            })
            a["calls"] += 1
            a["ms"] += int(s.get("latency_ms") or 0)
            a["prompt_tokens"] += int(s.get("prompt_tokens") or 0)
            a["completion_tokens"] += int(s.get("completion_tokens") or 0)
            if s.get("status") == "fail":
                a["fail"] += 1
            if s.get("status") == "skip":
                a["skip"] += 1

    for role, a in seat_agg.items():
        c = max(1, a["calls"])
        a["avg_ms"] = round(a["ms"] / c, 1)
        a["avg_completion_tokens"] = round(a["completion_tokens"] / c, 1)

    return {
        "runs": n,
        "overall": {
            "pass_rate": round(verified_runs / n, 3),
            "thinking_rate": round(think_runs / n, 3),
            "escalate_rate": round(escalated / n, 3),
            "avg_rounds": round(total_rounds / n, 2),
            "avg_total_ms": round(total_ms / n, 1),
        },
        "seats": seat_agg,
    }


if __name__ == "__main__":
    # Offline self-test: synthesise a run and print the snapshot + baseline.
    tr = RunTrace("Bảng giá GPU H100 của GreenNode?")
    with tr.step("TRIAGE", "orchestrator", "minimax/minimax-m2.5") as s:
        time.sleep(0.01)
        s.note("intent=pricing need_thinking=True shape=table").tokens(120, 40)
    with tr.step("THINK", "thinker", "qwen/qwen3-5-24b") as s:
        time.sleep(0.01)
        s.note("plan: tách SXM vs PCIe, tìm giá/giờ").tokens(300, 90)
    with tr.step("WRITE", "writer", "google/gemma-4-31b-it") as s:
        time.sleep(0.01)
        s.note("drafted table answer").tokens(800, 260)
    with tr.step("CRITIQUE", "orchestrator", "minimax/minimax-m2.5") as s:
        s.round(1).note("gate ok, cite [1..4] valid").tokens(200, 8)
    snap = tr.finish(verified=True, answer_len=512)
    print(json.dumps(snap, ensure_ascii=False, indent=2)[:900])
    print("\n=== baseline ===")
    print(json.dumps(baseline(), ensure_ascii=False, indent=2)[:900])
