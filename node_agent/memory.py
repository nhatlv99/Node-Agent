"""Tier MEM — conversation + long-term memory for Node Agent Assistant.

Two layers, modeled on Hermes' memory pattern (prefetch before a turn, sync
after) but self-contained so the node_agent pipeline stays independent of the
Hermes run-loop.

1. ConversationMemory (short-term, per session)
   - Holds the recent user/assistant turns of ONE chat session in memory.
   - prefetch() returns the trimmed history (token-budgeted) to prepend to the
     LLM messages so follow-ups ("thế còn H200?") keep context.
   - Ring-buffered + char-budgeted so a long chat can't blow the context window.

2. LongTermMemory (durable, cross-session)
   - SQLite-backed key/fact store keyed by session_id (stdlib sqlite3, no deps).
   - Persists durable facts the customer stated (their plan, region, GPU of
     interest, account email domain) so a returning session can recall them.
   - remember()/recall() are explicit; the agent layer decides what's worth
     persisting (we extract a few safe, structured signals — never secrets/PII
     values, only coarse intent tags).

Both are OPTIONAL and injected — the pipeline runs unchanged when memory is off,
mirroring the `provider=None` offline-verify pattern used elsewhere.
"""

from __future__ import annotations

import dataclasses
import sqlite3
import threading
import time
from pathlib import Path
from typing import Optional


# ── 1. Short-term conversation memory ────────────────────────────────────────
@dataclasses.dataclass
class Turn:
    role: str        # "user" | "assistant"
    content: str
    ts: float = dataclasses.field(default_factory=time.time)


class ConversationMemory:
    """Per-session chat history with a char budget, WRITE-THROUGH to SQLite.

    The pipeline was one-shot (no history). This lets follow-up questions keep
    context. History is prepended to the LLM messages BEFORE the grounded
    user turn, so the model sees the dialogue but the citation rules still bind
    only the final answer to the live evidence.

    Persistence: when `db_path` is given, every turn is written to SQLite so the
    transcript survives a server restart — this is what powers the left-rail
    "session cũ" list (list_sessions / full_transcript). A hot in-memory ring
    is still kept per session for fast context prefetch. When `db_path` is None
    the store is purely in-memory (the original behaviour, for tests/offline).
    """

    def __init__(self, max_turns: int = 12, max_chars: int = 6000,
                 db_path: str | Path | None = None) -> None:
        self.max_turns = max_turns
        self.max_chars = max_chars
        self._by_session: dict[str, list[Turn]] = {}
        self._lock = threading.Lock()
        self.db_path = str(db_path) if db_path else None
        if self.db_path:
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
            self._init_db()
            self._warm_load()

    # ── SQLite persistence ───────────────────────────────────────────────────
    def _conn(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path, check_same_thread=False)

    def _init_db(self) -> None:
        with self._lock, self._conn() as c:
            c.execute(
                """
                CREATE TABLE IF NOT EXISTS turns (
                    id         INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    role       TEXT NOT NULL,
                    content    TEXT NOT NULL,
                    ts         REAL NOT NULL
                )
                """
            )
            c.execute(
                "CREATE INDEX IF NOT EXISTS idx_turns_sid ON turns(session_id, id)"
            )

    def _warm_load(self) -> None:
        """Re-hydrate the in-memory ring (last max_turns per session) from disk
        so context prefetch works immediately after a restart."""
        try:
            with self._lock, self._conn() as c:
                rows = c.execute(
                    "SELECT session_id, role, content, ts FROM turns ORDER BY id"
                ).fetchall()
            for sid, role, content, ts in rows:
                buf = self._by_session.setdefault(sid, [])
                buf.append(Turn(role=role, content=content, ts=ts))
                if len(buf) > self.max_turns:
                    del buf[: len(buf) - self.max_turns]
        except Exception:
            pass

    def add(self, session_id: str, role: str, content: str) -> None:
        if not session_id or not content:
            return
        content = content.strip()
        with self._lock:
            buf = self._by_session.setdefault(session_id, [])
            buf.append(Turn(role=role, content=content))
            # ring-buffer by turn count (in-memory only; disk keeps everything)
            if len(buf) > self.max_turns:
                del buf[: len(buf) - self.max_turns]
        if self.db_path:
            try:
                with self._lock, self._conn() as c:
                    c.execute(
                        "INSERT INTO turns(session_id, role, content, ts) VALUES(?,?,?,?)",
                        (session_id, role, content, time.time()),
                    )
            except Exception:
                pass

    def history(self, session_id: str) -> list[dict]:
        """Return OpenAI-format history messages, trimmed to the char budget.

        Trims from the OLDEST end so the most recent context survives.
        """
        with self._lock:
            buf = list(self._by_session.get(session_id, []))
        # Walk newest→oldest accumulating until the budget is hit, then reverse.
        out: list[dict] = []
        total = 0
        for t in reversed(buf):
            c = len(t.content)
            if total + c > self.max_chars:
                break
            out.append({"role": t.role, "content": t.content})
            total += c
        out.reverse()
        return out

    def clear(self, session_id: str) -> None:
        with self._lock:
            self._by_session.pop(session_id, None)
        if self.db_path:
            try:
                with self._lock, self._conn() as c:
                    c.execute("DELETE FROM turns WHERE session_id=?", (session_id,))
            except Exception:
                pass

    # ── session management (left-rail list / load) ───────────────────────────
    def list_sessions(self, limit: int = 50) -> list[dict]:
        """Return recent sessions newest-first: id, title (first user turn),
        turn count, last-activity ts. Backed by SQLite so it survives restarts."""
        if not self.db_path:
            # in-memory fallback: derive from the hot ring
            out = []
            with self._lock:
                for sid, buf in self._by_session.items():
                    first_user = next((t.content for t in buf if t.role == "user"), "")
                    out.append({"session_id": sid, "title": first_user[:80],
                                "turns": len(buf), "last_ts": buf[-1].ts if buf else 0})
            return sorted(out, key=lambda x: -x["last_ts"])[:limit]
        try:
            with self._lock, self._conn() as c:
                rows = c.execute(
                    """
                    SELECT session_id,
                           COUNT(*) AS turns,
                           MAX(ts)  AS last_ts,
                           MIN(id)  AS first_id
                    FROM turns GROUP BY session_id
                    ORDER BY last_ts DESC LIMIT ?
                    """,
                    (limit,),
                ).fetchall()
                out = []
                for sid, turns, last_ts, first_id in rows:
                    title_row = c.execute(
                        "SELECT content FROM turns WHERE session_id=? AND role='user' "
                        "ORDER BY id LIMIT 1", (sid,)
                    ).fetchone()
                    title = (title_row[0] if title_row else "")[:80]
                    out.append({"session_id": sid, "title": title or "(chưa có câu hỏi)",
                                "turns": turns, "last_ts": last_ts})
                return out
        except Exception:
            return []

    def full_transcript(self, session_id: str) -> list[dict]:
        """Return the COMPLETE transcript of a session (not budget-trimmed) for
        loading an old conversation into the UI."""
        if not self.db_path:
            with self._lock:
                buf = list(self._by_session.get(session_id, []))
            return [{"role": t.role, "content": t.content, "ts": t.ts} for t in buf]
        try:
            with self._lock, self._conn() as c:
                rows = c.execute(
                    "SELECT role, content, ts FROM turns WHERE session_id=? ORDER BY id",
                    (session_id,),
                ).fetchall()
            return [{"role": r, "content": ct, "ts": ts} for r, ct, ts in rows]
        except Exception:
            return []


# ── 2. Long-term cross-session memory ────────────────────────────────────────
class LongTermMemory:
    """SQLite key/fact store keyed by session_id. stdlib only.

    Stores COARSE, durable signals (not secrets/PII values): e.g. the product
    the customer keeps asking about, their region, their plan tier. A returning
    session can recall("region") to personalise. Facts are short strings.
    """

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = str(db_path)
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._init_db()

    def _conn(self) -> sqlite3.Connection:
        # check_same_thread=False + our own lock = safe for the FastAPI workers.
        return sqlite3.connect(self.db_path, check_same_thread=False)

    def _init_db(self) -> None:
        with self._lock, self._conn() as c:
            c.execute(
                """
                CREATE TABLE IF NOT EXISTS facts (
                    session_id TEXT NOT NULL,
                    key        TEXT NOT NULL,
                    value      TEXT NOT NULL,
                    updated_at REAL NOT NULL,
                    PRIMARY KEY (session_id, key)
                )
                """
            )

    def remember(self, session_id: str, key: str, value: str) -> None:
        if not (session_id and key and value):
            return
        with self._lock, self._conn() as c:
            c.execute(
                "INSERT INTO facts(session_id, key, value, updated_at) "
                "VALUES(?,?,?,?) ON CONFLICT(session_id, key) "
                "DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at",
                (session_id, key, value[:500], time.time()),
            )

    def recall(self, session_id: str) -> dict[str, str]:
        with self._lock, self._conn() as c:
            rows = c.execute(
                "SELECT key, value FROM facts WHERE session_id=? ORDER BY updated_at",
                (session_id,),
            ).fetchall()
        return {k: v for k, v in rows}

    def forget(self, session_id: str) -> None:
        with self._lock, self._conn() as c:
            c.execute("DELETE FROM facts WHERE session_id=?", (session_id,))


# ── Fact extraction (LLM-free, conservative) ─────────────────────────────────
import re

# Coarse, safe signals only. We never store the customer's raw email / phone /
# account id — only which PRODUCT and which REGION/INTENT they care about, so a
# returning session can be greeted in context.
_PRODUCT_RE = re.compile(
    r"\b(h100|h200|gpu|vks|kubernetes|maas|idp|ocr|vstorage|object storage|"
    r"vserver|cpu instance|ai platform|vdb|vbackup|vcdn)\b", re.IGNORECASE)
_REGION_RE = re.compile(r"\b(hà nội|ha noi|hanoi|hcm|hồ chí minh|bangkok|thái lan|thailand|singapore)\b", re.IGNORECASE)


def extract_facts(question: str) -> dict[str, str]:
    """Pull coarse, non-PII facts from a question for long-term memory."""
    facts: dict[str, str] = {}
    prods = sorted({m.lower() for m in _PRODUCT_RE.findall(question)})
    if prods:
        facts["interested_products"] = ", ".join(prods[:5])
    region = _REGION_RE.search(question)
    if region:
        facts["region"] = region.group(0).lower()
    return facts


def build_memory_preamble(ltm_facts: dict[str, str]) -> str:
    """Render recalled long-term facts as a short system-prompt preamble."""
    if not ltm_facts:
        return ""
    bits = []
    if "interested_products" in ltm_facts:
        bits.append(f"sản phẩm quan tâm: {ltm_facts['interested_products']}")
    if "region" in ltm_facts:
        bits.append(f"khu vực: {ltm_facts['region']}")
    if not bits:
        return ""
    return ("BỐI CẢNH KHÁCH HÀNG (từ các lần trao đổi trước, dùng để cá nhân hoá, "
            "KHÔNG bịa thêm): " + "; ".join(bits))


if __name__ == "__main__":
    # Smoke test — no network, no deps.
    cm = ConversationMemory(max_turns=4, max_chars=200)
    sid = "s1"
    cm.add(sid, "user", "Giá GPU H100 bao nhiêu?")
    cm.add(sid, "assistant", "H100 từ $2.99/giờ.")
    cm.add(sid, "user", "Thế còn H200?")
    print("history:", [m["role"] for m in cm.history(sid)], "turns")

    import tempfile, os
    db = os.path.join(tempfile.gettempdir(), "na_mem_test.db")
    ltm = LongTermMemory(db)
    ltm.remember(sid, "region", "hcm")
    for k, v in extract_facts("Em muốn thuê H100 ở Hà Nội cho VKS").items():
        ltm.remember(sid, k, v)
    print("recall:", ltm.recall(sid))
    print("preamble:", build_memory_preamble(ltm.recall(sid)))
    ltm.forget(sid)
    print("after forget:", ltm.recall(sid))
