#!/usr/bin/env python3
"""Direct E2E against REAL VNG MaaS 3-seat harness (no web server).

Calls orchestrator.run() in-process so we don't need fastapi/uvicorn.
Wires the production VNG Cloud MaaS endpoint + the 3 seat models, then
runs a few representative questions and prints answer + sources +
chart + timing so we can eyeball the two just-fixed issues:
  1. duplicate source links at the end
  2. apology-instead-of-answer after a refine loop
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

WS = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(WS))

# --- wire VNG Cloud MaaS (production) -------------------------------------
MAAS_BASE_URL = "https://maas-llm-aiplatform-hcm.api.vngcloud.vn/v1"


def _load_key() -> str:
    kf = os.environ.get("NODE_AGENT_KEY_FILE", "").strip()
    if kf and Path(kf).is_file():
        return Path(kf).read_text(encoding="utf-8").strip()
    for name in ("NODE_AGENT_API_KEY", "MAAS_API_KEY", "AI_PLATFORM_API_KEY", "API_KEY"):
        v = os.environ.get(name, "").strip()
        if v:
            return v
    raise SystemExit("No MaaS key. Set NODE_AGENT_KEY_FILE=/path/to/Apikey.txt")


key = _load_key()
os.environ["NODE_AGENT_BASE_URL"] = MAAS_BASE_URL
os.environ["NODE_AGENT_API_KEY"] = key
# 3 seats (technical-behaviour mapping)
os.environ.setdefault("NODE_AGENT_MODEL_ORCHESTRATOR", "qwen/qwen3-5-27b")
os.environ.setdefault("NODE_AGENT_MODEL_THINKER", "minimax/minimax-m2.5")
os.environ.setdefault("NODE_AGENT_MODEL_WRITER", "google/gemma-4-31b-it")
os.environ["NODE_AGENT_MODEL"] = "google/gemma-4-31b-it"  # single-fallback

from node_agent.provider import Provider          # noqa: E402
from node_agent.reason import build_retriever      # noqa: E402
from node_agent.orchestrator import run            # noqa: E402
from node_agent.modes import get_mode              # noqa: E402

KB = os.environ.get("NODE_AGENT_KB", str(WS / "data" / "kb_chunks.jsonl"))

QUESTIONS = [
    "So sánh thông số VRAM và băng thông của H100 và H200",
    "GreenNode có hỗ trợ Kubernetes không và tính năng chính là gì",
    "Bảng giá thuê GPU H100 của GreenNode là bao nhiêu",
]


def main() -> None:
    print(f"[setup] base={MAAS_BASE_URL}")
    print(f"[setup] KB={KB}")
    t0 = time.time()
    retr = build_retriever(KB)
    print(f"[setup] retriever ready: {len(retr.docs)} chunks in {time.time()-t0:.1f}s")
    provider = Provider()
    system_prompt = get_mode("node_assistant")["prompt"]

    for i, q in enumerate(QUESTIONS, 1):
        print("\n" + "=" * 78)
        print(f"Q{i}: {q}")
        print("=" * 78)
        t = time.time()
        try:
            res = run(q, retr, provider, system_prompt=system_prompt,
                      session_id=f"e2e{i}")
        except Exception as e:
            print(f"!! ERROR: {type(e).__name__}: {e}")
            continue
        dt = time.time() - t
        print(f"\n--- ANSWER (verified={res.verified} rounds={res.rounds} {dt:.1f}s) ---")
        print(res.answer)
        print(f"\n--- SOURCES ({len(res.sources)}) ---")
        for s in res.sources:
            print(f"  [{s.n}] {s.title}  ->  {s.url}")
        # duplicate-url check
        urls = [s.url for s in res.sources]
        dupes = {u for u in urls if urls.count(u) > 1}
        print(f"\n[check] duplicate source URLs: {sorted(dupes) if dupes else 'NONE'}")


if __name__ == "__main__":
    main()
