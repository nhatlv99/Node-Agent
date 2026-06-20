#!/usr/bin/env python3
import json, subprocess, time, re, os
from pathlib import Path
from collections import Counter

SERVER = os.environ.get("NODE_AGENT_SERVER", "http://127.0.0.1:8077")
TOKEN = os.environ.get("NODE_AGENT_DASH_TOKEN") or os.environ.get("DASH_TOKEN") or "demo-key-change-me"
CASES_PATH = Path("/mnt/e/Node Agent Src/.hermes/generated_tests/greennode_nodeagent_testcases_128_corrected.json")
OUT_PATH = Path("/mnt/e/Node Agent Src/.hermes/generated_tests/full_e2e_live_128_report.json")

cases = json.loads(CASES_PATH.read_text())
print(f"Running {len(cases)} live E2E cases against {SERVER} ...")

def ask(question, sid):
    payload = json.dumps({"question": question, "session_id": sid})
    cmd = [
        "curl", "-s", "-X", "POST", f"{SERVER}/api/ask",
        "-H", "Content-Type: application/json",
        "-H", "Authorization: Bearer " + TOKEN,
        "-d", payload,
        "--max-time", "120",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=130)
    try:
        return json.loads(result.stdout)
    except Exception:
        return {"error": (result.stdout[:200] + " | " + result.stderr[:200]).strip()}

rows = []
started = time.time()
for i, c in enumerate(cases, 1):
    t0 = time.time()
    q = c["question"]
    sid = f"full-{i}"
    try:
        r = ask(q, sid)
        latency = round(time.time() - t0, 1)
        tri = r.get("triage") or {}
        answer = r.get("answer", "")
        sources = r.get("sources", [])
        verified = r.get("verified", False)
        tier = tri.get("route_tier", "?")
        think = tri.get("think_level", "?")
        band = tri.get("output_band", "?")
        tier_ok = tier == c["expect_tier"]
        think_ok = think == c["expect_think_level"]
        band_ok = band == c["expect_output_band"]
        content_ok = bool(re.search(c.get("check_regex") or ".", answer, re.I)) if answer else False
        if c.get("expect_refusal"):
            content_ok = content_ok or bool(re.search(r"không|chưa|khó|không có|vui lòng", answer, re.I))
        ok = tier_ok and think_ok and band_ok and content_ok
        rows.append({
            "id": c["id"], "category": c["category"], "question": q,
            "ok": ok, "latency": latency,
            "tier": tier, "think": think, "band": band,
            "exp_tier": c["expect_tier"], "exp_think": c["expect_think_level"], "exp_band": c["expect_output_band"],
            "tier_ok": tier_ok, "think_ok": think_ok, "band_ok": band_ok, "content_ok": content_ok,
            "sources": len(sources), "verified": verified,
            "answer_preview": answer[:200].replace("\n", " "),
            "error": r.get("error", ""),
        })
    except Exception as e:
        rows.append({
            "id": c["id"], "category": c["category"], "question": q,
            "ok": False, "latency": round(time.time()-t0,1),
            "tier": "?", "think": "?", "band": "?",
            "exp_tier": c["expect_tier"], "exp_think": c["expect_think_level"], "exp_band": c["expect_output_band"],
            "tier_ok": False, "think_ok": False, "band_ok": False, "content_ok": False,
            "sources": 0, "verified": False,
            "answer_preview": "",
            "error": str(e),
        })
    if i % 10 == 0:
        passed = sum(1 for r in rows if r["ok"])
        print(f"  progress {i}/{len(cases)} pass={passed}")

elapsed = round(time.time() - started, 1)
passed = sum(1 for r in rows if r["ok"])
failed = len(rows) - passed
summary = {
    "total": len(rows),
    "passed": passed,
    "failed": failed,
    "elapsed_sec": elapsed,
    "pass_rate": round(100.0 * passed / len(rows), 1) if rows else 0,
    "category_counts": dict(Counter(r["category"] for r in rows)),
    "category_pass": {cat: {"pass": sum(1 for r in rows if r["category"]==cat and r["ok"]), "total": sum(1 for r in rows if r["category"]==cat)} for cat in sorted(set(r["category"] for r in rows))},
    "tier_dist": dict(Counter(r["tier"] for r in rows)),
    "think_dist": dict(Counter(r["think"] for r in rows)),
    "band_dist": dict(Counter(r["band"] for r in rows)),
}
OUT_PATH.write_text(json.dumps({"summary": summary, "rows": rows}, ensure_ascii=False, indent=2))
print("DONE")
print(json.dumps(summary, ensure_ascii=False, indent=2))
print(f"saved: {OUT_PATH}")
