#!/usr/bin/env python3
"""Live E2E batch test: 25 representative GreenNode test cases.
Reads dash token from env DASH_TOKEN or falls back to default."""
import json, subprocess, time, re, os
from pathlib import Path
from collections import Counter

SERVER = os.environ.get("NODE_AGENT_SERVER", "http://127.0.0.1:8077")

# Read token same way serve.py does: env -> default
_token_default = "demo-key-change-me"
_token = os.environ.get("NODE_AGENT_DASH_TOKEN") or os.environ.get("DASH_TOKEN") or _token_default

cases = json.loads(Path("/mnt/e/Node Agent Src/.hermes/generated_tests/greennode_nodeagent_testcases_128_corrected.json").read_text())

picked = []
seen = set()
for c in cases:
    if c["category"] not in seen:
        picked.append(c)
        seen.add(c["category"])
for c in cases:
    if len(picked) >= 25:
        break
    if c not in picked:
        picked.append(c)

print(f"Running {len(picked)} live E2E cases against {SERVER}...")

def ask(question, sid="batch-e2e"):
    payload = json.dumps({"question": question, "session_id": sid})
    cmd = [
        "curl", "-s", "-X", "POST", f"{SERVER}/api/ask",
        "-H", "Content-Type: application/json",
        "-H", "Authorization: Bearer " + _token,
        "-d", payload,
        "--max-time", "120",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=130)
    try:
        return json.loads(result.stdout)
    except Exception:
        return {"error": (result.stdout[:200] + " | " + result.stderr[:200]).strip()}

pass_n = 0
fail_n = 0
rows = []
for i, c in enumerate(picked, 1):
    q = c["question"]
    t0 = time.time()
    try:
        r = ask(q, f"batch-{i}")
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
        if ok:
            pass_n += 1
        else:
            fail_n += 1
        rows.append({
            "id": c["id"], "cat": c["category"], "ok": ok, "lat": latency,
            "tier": tier, "think": think, "band": band,
            "exp_t": c["expect_tier"], "exp_th": c["expect_think_level"], "exp_b": c["expect_output_band"],
            "tier_ok": tier_ok, "think_ok": think_ok, "band_ok": band_ok, "content_ok": content_ok,
            "src": len(sources), "verified": verified,
            "ans": answer[:140].replace("\n", " "),
        })
    except Exception as e:
        fail_n += 1
        rows.append({
            "id": c["id"], "cat": c["category"], "ok": False,
            "err": str(e)[:80], "lat": round(time.time() - t0, 1), "ans": "",
        })

print("=" * 90)
print(f"LIVE E2E: {len(picked)} cases | {pass_n} PASS | {fail_n} FAIL")
print("=" * 90 + "\n")

for r in rows:
    if "err" in r:
        print(f"ERR  {r['id']:5} [{r['cat']:22}] lat={r['lat']:5.1f}s  {r['err']}")
    else:
        s = "PASS" if r["ok"] else "FAIL"
        print(
            f"{s:4} {r['id']:5} lat={r['lat']:5.1f}s src={r['src']} "
            f"tier={r['tier']:6}/{r['exp_t']:6} think={r['think']:6}/{r['exp_th']:6} "
            f"band={r['band']:2}/{r['exp_b']:2} ({r['tier_ok']},{r['think_ok']},{r['band_ok']},{r['content_ok']})"
        )

fails = [r for r in rows if not r.get("ok") and "err" not in r]
if fails:
    print(f"\n--- FAILURES DETAIL ({len(fails)}) ---")
    for r in fails:
        probs = []
        if not r["tier_ok"]:
            probs.append(f"tier={r['tier']} exp={r['exp_t']}")
        if not r["think_ok"]:
            probs.append(f"think={r['think']} exp={r['exp_th']}")
        if not r["band_ok"]:
            probs.append(f"band={r['band']} exp={r['exp_b']}")
        if not r["content_ok"]:
            probs.append("content_fail")
        print(f"  {r['id']}: {', '.join(probs)}")
        print(f"    ans: {r.get('ans', '')[:150]}")

print("\n--- DISTRIBUTIONS ---")
ok_rows = [r for r in rows if "err" not in r]
print("tier:", Counter(r.get("tier", "?") for r in ok_rows))
print("think:", Counter(r.get("think", "?") for r in ok_rows))
print("band:", Counter(r.get("band", "?") for r in ok_rows))
