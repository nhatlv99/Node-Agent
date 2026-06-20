#!/usr/bin/env python3
"""Token-control E2E: verify 6 signature questions + token budget.
Usage: python3 scripts/test_token_control.py
"""
import json, os, sys, time, re, urllib.request, urllib.error

SERVER = os.environ.get("NODE_AGENT_SERVER", "http://127.0.0.1:8077")
TOKEN = os.environ.get("NODE_AGENT_TOKEN", "demo-key-change-me")

CASES = [
    {"id": "C1", "q": "GreenNode có những dịch vụ gì?",
     "expect_think": "high", "expect_band": "M", "expect_tier": "medium",
     "check": "dịch|service|GPU|VKS|MaaS|storage|cloud"},
    {"id": "C2", "q": "So sánh tất cả các dòng GPU của GreenNode với nhau",
     "expect_think": "high", "expect_band": "L", "expect_tier": "heavy",
     "check": "H100|H200|GPU|A100|L40"},
    {"id": "C3", "q": "So sánh H100 và H200",
     "expect_think": "low", "expect_band": "M", "expect_tier": "heavy",
     "check": "80|141|H100|H200|VRAM"},
    {"id": "C4a", "q": "GreenNode có bao nhiêu loại instance flavor",
     "expect_think": "medium", "expect_band": "L", "expect_tier": "medium",
     "check": "instance|flavor|gói|loại"},
    {"id": "C4b", "q": "Phân tích chi tiết từng gói instance flavor của GreenNode và vẽ chart",
     "expect_think": "high", "expect_band": "XL", "expect_tier": "heavy",
     "check": "flavor|instance|gói|chart|phân tích"},
    {"id": "C5", "q": "Giá thuê GPU H100 bao nhiêu",
     "expect_think": "none", "expect_band": "S", "expect_tier": "light",
     "check": "giá|H100|liên hệ|thuê"},
]


def ask(question, timeout=120):
    url = SERVER + "/api/ask"
    body = json.dumps({"question": question, "session_id": "token-ctrl-test"}).encode()
    req = urllib.request.Request(url, data=body,
        headers={"Content-Type": "application/json", "X-API-Key": TOKEN}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def main():
    print("Token-control E2E -> " + SERVER + "\n")
    passed = failed = 0
    results = []
    for c in CASES:
        print("=" * 70)
        print(c["id"] + ": " + c["q"])
        print("=" * 70)
        t0 = time.time()
        try:
            res = ask(c["q"])
        except Exception as e:
            print("  ERROR: " + str(e))
            failed += 1
            results.append({"id": c["id"], "status": "ERROR"})
            continue
        elapsed = time.time() - t0
        answer = res.get("answer", "")
        sources = res.get("sources", [])
        verified = res.get("verified", False)
        triage = res.get("triage", {})
        rounds = res.get("rounds", 0)
        think_level = triage.get("think_level", "?")
        output_band = triage.get("output_band", "?")
        route_tier = triage.get("route_tier", "?")
        intent = triage.get("intent", "?")
        think_ok = think_level == c["expect_think"]
        band_ok = output_band == c["expect_band"]
        tier_ok = route_tier == c["expect_tier"]
        cls_pass = think_ok and band_ok and tier_ok
        content_ok = bool(re.search(c["check"], answer, re.IGNORECASE)) if answer else False
        fabric_fail = len(sources) == 0 and bool(re.search(r"\d+", answer)) and intent != "meta"
        status = "PASS" if (cls_pass and content_ok and not fabric_fail) else "FAIL"
        if status == "PASS":
            passed += 1
        else:
            failed += 1
        tag = lambda ok: "V" if ok else "X"
        print("  think=" + think_level + " " + tag(think_ok)
              + "  band=" + output_band + " " + tag(band_ok)
              + "  tier=" + route_tier + " " + tag(tier_ok)
              + "  content " + tag(content_ok)
              + "  fabric " + tag(not fabric_fail))
        print("  latency=" + str(round(elapsed, 1)) + "s  rounds=" + str(rounds)
              + "  sources=" + str(len(sources)) + "  verified=" + str(verified)
              + "  intent=" + intent)
        print("  answer: " + answer[:120].replace("\n", " ") + "...")
        print("  -> " + status)
        results.append({"id": c["id"], "status": status, "think": think_level,
                        "band": output_band, "tier": route_tier,
                        "latency": round(elapsed, 1), "sources": len(sources)})
    print("\n" + "=" * 70)
    print("SUMMARY: " + str(passed) + "/" + str(passed + failed) + " pass, " + str(failed) + " fail")
    print("=" * 70)
    for r in results:
        print("  " + r["id"] + " " + r.get("status", "?"))
    print()
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
