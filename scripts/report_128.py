#!/usr/bin/env python3
"""Pretty-print the full_e2e_live_128 report as terminal tables.

Reads .hermes/generated_tests/full_e2e_live_128_report.json (written at the end
of scripts/full_e2e_live_128.py) and renders:
  - headline pass-rate + elapsed
  - per-category pass table
  - tier / think / band distribution
  - the worst fail buckets (which axis missed most)
  - a sample of failing cases with the mismatch reason

Safe to run any time; if the report file does not exist yet it says so and exits.
"""
import json
from pathlib import Path
from collections import Counter

REPORT = Path("/mnt/e/Node Agent Src/.hermes/generated_tests/full_e2e_live_128_report.json")


def bar(n, total, width=24):
    if not total:
        return ""
    fill = int(round(width * n / total))
    return "#" * fill + "." * (width - fill)


def main():
    if not REPORT.exists():
        print(f"No report yet at {REPORT}")
        print("The 128-case run writes it only at the very end. Still running.")
        return
    data = json.loads(REPORT.read_text(encoding="utf-8"))
    s = data["summary"]
    rows = data.get("rows", [])

    print("=" * 60)
    print(f"FULL 128-CASE LIVE E2E  —  {s['passed']}/{s['total']} pass "
          f"({s['pass_rate']}%)  |  {s['elapsed_sec']}s "
          f"({s['elapsed_sec']/60:.1f} min)")
    print("=" * 60)

    # per-category
    print("\nPER-CATEGORY PASS:")
    cp = s.get("category_pass", {})
    for cat in sorted(cp):
        p, t = cp[cat]["pass"], cp[cat]["total"]
        pct = round(100.0 * p / t, 0) if t else 0
        print(f"  {cat:22} {p:2}/{t:<2} {int(pct):3}%  {bar(p, t)}")

    # axis accuracy (tier/think/band/content)
    print("\nAXIS ACCURACY (where the routing missed):")
    for axis in ("tier_ok", "think_ok", "band_ok", "content_ok"):
        ok = sum(1 for r in rows if r.get(axis))
        print(f"  {axis:12} {ok:3}/{len(rows)}  {bar(ok, len(rows))}")

    # distributions
    for label, key in (("TIER", "tier_dist"), ("THINK", "think_dist"), ("BAND", "band_dist")):
        print(f"\n{label} DISTRIBUTION: {s.get(key, {})}")

    # fail buckets
    fails = [r for r in rows if not r["ok"]]
    print(f"\nFAILS: {len(fails)}")
    if fails:
        cat_fail = Counter(r["category"] for r in fails)
        print("  by category:", dict(cat_fail))
        # axis miss breakdown among fails
        axis_miss = Counter()
        for r in fails:
            for axis in ("tier_ok", "think_ok", "band_ok", "content_ok"):
                if not r.get(axis):
                    axis_miss[axis] += 1
        print("  axis misses among fails:", dict(axis_miss))
        print("\n  SAMPLE FAILS (first 12):")
        for r in fails[:12]:
            miss = [a.replace("_ok", "") for a in ("tier_ok", "think_ok", "band_ok", "content_ok") if not r.get(a)]
            print(f"   [{r['id']}] {r['category']:14} miss={','.join(miss):22} "
                  f"got tier={r['tier']}/think={r['think']}/band={r['band']} "
                  f"exp {r['exp_tier']}/{r['exp_think']}/{r['exp_output_band'] if 'exp_output_band' in r else r['exp_band']}")
            if r.get("error"):
                print(f"        ERROR: {r['error'][:120]}")
            print(f"        Q: {r['question'][:80]}")

    # latency
    lat = [r["latency"] for r in rows if isinstance(r.get("latency"), (int, float))]
    if lat:
        lat_sorted = sorted(lat)
        p50 = lat_sorted[len(lat_sorted)//2]
        p95 = lat_sorted[int(len(lat_sorted)*0.95)]
        print(f"\nLATENCY: median={p50}s  p95={p95}s  max={max(lat)}s")


if __name__ == "__main__":
    main()
