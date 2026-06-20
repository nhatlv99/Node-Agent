#!/usr/bin/env python3
"""Sequential E2E chart test — calls the REAL 3-seat harness (qwen/minimax/gemma
on VNG MaaS) via /api/ask, one case at a time. For each case it records:
  • which chart type(s) the pipeline emitted,
  • whether it matched the expected family,
  • whether every chart number appears in the answer text (anti-fabrication),
  • quality score + source count,
  • elapsed seconds (the harness is slow; reasoning seats add latency).

Writes scripts/e2e_chart_report.md after EACH case so Nhật can follow live.

Run (background):
  NODE_AGENT_KEY_FILE=... already set on the server; this only needs the token.
  .venv/bin/python scripts/test_e2e_charts.py

Env:
  NA_BASE                default http://127.0.0.1:8077
  NODE_AGENT_DASH_TOKEN  default demo-key-change-me
  NA_CASES               optional: 'core' (7), 'full' (default, 13)
"""
from __future__ import annotations

import json
import os
import re
import time
import urllib.request

BASE = os.environ.get("NA_BASE", "http://127.0.0.1:8077")
TOKEN = os.environ.get("NODE_AGENT_DASH_TOKEN", "demo-key-change-me")
REPORT = os.path.join(os.path.dirname(__file__), "e2e_chart_report.md")

# (label, question, expected_chart_types). expected=[] means NO chart is the
# correct outcome (a how-to / concept question — charting it would be wrong).
# For multi (>=2 expected) ALL must appear; for single ANY one counts as a hit.
CASE_SETS = {
    "core": [
        ("bar — so sánh 2 GPU", "So sánh thông số VRAM và băng thông H100 và H200", ["bar"]),
        ("hbar — xếp hạng GPU", "Xếp hạng giá thuê các dòng GPU của GreenNode từ thấp đến cao", ["hbar", "bar"]),
        ("line — xu hướng thời gian", "Xu hướng giá thuê GPU thay đổi thế nào theo thời gian", ["line"]),
        ("donut — cơ cấu chi phí", "Cơ cấu chi phí khi thuê một cụm GPU gồm những thành phần nào", ["donut", "pie"]),
        ("radar — năng lực 1 GPU", "Hồ sơ năng lực đa tiêu chí của riêng H100", ["radar"]),
        ("gauge — SLA/uptime", "Mức độ cam kết SLA uptime của dịch vụ GreenNode là bao nhiêu", ["gauge"]),
        ("multi — chi phí + sức mạnh", "Cho em biết cả chi phí lẫn sức mạnh của riêng GPU H100", ["donut", "radar"]),
    ],
    "extra": [
        # more bar/line/compare phrasings to test consistency, not luck
        ("bar — bảng giá nhiều GPU", "Bảng giá thuê các loại GPU GreenNode hiện có", ["bar", "hbar"]),
        ("compare — 3 GPU", "So sánh H100, H200 và A100 về bộ nhớ và giá", ["bar", "hbar"]),
        ("line — tăng trưởng", "Diễn biến số lượng model trên MaaS qua các tháng gần đây", ["line", "area"]),
        # negative cases — a chart would be WRONG here
        ("howto — KHÔNG chart", "Hướng dẫn em các bước tạo một GPU instance trên GreenNode", []),
        ("concept — KHÔNG chart", "GreenNode AI Platform là gì và dùng để làm gì", []),
        ("single-spec — KHÔNG chart", "VRAM của H100 là bao nhiêu", []),
    ],
}


def ask(q: str) -> dict:
    body = json.dumps({"question": q, "mode": "node_assistant",
                       "live": True, "harness": True}).encode()
    req = urllib.request.Request(
        f"{BASE}/api/ask", data=body,
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {TOKEN}"})
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.load(r)


def chart_specs(answer: str) -> list:
    out = []
    for c in re.findall(r"```chart\s*([\s\S]*?)```", answer):
        try:
            out.append(json.loads(c.strip()))
        except Exception:
            out.append({"type": "PARSE_ERR"})
    return out


def chart_numbers(specs: list) -> list:
    nums = []
    for spec in specs:
        for s in spec.get("series", []):
            for v in s.get("data", []):
                if isinstance(v, (int, float)):
                    nums.append(v)
                elif isinstance(v, list):
                    nums.extend(x for x in v if isinstance(x, (int, float)))
        if isinstance(spec.get("value"), (int, float)):
            nums.append(spec["value"])
        nums.extend(x for x in spec.get("data", []) if isinstance(x, (int, float)))
    return nums


def write_report(rows: list, done: bool) -> None:
    npass = sum(1 for r in rows if r["hit"] == "ok")
    nfab = sum(1 for r in rows if r["fab"].startswith("NGỜ"))
    lines = [
        "# E2E chart test — 3-seat harness (qwen/minimax/gemma @ VNG MaaS)\n",
        f"_cập nhật: {time.strftime('%H:%M:%S')} — "
        f"{'XONG' if done else 'đang chạy…'} · {npass}/{len(rows)} đúng · "
        f"{nfab} ca nghi bịa_\n",
        "| # | Câu hỏi | Chart kỳ vọng | Model ra | Khớp? | q | #nguồn | Bịa? | s |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        exp = "+".join(r["expect"]) if r["expect"] else "(không chart)"
        got = "+".join(r["got"]) if r["got"] else "—"
        lines.append(
            f"| {r['n']} | {r['label']} | {exp} | {got} | {r['hit']} | "
            f"{r['q']} | {r['nsrc']} | {r['fab']} | {r['elapsed']} |")
    with open(REPORT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main() -> None:
    which = os.environ.get("NA_CASES", "full")
    cases = CASE_SETS["core"][:]
    if which == "full":
        cases += CASE_SETS["extra"]
    elif which == "extra":
        cases = CASE_SETS["extra"][:]

    rows = []
    for i, (label, q, expect) in enumerate(cases, 1):
        t0 = time.time()
        row = {"n": i, "label": label, "expect": expect, "got": [],
               "hit": "?", "q": "?", "nsrc": 0, "fab": "?", "elapsed": 0}
        try:
            d = ask(q)
            ans = d.get("answer", "")
            specs = chart_specs(ans)
            got = [s.get("type", "?") for s in specs]
            row["got"] = got
            row["q"] = d.get("quality_score", "?")
            row["nsrc"] = len(d.get("sources", []))
            if not expect:
                # negative case: PASS means NO chart emitted
                row["hit"] = "ok" if not got else "miss"
            else:
                # expect is a set of ACCEPTABLE types (bar/hbar interchangeable,
                # donut/pie interchangeable). For a multi-chart case we encode the
                # required combo explicitly with '+' inside one token. Here ANY
                # listed type counts as a hit unless the label says 'multi'.
                if "multi" in label:
                    row["hit"] = "ok" if all(e in got for e in expect) else "miss"
                else:
                    row["hit"] = "ok" if any(e in got for e in expect) else "miss"
            nums = chart_numbers(specs)
            missing = [n for n in nums
                       if str(n) not in ans and str(int(n)) not in ans
                       and f"{n:g}" not in ans] if nums else []
            row["fab"] = "không" if not missing else f"NGỜ {missing[:3]}"
        except Exception as e:
            row["got"] = [f"ERR:{str(e)[:40]}"]
            row["hit"] = "err"
            row["fab"] = "—"
        row["elapsed"] = int(time.time() - t0)
        rows.append(row)
        write_report(rows, done=False)
        print(f"[{row['elapsed']}s] {label} → {row['got']} "
              f"hit={row['hit']} fab={row['fab']}")
    write_report(rows, done=True)
    print("DONE_ALL")


if __name__ == "__main__":
    main()
