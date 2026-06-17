#!/usr/bin/env python3
"""Deterministic test for the chart PLAN (single vs multi vs none).

This is the single source of truth for "when 1 chart, when many, what type".
It runs OFFLINE (no gateway, no LLM, instant) — `_plan_charts` is pure.

Run: hermes-fork/.venv/bin/python scripts/test_chart_plan.py
"""
from __future__ import annotations

import sys
from pathlib import Path

# Import the package from the workspace root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from node_agent.orchestrator import _plan_charts  # noqa: E402

# Each case: (label, question, n_compar, has_numbers, intent, shape, expected_plan)
# n_compar = how many DISTINCT comparable tokens (h100/h200/gói…) the gate counted.
CASES = [
    # ── NONE: nothing to plot ────────────────────────────────────────────────
    ("no numbers → none",
     "GreenNode có hỗ trợ Kubernetes không?", 0, False, "general", "short", []),
    ("howto, no numbers → none",
     "Cách reset mật khẩu tài khoản", 0, False, "howto", "steps", []),
    ("single object, no viz signal, no numbers → none",
     "GreenNode là gì", 0, False, "general", "short", []),

    # ── SINGLE: one chart, type by data shape ────────────────────────────────
    ("compare 2 objects → single bar",
     "So sánh thông số H100 và H200", 2, True, "compare", "table", ["bar"]),
    ("compare 3+ objects → single bar",
     "So sánh H100 với A100 và V100 về VRAM", 3, True, "compare", "table", ["bar"]),
    ("compare but explicit profile → single radar",
     "So sánh hồ sơ năng lực đa tiêu chí của H100 và H200", 2, True, "compare", "table", ["radar"]),
    ("composition wording → single donut",
     "Cơ cấu chi phí khi thuê một cụm GPU gồm những gì", 0, True, "pricing", "table", ["donut"]),
    ("trend wording → single line",
     "Xu hướng giá GPU H100 theo từng tháng", 0, True, "pricing", "table", ["line"]),
    ("distribution wording → single histogram",
     "Phân bố thời gian phản hồi của API MaaS", 0, True, "general", "bullets", ["histogram"]),
    ("funnel wording → single funnel",
     "Tỷ lệ chuyển đổi qua các bước đăng ký dùng thử trả phí", 0, True, "general", "bullets", ["funnel"]),
    ("heatmap wording → single heatmap",
     "Mật độ sử dụng GPU theo giờ trong ngày theo từng ngày", 0, True, "general", "bullets", ["heatmap"]),
    ("boxplot wording → single boxplot",
     "Độ phân tán giá thuê theo nhóm cấu hình, trung vị và tứ phân vị", 0, True, "pricing", "table", ["boxplot"]),
    ("treemap wording → single treemap",
     "Tỉ trọng nhiều hạng mục dịch vụ bằng cây phân cấp", 0, True, "general", "bullets", ["treemap"]),
    ("scatter wording → single scatter",
     "Tương quan giữa giá và hiệu năng các dòng GPU", 0, True, "compare", "table", ["scatter"]),
    ("gauge wording → single gauge",
     "Mức độ sẵn sàng SLA uptime của dịch vụ là bao nhiêu", 0, True, "general", "short", ["gauge"]),
    ("pricing multi-row table → single bar",
     "Bảng giá các gói thuê GPU", 0, True, "pricing", "table", ["bar"]),

    # ── MULTI: one object, two different-natured aspects ─────────────────────
    ("single object cost+power → donut+radar",
     "Cho em biết chi phí và sức mạnh của riêng H100", 1, True, "pricing", "table", ["donut", "radar"]),
    ("single object price+performance → donut+radar",
     "Giá thuê và hiệu năng của H100 thế nào", 1, True, "pricing", "table", ["donut", "radar"]),

    # ── EDGE: cost+power wording but it's a 2-object comparison → bar (single)─
    ("two objects + cost&power words → still single bar (comparison wins)",
     "So sánh chi phí và hiệu năng H100 với H200", 2, True, "compare", "table", ["bar"]),
    ("cost+power BUT no numbers → none",
     "Chi phí và sức mạnh của H100", 1, False, "pricing", "table", []),
]


def main() -> int:
    npass = nfail = 0
    for label, q, ncmp, hasnum, intent, shape, expected in CASES:
        got = _plan_charts(q, ncmp, hasnum, intent, shape)
        ok = got == expected
        mark = "ok " if ok else "XX "
        print(f"{mark}{label:<55} got={got} expected={expected}")
        if ok:
            npass += 1
        else:
            nfail += 1
    print(f"\n{npass} pass / {nfail} fail")
    return 1 if nfail else 0


if __name__ == "__main__":
    sys.exit(main())
