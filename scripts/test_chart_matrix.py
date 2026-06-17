#!/usr/bin/env python3
"""FULL chart-type matrix: one representative customer question per renderer
type, run through the deterministic planner `_plan_charts`. Exposes HONESTLY
which of the 16 renderer types the planner can actually produce, and which it
cannot (and therefore depend on the model's own discretion or another mode).

Run: hermes-fork/.venv/bin/python scripts/test_chart_matrix.py
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from node_agent.orchestrator import _plan_charts, _COMPARABLE_RE  # noqa: E402

# All 16 renderer types the frontend _chartOption supports.
ALL_TYPES = ["bar", "hbar", "line", "area", "pie", "donut", "radar", "gauge",
             "scatter", "candlestick", "heatmap", "histogram", "boxplot",
             "funnel", "treemap"]

# (renderer_type, a representative GreenNode-customer question that SHOULD aim
#  at that chart, intent, shape). has_numbers assumed True (evidence has data).
CASES = [
    ("bar",       "So sánh giá thuê GPU H100 và H200", "compare", "table"),
    ("hbar",      "Xếp hạng giá thuê 8 dòng GPU từ thấp đến cao", "pricing", "table"),
    ("line",      "Xu hướng giá GPU H100 theo từng tháng năm nay", "pricing", "table"),
    ("area",      "Lưu lượng băng thông tích luỹ theo thời gian", "general", "bullets"),
    ("pie",       "Vẽ biểu đồ tròn thị phần các dòng GPU", "general", "table"),
    ("donut",     "Cơ cấu chi phí một cụm GPU gồm những gì", "pricing", "table"),
    ("radar",     "Hồ sơ năng lực đa tiêu chí của H100", "spec", "bullets"),
    ("gauge",     "Mức độ sẵn sàng SLA uptime của dịch vụ", "general", "short"),
    ("scatter",   "Tương quan giữa giá và hiệu năng các GPU", "compare", "table"),
    ("candlestick","Biểu đồ nến giá cổ phiếu VN30 tuần này", "general", "table"),
    ("heatmap",   "Mật độ sử dụng GPU theo giờ trong ngày", "general", "bullets"),
    ("histogram", "Phân bố thời gian phản hồi của API MaaS", "general", "bullets"),
    ("boxplot",   "Độ phân tán giá thuê theo nhóm, trung vị tứ phân vị", "pricing", "table"),
    ("funnel",    "Tỷ lệ chuyển đổi qua các bước đăng ký dùng thử trả phí", "general", "bullets"),
    ("treemap",   "Tỉ trọng nhiều hạng mục dịch vụ bằng cây phân cấp", "general", "bullets"),
]


def n_compar(q: str) -> int:
    return len({m.group(0).lower() for m in _COMPARABLE_RE.finditer(q.lower())})


def main() -> int:
    reachable = set()
    print(f"{'TARGET':<12} {'PLANNER OUTPUT':<22} HIT?")
    print("-" * 50)
    for target, q, intent, shape in CASES:
        plan = _plan_charts(q, n_compar(q), True, intent, shape)
        for p in plan:
            reachable.add(p)
        hit = "ok" if target in plan else "XX  (planner can't pick this)"
        print(f"{target:<12} {str(plan):<22} {hit}")

    # Multi-chart case (the special feature).
    multi_q = "Cho em biết chi phí và sức mạnh của riêng H100"
    multi = _plan_charts(multi_q, n_compar(multi_q), True, "pricing", "table")
    print("-" * 50)
    print(f"{'MULTI':<12} {str(multi):<22} "
          f"{'ok (donut+radar)' if multi == ['donut','radar'] else 'XX'}")

    print("\n=== COVERAGE vs all 16 renderer types ===")
    never = [t for t in ALL_TYPES if t not in reachable]
    print(f"planner CAN produce ({len(reachable)}): {sorted(reachable)}")
    print(f"planner NEVER produces ({len(never)}): {never}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
