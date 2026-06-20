#!/usr/bin/env python3
"""Offline test for _chart_from_table — the deterministic table→chart safety net.

Proves: (1) a same-unit column → chart with numbers FROM the table (no fabrication);
(2) a mixed-unit spec table → NO chart (ship table alone, never gently lie);
(3) a single-row / no-number table → NO chart.

Run: .venv/bin/python scripts/test_chart_from_table.py
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from node_agent.orchestrator import _chart_from_table  # noqa: E402

# Same-unit price table across 3 GPUs → bar with [2.69, 1.99, 0.39] GB? no, USD.
SAME_UNIT = """Dạ để Anh/Chị dễ so sánh, em tổng hợp giá thuê on-demand:

| Dòng GPU | Giá thuê |
|---|---|
| H100 | 2.69 USD |
| A100 | 1.99 USD |
| L40S | 0.39 USD |

Giá có thể thay đổi [1]."""

# Mixed-unit spec table (VRAM GB, bandwidth TB/s, price USD) → NO chart.
MIXED_UNIT = """Thông số H100 vs H200:

| Thông số | H100 | H200 |
|---|---|---|
| VRAM | 80 GB | 141 GB |
| Băng thông | 3.35 TB/s | 4.8 TB/s |
| Giá | 2.69 USD | 3.99 USD |

Nguồn [1]."""

# A 2-object × VRAM-only comparison (header is the metric, rows are objects).
SAME_UNIT_2 = """| GPU | VRAM |
|---|---|
| H100 | 80 GB |
| H200 | 141 GB |"""

# Single data row → not enough to plot.
ONE_ROW = """| GPU | VRAM |
|---|---|
| H100 | 80 GB |"""

# A genuine composition table whose %-column sums to ~100 → donut is safe.
COMPOSITION = """Cơ cấu chi phí một cụm GPU:

| Thành phần | Tỉ trọng |
|---|---|
| Compute | 70 % |
| Storage | 20 % |
| Network | 10 % |

Nguồn [1]."""

# An SLA table carrying a single percentage → gauge is safe.
SLA = """Cam kết dịch vụ:

| Chỉ tiêu | Giá trị |
|---|---|
| Uptime SLA | 99.9 % |

Nguồn [1]."""

# REGRESSION (bug 2026-06-16): a STATUS table — prose cells + citation [n] —
# must NOT be plotted. The old loose regex scraped [1]/[2]/+42% out of narrative
# cells and built a garbage chart data=[1.0, 200.0, 2.0]. Now → NO chart.
STATUS_TABLE = """Nguồn [1], [2] không có dữ liệu chuỗi thời gian về giá GPU.

| Hạng mục | Trạng thái dữ liệu | Ghi chú |
|---|---|---|
| Giá thuê GPU theo thời gian | Chưa có trong ngữ cảnh | Nguồn [1], [2] không nêu mốc giá |
| So sánh H100 vs H200 (thông số) | Có | H200 141GB, băng thông +42% so H100 [1] |
| Bảng giá hiện hành GreenNode | Chưa có trong ngữ cảnh | Cần tra trực tiếp [2] |

Em chưa đủ căn cứ dựng biểu đồ xu hướng cho Anh/Chị."""

CASES = [
    ("same-unit price (bar)", SAME_UNIT, ["bar"], True),
    ("status table (prose+cite) → NO chart", STATUS_TABLE, ["line"], False),
    ("mixed-unit spec → NO chart", MIXED_UNIT, ["bar"], False),
    ("same-unit VRAM 2 rows (bar)", SAME_UNIT_2, ["bar"], True),
    ("one row → NO chart", ONE_ROW, ["bar"], False),
    ("donut on non-100 cols → NO chart", SAME_UNIT, ["donut"], False),
    ("donut on real composition → chart", COMPOSITION, ["donut"], True),
    ("gauge on SLA % → chart", SLA, ["gauge"], True),
    ("gauge but no % → NO chart", SAME_UNIT_2, ["gauge"], False),
    ("empty plan → nothing", SAME_UNIT, [], False),
]

npass = nfail = 0
for label, ans, plan, expect_chart in CASES:
    out = _chart_from_table(ans, plan)
    got_chart = out.strip().startswith("```chart")
    ok = got_chart == expect_chart
    detail = ""
    if got_chart:
        m = re.search(r"```chart\s*([\s\S]*?)```", out)
        spec = json.loads(m.group(1).strip())
        # verify EVERY number in the chart appears in the source table
        nums = [v for s in spec.get("series", []) for v in s["data"]]
        missing = [n for n in nums
                   if str(n) not in ans and str(int(n)) not in ans
                   and f"{n:g}" not in ans]
        detail = f"type={spec['type']} unit={spec.get('unit','')} data={nums}"
        if missing:
            ok = False
            detail += f"  FABRICATED={missing}"
    mark = "ok " if ok else "XX "
    print(f"{mark}{label:<34} chart={got_chart}  {detail}")
    npass += ok
    nfail += (not ok)

print(f"\n{npass} pass / {nfail} fail")
sys.exit(1 if nfail else 0)
