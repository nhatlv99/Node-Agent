# E2E chart test — 3-seat harness (qwen/minimax/gemma @ VNG MaaS)

_cập nhật: 05:03:21 — XONG · 10/13 đúng · 0 ca nghi bịa_

| # | Câu hỏi | Chart kỳ vọng | Model ra | Khớp? | q | #nguồn | Bịa? | s |
|---|---|---|---|---|---|---|---|---|
| 1 | bar — so sánh 2 GPU | bar | bar+bar | ok | 1.0 | 13 | không | 30 |
| 2 | hbar — xếp hạng GPU | hbar+bar | hbar | ok | 1.0 | 9 | không | 19 |
| 3 | line — xu hướng thời gian | line | line | ok | 1.0 | 9 | không | 22 |
| 4 | donut — cơ cấu chi phí | donut+pie | — | miss | 1.0 | 7 | không | 15 |
| 5 | radar — năng lực 1 GPU | radar | radar | ok | 1.0 | 9 | không | 23 |
| 6 | gauge — SLA/uptime | gauge | gauge | ok | 1.0 | 7 | không | 2 |
| 7 | multi — chi phí + sức mạnh | donut+radar | radar | miss | 1.0 | 8 | không | 67 |
| 8 | bar — bảng giá nhiều GPU | bar+hbar | bar | ok | 1.0 | 9 | không | 7 |
| 9 | compare — 3 GPU | bar+hbar | bar | ok | 1.0 | 13 | không | 36 |
| 10 | line — tăng trưởng | line+area | — | miss | 1.0 | 10 | không | 32 |
| 11 | howto — KHÔNG chart | (không chart) | — | ok | 1.0 | 10 | không | 28 |
| 12 | concept — KHÔNG chart | (không chart) | — | ok | 1.0 | 15 | không | 57 |
| 13 | single-spec — KHÔNG chart | (không chart) | — | ok | 1.0 | 9 | không | 3 |
