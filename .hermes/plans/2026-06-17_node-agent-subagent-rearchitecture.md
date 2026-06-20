# Node Agent — Subagent Re-architecture Plan

Date: 2026-06-17
Author: Nhật — DevOps / Model Agent, GreenNode
Status: DRAFT — internal review

---

## 0. Executive Summary

Node Agent hiện tại chạy theo mô hình **single-orchestrator** (Cursor-style): một chuỗi gather→think→write→critique, tất evidence nhồi vào writer context. Vấn đề chính:
- **Input dilution**: writer (gemma) nhận 9+ chunks, mất neo câu hỏi gốc.
- **Lost-in-the-middle**: câu hỏi nằm đầu context, evidence nằm giữa, answer ở cuối — model lightweight bám cuối mạnh, đầu yếu.
- **No task routing**: mọi request đều chạy gather→think→write dù câu hỏi đơn giản (pricing) hay phức tạp (sánh so).

Giải pháp: chuyển sang **hybrid subagent architecture** (isolated-context worker model) bổ sung **input relevance gate** + **question anchor** + **token-budget routing**.

---

## 1. Reference Systems (tài liệu thật từ E:\System Prompt)

### 1.1 Cursor Agent 2.0
- **Không có subagent**. Mọi work trong một context.
- Strategy: "maximize relevant context" — gather research trước rồi mới code.
- "Split complex tasks into smaller independent sub-tasks" — parallel tool calls.
- **Risk**: context window full → compact (lossy).
- **Lesson cho Node Agent**: front-load research step trước khi generate answer.

### 1.2 Anthropic Claude Code 2.0
- **Task tool** (subagent): "proactively use Task tool when task matches agent description."
- Subagent = **fresh context, clean slate**. Main agent retains orchestration only.
- Parallel Task calls: "send single message with multiple Task tool calls."
- **Lesson cho Node Agent**: writer cần isolated context, KHÔNG nhận toàn bộ evidence raw.

### 1.3 Qodo
- Quest-based decomposition: Design → Action plan.
- **Lesson**: task decomposition theo phases, không dồn hết.

---

## 2. Gap Analysis: Node Agent hiện tại

| Gap | Cursor approach | Anthropic approach | Node Agent hiện tại |
|-----|----------------|-------------------|-------------------|
| Context bloat | Front-load research, compact | Subagent isolation | Nhồi hết vào writer |
| Question drift | "maximize relevant context" | Fresh context per task | Câu hỏi 1 lần trong context |
| Task routing | Split into smaller tasks | Proactive task matching | `need_thinking` binary |
| Tool calls | Parallel batch | Parallel Task | Sequential gather→write |
| Long session | Compact | Subagent reset | No mechanism |

---

## 3. Proposed Architecture (5 Phases)

### Phase 1: Input Relevance Gate (H1)
**Goal**: Lọc evidence LOÃNG trước khi vào writer.
- Sau gather: re-rank chunk theo relevance score với câu hỏi gốc.
- Drop chunk relevance < threshold (configurable, default 0.3).
- Cap sources thực đưa vào writer: max 5-6 chunks (thay vì 9+).
- `_source_rank` hiện tại chỉ dùng official-first; thêm cosine similarity (BM25 score sẵn có).

**Files**: `node_agent/orchestrator.py` (gather section, ~line 1040-1065)
**Verification**: E2E test — Q1-Q3, check `duplicate source URLs: NONE`, check answer không mất data quan trọng.

### Phase 2: Question Anchor + Re-anchor (H2)
**Goal**: Writer KHÔNG BAO GIỜ quên câu hỏi gốc.
- Đặt câu hỏi gốc **đầu + cuối** context (recency effect).
- Thêm "intent anchor" dòng: "Trả lời câu hỏi trên. Không trả lời thêm."
- Writer `fixmsg` (corrective refine) đã có; mở rộng sang write lần đầu.

**Files**: `node_agent/orchestrator.py` (write section, ~line 1140), `node_agent/reason.py` (build_messages)
**Verification**: E2E — answer luôn xoay quanh câu hỏi, không lan man.

### Phase 3: Thinker as Focus Compressor (H3)
**Goal**: Thinker (minimax) nén evidence thành tóm tắt focus, KHÔNG chỉ plan.
- Hiện thinker chỉ log `react_plan` rồi bỏ.
- Nâng lên: thinker output = "compressed evidence summary" (3-5 điểm chính xoay quanh câu hỏi).
- Writer nhận: question + compressed summary (thay raw evidence).
- Giảm token ~40-60% cho writer, giữ signal mạnh.

**Files**: `node_agent/orchestrator.py` (think→write handoff, ~line 993-1010)
**Verification**: Measure writer input token trước/sau; E2E answer quality.

### Phase 4: Token-Budget Routing (H4)
**Goal**: Phân biệt task nặng/nhẹ theo ngưỡng token, không nhồi đều.
- **Light path** (spec/pricing câu đơn): skip thinker, chỉ KB retrieve + write.
- **Medium path** (comparison, how-to): gather + write (no thinker).
- **Heavy path** (multi-product, complex analysis): full gather→think→write.
- Routing dựa vào `triage.answer_shape` + estimated token footprint.
- "2-3-4 orchestrator": phân thành lightweight coordinator + specialized workers.

**Files**: `node_agent/orchestrator.py` (triage section, ~line 896-940), `node_agent/loop_budget.py`
**Verification**: Measure latency; light path < 15s, heavy path 30-90s.

### Phase 5: Dual-Budget Writer (H5)
**Goal**: Output "dài hơn thông minh" mà KHÔNG nhồi nhét.
- Split writer budget: **data** (bảng/số/tên) = terse; **glue** (diễn giải, gợi mở) = expanded.
- Writer prompt: "data sections: compact, exact. Prose sections: can be richer, conversational."
- Tổng answer length controlled by `max_sentences` từ triage.

**Files**: `node_agent/orchestrator.py` (write section prompt), `node_agent/modes.py` (system prompt)
**Verification**: Read answer — data dense, prose rich, không nhồi.

---

## 4. Execution Order (phased delivery)

```
Phase 1+2 (Week 1)  ──→ E2E verify ──→ Commit   [DONE 2026-06-17]
Phase 3 (Week 1-2)  ──→ E2E verify ──→ Commit   [DONE 2026-06-17]
Phase 5 (Week 2)    ──→ E2E verify ──→ Commit   [DONE 2026-06-17]
New tool compare_products (react.py)            [DONE 2026-06-17]
Phase 4 (Week 2)    ──→ E2E verify ──→ Commit   [DONE 2026-06-17 — route_tier light/medium/heavy]
Full regression (Week 2)                        [PENDING]
```

## STATUS 2026-06-17 — Phase 4 done
- route_tier field on Triage: "light" (single pricing/spec → KB-only, skip web),
  "medium" (general/compare-without-thinking → hybrid gather), "heavy" (need_thinking → ReAct).
- _heuristic_triage sets tier; shown in as_note (tier=...) on dashboard trace.
- kb_strong gate now tier-aware: light needs ≥1 official number chunk, medium needs ≥2
  → KB-first skips the ~50s Playwright crawl when the answer already lives in KB.
- E2E latency: Q1 compare(heavy) 40s→21s, Q2 Kubernetes(medium) ~30s, Q3 price(light) 5.6s.
  All verified=True, 0 duplicate URLs, anti-fabrication intact.

## STATUS 2026-06-17
- Phase 1 relevance gate: TOP_K_EVIDENCE=6, MIN_RELEVANCE_FACTOR=0.6 in orchestrator.py. E2E: Q1 9→1 nguồn, 0 dup URL.
- Phase 2 question anchor: reason.py build_messages — câu hỏi đặt đầu, intent anchor "chỉ trả lời đúng câu hỏi trên". E2E: trả lời bám trục.
- Phase 3 thinker compressor: react_plan giờ dedup + bullet 5 điểm (thay raw join). E2E: writer bám tốt, không lan man.
- Phase 5 dual-budget: max_tokens 900/1300 + 200 cho evaluate/explore, 1600 wants_chart. E2E: Q2 đầy đặn 3 nhóm tính năng + diễn giải.
- New tool compare_products: structured KB compare lookup, official-only, trong react._ACTIONS + _SYS prompt.
- Còn lại: Phase 4 (light/medium/heavy routing), regression suite.

Each phase: code → compile check → E2E test → commit → next.
No phase skipped. No phase batched without verify.

---

## 5. Risk & Tradeoff

| Risk | Mitigation |
|------|-----------|
| Phase 1 drops too much evidence | Threshold configurable; verify on E2E |
| Phase 3 thinker compression loses numbers | Thinker receives RAW evidence, outputs summary with numbers preserved |
| Phase 4 routing miscategorizes | Conservative: light path fallback to medium on uncertainty |
| Phase 5 glue expansion too verbose | max_sentences hard cap per answer_shape |

---

## 6. Open Questions for anh Nhật

1. Thinker (minimax-m2.5) có đủ capacity cho Phase 3 compression? Hay cần swap role mapping?
2. Anh muốn max writer token output cho câu trả lời "dài" (heavy) bao nhiêu? 800? 1200?
3. Phase 4 routing threshold: anh muốn hardcode hay config file?
4. Anh muốn plan này patch vào `node-agent-assistant` skill luôn hay giữ riêng?

---

## 7. Files Likely Changed

- `node_agent/orchestrator.py` — main pipeline (phases 1-4)
- `node_agent/reason.py` — message building (phase 2)
- `node_agent/quality.py` — verify gate (phase 3)
- `node_agent/roles.py` — role mapping (phase 4)
- `node_agent/loop_budget.py` — token budget (phase 4)
- `node_agent/modes.py` — system prompt (phase 5)
- `scripts/e2e_direct.py` — test script (already exists)
