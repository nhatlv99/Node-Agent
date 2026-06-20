# Token-Control Plan — Complexity-Aware Budgeting cho Node Agent

Date: 2026-06-17
Author: Nhật — DevOps / Model Agent, GreenNode
Status: DESIGN — chờ duyệt trước khi code

---

## 0. Đề bài (anh Nhật đặt)

Kiểm soát output token của TOÀN BỘ LLM trong pipeline dựa trên độ phức tạp
câu hỏi. Mỗi câu cần: (a) đánh dấu mức thinking (low/medium/high), (b) cấp
ngân sách output theo chặng 4000 / 8000 / 12000 / 16000 token, HOẶC xuất ra
một con số token cụ thể, (c) tôn trọng TÍNH CHẤT từng model (qwen tắt được
thinking, minimax không tắt được → tốn token, gemma không reasoning).

### 4 vsignature anh đưa (em phân loại lại theo góc model lớn)

| # | Câu hỏi | Thinking | Output | Lý do |
|---|---------|----------|--------|-------|
| 1 | "GreenNode có những dịch vụ gì?" | HIGH | MEDIUM | Chung chung → phải đoán ý người dùng, bao quát nhiều nhóm dịch vụ, nhưng câu trả lời không cần quá dài |
| 2 | "So sánh 1 đối tượng với NHIỀU đối tượng" | HIGH | HIGH | Loop qua từng trang, thu thập + summary từng đối tượng rồi mới so → vừa nghĩ nhiều vừa viết dài |
| 3 | "So sánh H100 và H200" | LOW | MEDIUM | 2 đối tượng cố định, doc có sẵn bảng → chỉ search + summary, không cần loop nhiều |
| 4 | "GreenNode có bao nhiêu instance flavor" | MEDIUM→HIGH | MEDIUM→HIGH | Có 2 chế độ trả lời: (4a) liệt kê thẳng từ data column [medium]; (4b) khai thác toàn bộ flavor, lập bảng + chart + phân tích từng gói [high] |

Nhận định cốt lõi: **thinking và output là HAI TRỤC ĐỘC LẬP.** Câu 1 nghĩ nhiều
viết vừa; câu 3 nghĩ ít viết vừa; câu 2 nghĩ nhiều viết nhiều. Pipeline hiện tại
chỉ có 1 trục (`route_tier` light/medium/heavy gộp chung) → cần tách.

---

## 1. Hiện trạng (đã đọc code)

- `route_tier` (Phase 4 vừa làm): light/medium/heavy — gộp cả thinking lẫn gather.
- `need_thinking`: bool — bật ReAct hay không.
- `max_sentences`: trần câu phần DATA (heuristic theo intent).
- Output token hiện HARDCODE ở writer: 900 / 1300 / +200 / 1600 (Phase 5).
- Thinker (minimax) hardcode `max_tokens=1200` mỗi ReAct round.
- `_THINK_OFF`: qwen tắt thinking được; minimax KHÔNG (luôn tốn ~200+ token CoT ẩn);
  gemma không reasoning.
- `LoopBudget`: đếm SỐ LẦN gọi LLM (GATHER_MAX=3, REFINE_MAX=2, TOTAL_LLM_MAX=8).
  KHÔNG đếm token — chỉ đếm số call.

Gap: pipeline đếm "bao nhiêu lần gọi" nhưng KHÔNG kiểm soát "mỗi lần tốn bao nhiêu
token", và không tách thinking-budget khỏi output-budget.

---

## 2. Thiết kế: hai trục + bảng tra token

### 2.1 Trục THINKING (think_level) — quyết định gather depth + thinker token

| think_level | Khi nào | GATHER_MAX | Thinker max_tokens/round | enable_thinking |
|-------------|---------|-----------|--------------------------|-----------------|
| `none`  | pricing/spec đơn (Câu... light) | 0 (KB-first) | — không gọi thinker | n/a |
| `low`   | compare 2 đối tượng có sẵn doc (Câu 3) | 1 | 800 | qwen: off |
| `medium`| liệt kê có cấu trúc (Câu 4a) | 2 | 1200 | qwen: off |
| `high`  | chung chung / 1-vs-nhiều / phân tích sâu (Câu 1, 2, 4b) | 3 | 1600 | minimax giữ thinking |

### 2.2 Trục OUTPUT (output_band) — quyết định writer max_tokens theo chặng

Anh muốn chặng 4000/8000/12000/16000. Đây là token chặng cho TỔNG output budget
một lượt (không phải chỉ 1 câu). Map:

| output_band | Token ceiling | Dùng cho | Hình dạng |
|-------------|--------------|----------|-----------|
| `S` (small)  | 4000  | meta, pricing 1 dòng | đoạn ngắn / 1 bảng nhỏ |
| `M` (medium) | 8000  | Câu 1, Câu 3 | bảng + 2-3 câu diễn giải |
| `L` (large)  | 12000 | Câu 2, Câu 4a | nhiều bảng / liệt kê dài + glue |
| `XL` (xlarge)| 16000 | Câu 4b (bảng + chart + phân tích sâu từng gói) | bảng + nhiều chart + phân tích |

QUAN TRỌNG: 4000-16000 là ngân sách TOÀN LƯỢT chia cho các seat, KHÔNG phải
max_tokens 1 call. Một câu XL có thể tiêu: thinker 3×1600 + writer 4000 + refine
2×2000 ≈ 12800 → nằm trong band 16000.

### 2.3 Hàm xuất token cụ thể (anh hỏi "có thể xuất giá trị token cần thiết không")

Có. Thay vì chỉ trả band, ta trả luôn con số ước lượng dựa tính chất model:

```
def estimate_token_budget(think_level, output_band, seats) -> TokenPlan:
    # writer (gemma, non-reasoning): output ≈ visible answer, 1:1
    writer_out = {"S":700, "M":1300, "L":2200, "XL":3500}[output_band]
    # thinker (minimax, thinking không tắt): cộng overhead CoT ẩn ~40%
    think_per_round = {"none":0,"low":800,"medium":1200,"high":1600}[think_level]
    think_rounds   = {"none":0,"low":1,"medium":2,"high":3}[think_level]
    thinker_total  = int(think_per_round * think_rounds * 1.4)  # minimax CoT tax
    # orchestrator (qwen, thinking OFF): triage + judge, rẻ
    orch_total = 120 + 400  # triage JSON + 1 G-Eval judge
    # refine: tối đa REFINE_MAX, mỗi lần ≈ writer_out
    refine_total = writer_out * REFINE_MAX
    total = writer_out + thinker_total + orch_total + refine_total
    return TokenPlan(writer_out, thinker_total, orch_total, refine_total,
                     total=total, band=output_band)
```

→ Pipeline có thể LOG ra dashboard: "câu này dự chi ~X token, band Y". Anh nhìn
thấy con số thật trên trace.

---

## 3. Phân loại 4 câu qua hệ thống mới

| Câu | think_level | output_band | est. total token |
|-----|-------------|-------------|------------------|
| 1. dịch vụ gì (chung chung) | high | M (8000) | ~1300 + 3×1600×1.4 + 520 + 2600 ≈ 11140 |
| 2. 1-vs-nhiều | high | L (12000) | ~2200 + 6720 + 520 + 4400 ≈ 13840 → clamp/cảnh báo |
| 3. H100 vs H200 | low | M (8000) | ~1300 + 800×1.4 + 520 + 2600 ≈ 5540 |
| 4a. liệt kê flavor | medium | L (12000) | ~2200 + 1200×2×1.4 + 520 + 4400 ≈ 10480 |
| 4b. flavor + chart + phân tích sâu | high | XL (16000) | ~3500 + 6720 + 520 + 7000 ≈ 17740 → clamp 16000 |

Quan sát: Câu 2 và 4b chạm trần band → cần (a) clamp writer/refine, hoặc (b)
nâng band, hoặc (c) cắt refine xuống 1. Đây là điểm cần anh quyết policy.

---

## 4. Cách quyết think_level và output_band (classifier)

Hai tầng, rẻ trước đắt sau:

1. **Heuristic (0 token):** mở rộng `_heuristic_triage`. Tín hiệu:
   - "những ... gì / là gì / gồm gì" + không tên sản phẩm cụ thể → think=high, band=M (Câu 1)
   - "so sánh" + đúng 2 đối tượng tên cụ thể → think=low, band=M (Câu 3)
   - "so sánh" + "tất cả / các / nhiều" hoặc >2 đối tượng → think=high, band=L (Câu 2)
   - "bao nhiêu / liệt kê / danh sách" + "instance/flavor/gói" → think=medium, band=L (Câu 4a)
   - kèm "phân tích / chi tiết / chart / biểu đồ / từng gói" → think=high, band=XL (Câu 4b)

2. **LLM classifier (≤120 token, chỉ khi heuristic mơ hồ):** mở rộng `_llm_triage`
   trả thêm 2 field `think_level` + `output_band`. Chạy bằng qwen (thinking OFF →
   rẻ, nhanh). Đây là TOOLCALL MỚI cần code: `classify_complexity`.

---

## 5. Toolcall / thay đổi cần code (tôn trọng "thiếu toolcall thì code thêm")

| # | Thành phần | File | Loại |
|---|-----------|------|------|
| T1 | Field `think_level`, `output_band` trên Triage | orchestrator.py | data model |
| T2 | `estimate_token_budget()` + dataclass `TokenPlan` | loop_budget.py | logic mới |
| T3 | Mở rộng `_heuristic_triage` set 2 trục | orchestrator.py | logic |
| T4 | Mở rộng `_llm_triage` trả think_level+output_band (qwen, thinking off) | orchestrator.py | LLM call |
| T5 | Thinker max_tokens động theo think_level (thay hardcode 1200) | react.py, orchestrator.py | wiring |
| T6 | Writer max_tokens động theo output_band (thay Phase 5 hardcode) | orchestrator.py | wiring |
| T7 | `LoopBudget` thêm token accounting (đếm token thật, không chỉ count call) | loop_budget.py | logic |
| T8 | Emit TokenPlan ra dashboard trace (anh nhìn con số thật) | orchestrator.py, api.py | observability |

Toolcall mới rõ ràng: **T4 `classify_complexity`** (LLM phân loại 2 trục) và
**T2 `estimate_token_budget`** (hàm xuất token cụ thể anh yêu cầu).

---

## 6. Tôn trọng tính chất model (token tax theo seat)

| Seat | Model | enable_thinking | Token đặc tính | Hệ quả budget |
|------|-------|-----------------|----------------|---------------|
| orchestrator | qwen3-5-27b | OFF được | rẻ, output≈visible | triage/judge giá thấp, dùng cho classifier |
| thinker | minimax-m2.5 | KHÔNG tắt | +~40% CoT ẩn | nhân 1.4 khi ước lượng; cap rounds chặt |
| writer | gemma-4-31b-it | non-reasoning | output 1:1 visible | band map thẳng vào max_tokens |

→ Công thức ước lượng (mục 2.3) đã nhúng các tax này. Đây là "dựa vào tính chất
từng LLM biết trước" anh nói.

---

## 7. Execution order (nếu anh duyệt)

```
Bước 1: T1+T2 (data model + estimate_token_budget)        → unit test, in con số
Bước 2: T3 (heuristic 2 trục) + phân loại 4 câu mẫu        → assert đúng bảng mục 3
Bước 3: T5+T6 (wiring thinker/writer max_tokens động)      → E2E 4 câu, đo token thật
Bước 4: T4 (classify_complexity LLM, qwen)                 → E2E câu mơ hồ
Bước 5: T7+T8 (token accounting + dashboard emit)          → anh nhìn trace
Bước 6: regression toàn bộ intent
```

Mỗi bước: code → py_compile → E2E thật trên VNG MaaS → commit. Không batch.

---

## 8. Câu hỏi cần anh quyết trước khi code

1. **Chặng cố định hay con số động?** Anh muốn output clamp đúng vào 4000/8000/
   12000/16000 (4 nấc cứng), hay muốn pipeline xuất con số token động (mục 2.3) rồi
   chỉ dùng 4 nấc làm trần an toàn? (Em nghiêng phương án 2: số động + nấc làm ceiling.)
2. **Câu chạm trần (Câu 2, 4b):** clamp writer, hay cho nâng band, hay cắt refine xuống 1?
3. **Classifier:** chỉ heuristic (rẻ, 0 token) hay luôn gọi qwen classify (chính xác hơn,
   tốn ~120 token/câu)? Em đề xuất hybrid: heuristic trước, qwen chỉ khi mơ hồ.
4. **TOTAL_LLM_MAX** hiện 8 (đếm call). Có giữ song song với token-ceiling mới, hay
   thay hẳn bằng token budget?
