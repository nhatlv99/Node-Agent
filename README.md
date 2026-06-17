# Node Agent Assistant

Trợ lý AI hỗ trợ khách hàng GreenNode — pipeline 3 model (qwen điều phối · minimax suy luận · gemma soạn) chạy trên VNG Cloud MaaS, trả lời có trích dẫn [n] và tự dựng biểu đồ từ số liệu thật trong KB.

## Yêu cầu

- Python 3.12+
- 1 API key VNG Cloud MaaS (OpenAI-compatible endpoint)

## Cài đặt (macOS)

```bash
cd "Node Agent Src"
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

(Tùy chọn) Nếu muốn crawl helpdesk live (render SPA), cài thêm Playwright. Bỏ qua cũng được — app tự fallback về trả lời từ KB:

```bash
pip install playwright
python -m playwright install chromium
```

## Cấu hình API key (KHÔNG hardcode trong repo)

serve.py đọc key MaaS theo thứ tự ưu tiên, in-process, không bao giờ echo ra shell:

1. Biến môi trường `NODE_AGENT_API_KEY` (ưu tiên cho deploy)
2. `NODE_AGENT_KEY_FILE=/đường/dẫn/tới/key.txt` (file 1 dòng chứa key)
3. `~/.node_agent_maas_key` (file local, đã nằm trong .gitignore)

Cách nhanh nhất:

```bash
export NODE_AGENT_API_KEY="<dán-key-MaaS-vào-đây>"
```

## Chạy

```bash
source .venv/bin/activate
python serve.py --port 8077
```

Mở http://127.0.0.1:8077 — token đăng nhập mặc định `demo-key-change-me` (đổi bằng `--token`).

## Câu hỏi thử (để thấy đủ các loại chart)

| Câu hỏi | Chart |
|---|---|
| So sánh VRAM và băng thông H100 với H200 | bar |
| Xếp hạng giá thuê các dòng GPU GreenNode | hbar |
| Mức cam kết SLA uptime của GreenNode | gauge 99.99% |
| Hồ sơ năng lực đa tiêu chí của H100 | radar |
| Chi phí và sức mạnh của H100 | donut + radar |

## Cấu trúc câu trả lời

Mỗi câu trả lời theo khung: mở đầu (xác nhận đang giúp gì) → bảng/dữ liệu + biểu đồ (nếu có số liệu phù hợp) → diễn giải ngắn → mời bước tiếp. Bot KHÔNG bịa số: nếu KB không có dữ liệu phù hợp cho 1 loại biểu đồ, nó nói rõ và không vẽ.

## Tùy chọn nâng cao

- `--dev-gateway` — dùng gateway nội bộ (đọc từ Hermes config) thay vì MaaS. Chỉ dành cho môi trường dev gốc, không dùng trên máy khác.
- `--no-llm` — chạy chế độ chỉ truy hồi KB (không gọi LLM).
- `--model` — đổi model writer mặc định (mặc định `google/gemma-4-31b-it`).
- Per-seat: đặt `NODE_AGENT_MODEL_ORCHESTRATOR` / `_THINKER` / `_WRITER` để override model từng seat.

## Dữ liệu

- `data/kb_chunks.jsonl` — KB thật (2716 chunk, crawl từ greennode.ai / vngcloud).
- File `*.db` (convo / memory / trace) tự sinh khi chạy, đã nằm trong .gitignore.
