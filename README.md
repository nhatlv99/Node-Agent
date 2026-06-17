# Node Agent Assistant

AI assistant for GreenNode customers — 3-model pipeline (qwen orchestrates · minimax reasons · gemma writes) running on VNG Cloud MaaS, with cited answers `[n]` and auto-generated charts from real KB data.

Fully responsive UI (desktop, iPad, mobile). Deployed on VNG Cloud AgentBase via Docker Hub.

## Requirements

- Python 3.12+
- VNG Cloud MaaS API key (OpenAI-compatible endpoint)

## Setup (macOS)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Optional — only needed for live helpdesk crawling (SPA rendering). The app falls back to KB-only answers without it:

```bash
pip install playwright
python -m playwright install chromium
```

## API Key Configuration

The key is read in-process and never echoed to the shell or committed to the repo.

**Local dev — use `.env`** (recommended):

```bash
cp .env.example .env
# open .env and paste your key into NODE_AGENT_API_KEY=
```

**Production (AgentBase) — environment variable:**

Set `NODE_AGENT_API_KEY` directly in the AgentBase dashboard (see Deploy section below).

`.env` is in `.gitignore` and will never be committed.

## Running

```bash
source .venv/bin/activate
python serve.py --port 8077
```

Open http://127.0.0.1:8077 — default dashboard token is `demo-key-change-me` (override with `--token`).

## Sample Questions (to see all chart types)

| Question | Chart |
|---|---|
| Compare VRAM and bandwidth of H100 vs H200 | bar |
| Rank GreenNode GPU rental prices | hbar |
| GreenNode SLA uptime commitment | gauge 99.99% |
| H100 multi-criteria capability profile | radar |
| H100 cost vs performance | donut + radar |

## Answer Structure

Each answer follows: opening (confirms what it's helping with) → table/data + chart (when KB has matching numbers) → brief analysis → next-step prompt. The bot never fabricates numbers — if the KB has no suitable data for a chart type, it says so and skips the chart.

## Advanced Options

- `--dev-gateway` — use the internal gateway (reads from Hermes config) instead of MaaS. Dev environment only, not for use on other machines.
- `--no-llm` — KB retrieval-only mode (no LLM calls).
- `--model` — override the default writer model (default: `google/gemma-4-31b-it`).
- Per-seat: set `NODE_AGENT_MODEL_ORCHESTRATOR` / `_THINKER` / `_WRITER` to override each seat's model.

## Deploy (Docker Hub → VNG Cloud AgentBase)

### Build and push image

```bash
# First time: log in to Docker Hub
docker login
export DOCKER_USER=your_dockerhub_username

# Each time you want to deploy a new build
./scripts/build_push.sh          # build + push :latest
./scripts/build_push.sh v1.1     # also push a pinned version tag
```

The script builds with `--platform linux/amd64` for compatibility with x86 servers when building on Apple Silicon.

### Pull the new image on AgentBase

Go to **VNG Cloud AgentBase dashboard** → select the service → click **Redeploy** (or **Update image**). AgentBase will pull the latest image from Docker Hub automatically.

Environment variables to configure on AgentBase:

| Variable | Value |
|---|---|
| `NODE_AGENT_API_KEY` | VNG MaaS API key |
| `NODE_AGENT_DASH_TOKEN` | Dashboard login token |
| `NODE_AGENT_MODEL` | `google/gemma-4-31b-it` (default) |

Container exposes port `8080`, health check at `/health`.

## Responsive UI

The UI is fully responsive across all screen sizes:

- **Desktop** — full layout, conversation sidebar, Harness monitor
- **iPad (481–1024px)** — compact composer, kanban panel auto-scales to screen width
- **Mobile (≤ 480px)** — 28px title, dock with `safe-area-inset-bottom` for iPhone home bar, full-width kanban panel anchored to bottom

## Data

- `data/kb_chunks.jsonl` — production KB (2716 chunks, crawled from greennode.ai / vngcloud).
- `*.db` files (conversations, memory, traces) are auto-generated at runtime and listed in `.gitignore`.
