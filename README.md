# Node Agent Assistant

AI-powered customer support assistant for GreenNode — answers questions about High-Performance Cloud, AI Platform, and Intelligent Automation with cited sources and auto-generated charts.

---

## The Problem

GreenNode's customers — engineers, DevOps teams, and technical decision-makers evaluating cloud infrastructure — frequently need fast, accurate answers about GPU specs, pricing, SLA terms, and platform capabilities. This information is spread across documentation, pricing pages, and product guides, making it time-consuming to find and easy to misinterpret.

Support teams face repetitive queries that slow down response times, while customers face delays when they need to make procurement or architecture decisions quickly.

## Who It's For

- **Technical evaluators** comparing GreenNode GPU instances, AI Platform features, or automation services against alternatives
- **Existing customers** looking up SLA commitments, model availability, or billing details without opening a ticket
- **GreenNode support staff** who need a consistent, source-backed reference to handle inquiries faster

## Value Delivered

- **Instant, cited answers** — every response references the exact KB source, so users can verify claims rather than take them on faith
- **Data made visual** — numerical comparisons (pricing, VRAM, bandwidth, SLA) are rendered as charts automatically, reducing cognitive load
- **No hallucinated specs** — the bot only cites numbers that exist in the knowledge base; if the data isn't there, it says so
- **Always available** — runs 24/7 on VNG Cloud AgentBase without requiring a support agent on standby

---

## What It Does

Node Agent Assistant handles customer inquiries across GreenNode's product lines using a 3-model pipeline:

- **Orchestrator** — understands the question, plans the retrieval strategy
- **Thinker** — reasons over retrieved knowledge base chunks
- **Writer** — composes the final answer with inline citations `[n]` and chart recommendations

Every answer follows a consistent structure: context confirmation → data table or chart (when numbers are available) → brief analysis → suggested next step. The bot never fabricates figures — if the knowledge base lacks data for a chart, it says so.

Supported chart types: bar, horizontal bar, gauge, radar, donut.

---

## Deployment

The assistant runs on **VNG Cloud AgentBase** and is deployed via Docker Hub.

To ship a new version:

```bash
export DOCKER_USER=your_dockerhub_username
./scripts/build_push.sh          # builds and pushes :latest
./scripts/build_push.sh v1.1     # also tags a pinned version
```

Then go to **AgentBase dashboard → service → Redeploy** to pull the new image.

Required environment variables on AgentBase:

| Variable | Description |
|---|---|
| `NODE_AGENT_API_KEY` | VNG Cloud MaaS API key |
| `NODE_AGENT_DASH_TOKEN` | Dashboard access token |
| `NODE_AGENT_MODEL` | Writer model (default: `google/gemma-4-31b-it`) |

Container port: `8080` — health check: `/health`.

---

## Interface

The UI is fully responsive across desktop, iPad, and mobile. A conversation sidebar keeps session history, and a live Harness monitor shows the 3-model pipeline in real time as each answer is generated.

---

## Data

Knowledge base: `data/kb_chunks.jsonl` — 2,716 chunks crawled from greennode.ai and vngcloud documentation.
