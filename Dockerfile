FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY node_agent/ ./node_agent/
COPY data/kb_chunks.jsonl ./data/kb_chunks.jsonl
COPY serve.py .

# AgentBase requires port 8080 and health check at /health
EXPOSE 8080

ENV NODE_AGENT_DASH_TOKEN=demo-key-change-me

ENTRYPOINT ["python", "serve.py", "--port", "8080", "--host", "0.0.0.0"]
