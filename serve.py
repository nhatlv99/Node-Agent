#!/usr/bin/env python3
"""Launcher for the Node Agent Assistant dashboard.

Reads the LLM credentials from Nhật's Hermes config IN-PROCESS (never echoed,
never exported to the shell) and wires Tier-0 env for the FastAPI app, then
runs uvicorn. Uses the stand-in gateway model until the VNG MaaS key lands.

Run:
    hermes-fork/.venv/bin/python serve.py [--port 8077] [--model xapi3/kr/claude-opus-4.6]

The dashboard token defaults to 'demo-key-change-me' (override --token).
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

# ruamel/pyyaml both ship in the hermes-fork venv; prefer stdlib-ish yaml.
import yaml

HERMES_CONFIG = os.environ.get(
    "HERMES_CONFIG", "/mnt/e/Hermes/data/config.yaml"
)

# VNG Cloud MaaS — the CONTEST endpoint (OpenAI-compatible, vLLM backend).
# This is prod: gemma-4-31b-it / qwen3-5-27b / minimax-m2.5 served from here.
MAAS_BASE_URL = "https://maas-llm-aiplatform-hcm.api.vngcloud.vn/v1"


def _read_key_from_file(path: str) -> str:
    """Read a bearer key from a file (first non-empty line). Never logged."""
    try:
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            s = line.strip()
            if s:
                return s
    except OSError:
        pass
    return ""


def _load_maas_key() -> str:
    """Resolve the MaaS API key WITHOUT hardcoding it in the repo.

    Precedence (in-process only, never echoed, never exported to a shell):
      1. NODE_AGENT_API_KEY        (env — preferred for CI / VPS deploy)
      2. NODE_AGENT_KEY_FILE       (path to a key file, e.g. the Downloads/Key)
      3. ~/.node_agent_maas_key    (a local untracked file)
    """
    k = os.environ.get("NODE_AGENT_API_KEY", "").strip()
    if k:
        return k
    kf = os.environ.get("NODE_AGENT_KEY_FILE", "").strip()
    if kf:
        k = _read_key_from_file(kf)
        if k:
            return k
    default_kf = Path.home() / ".node_agent_maas_key"
    if default_kf.exists():
        return _read_key_from_file(str(default_kf))
    return ""


def _load_gateway_creds() -> tuple[str, str]:
    """Return (base_url, api_key) from the Hermes config's model block.

    Dev-only fallback (the in-house gateway stand-in). Read in-process only.
    """
    cfg = yaml.safe_load(Path(HERMES_CONFIG).read_text(encoding="utf-8"))
    model = cfg.get("model", {})
    base = model.get("base_url", "")
    key = model.get("api_key", "")
    if not base or not key:
        raise SystemExit(
            f"No base_url/api_key in {HERMES_CONFIG} model block. "
            "Set NODE_AGENT_BASE_URL/API_KEY/MODEL manually instead."
        )
    return base, key


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8077)
    ap.add_argument("--host", default="127.0.0.1")
    # Default WRITER model on MaaS (drives NODE_AGENT_MODEL single-fallback).
    # Per-seat models come from roles._DEV_DEFAULTS below.
    ap.add_argument("--model", default="google/gemma-4-31b-it")
    ap.add_argument("--token", default="demo-key-change-me")
    ap.add_argument(
        "--no-llm",
        action="store_true",
        help="serve retrieve-only (skip reading creds)",
    )
    ap.add_argument(
        "--dev-gateway",
        action="store_true",
        help="use the in-house gateway stand-in (Hermes config) instead of VNG MaaS",
    )
    args = ap.parse_args()

    if not args.no_llm:
        if args.dev_gateway:
            # Dev path: in-house gateway stand-in (opus seats from Hermes config).
            base, key = _load_gateway_creds()
        else:
            # PROD path: VNG Cloud MaaS (the contest endpoint). Key resolved
            # from env / key-file — never hardcoded in the repo.
            base = MAAS_BASE_URL
            key = _load_maas_key()
            if not key:
                raise SystemExit(
                    "No MaaS key. Set NODE_AGENT_API_KEY, or NODE_AGENT_KEY_FILE="
                    "/path/to/key, or ~/.node_agent_maas_key (or run --dev-gateway)."
                )
        os.environ["NODE_AGENT_BASE_URL"] = base
        os.environ["NODE_AGENT_API_KEY"] = key
        os.environ["NODE_AGENT_MODEL"] = args.model
        print(f"[serve] LLM wired: {args.model} @ {base} (key hidden)")
        # Per-seat wiring for the 3-model harness. In dev we map each logical
        # role to its gateway stand-in (roles._DEV_DEFAULTS: haiku/opus/sonnet)
        # so the Kanban board shows three DISTINCT models cooperating. On the
        # VPS, set NODE_AGENT_MODEL_{ORCHESTRATOR,THINKER,WRITER} to the real
        # VNG MaaS names and these dev defaults are ignored (env wins).
        from node_agent import roles as _roles
        for _role, _envkey in _roles._ENV_KEY.items():
            if not os.environ.get(_envkey):
                os.environ[_envkey] = _roles._DEV_DEFAULTS[_role]
            print(f"[serve] seat {_role:<13} → {os.environ[_envkey]}")
    else:
        print("[serve] retrieve-only mode (no LLM)")

    os.environ["NODE_AGENT_DASH_TOKEN"] = args.token

    # Auto-wire SearXNG if a local instance is reachable (the live search
    # backend). Falls back to KB-seeded live crawl when not set / not up.
    if not os.environ.get("NODE_AGENT_SEARXNG_URL"):
        os.environ["NODE_AGENT_SEARXNG_URL"] = "http://localhost:8888"
    print(f"[serve] SearXNG: {os.environ['NODE_AGENT_SEARXNG_URL']}")

    import uvicorn

    print(f"[serve] dashboard → http://{args.host}:{args.port}  token={args.token}")
    uvicorn.run("node_agent.api:app", host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
