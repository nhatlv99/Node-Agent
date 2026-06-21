#!/usr/bin/env python3
"""Launcher for Node Agent Assistant dashboard.

VNG Cloud MaaS only. No external gateway fallback, no alternate dev path.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import yaml

MAAS_BASE_URL = "https://maas-llm-aiplatform-hcm.api.vngcloud.vn/v1"
_ENV_KEY_NAMES = ("NODE_AGENT_API_KEY", "MAAS_API_KEY", "AI_PLATFORM_API_KEY", "API_KEY")


def _read_key_from_file(path: str) -> str:
    try:
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            if "=" in s and not s.startswith("="):
                return s.split("=", 1)[1].strip().strip("'\"")
            return s.strip("'\"")
    except OSError:
        pass
    return ""


def _load_dotenv(start: Path) -> None:
    for cand in (start / ".env", Path.cwd() / ".env"):
        if not cand.exists():
            continue
        try:
            for line in cand.read_text(encoding="utf-8").splitlines():
                s = line.strip()
                if not s or s.startswith("#") or "=" not in s:
                    continue
                key, val = s.split("=", 1)
                key = key.strip()
                val = val.strip().strip('"\'')
                if key and key not in os.environ:
                    os.environ[key] = val
        except OSError:
            pass
        break


def _load_maas_key() -> str:
    for name in _ENV_KEY_NAMES:
        v = os.environ.get(name, "").strip()
        if v:
            return v

    key_file = os.environ.get("NODE_AGENT_KEY_FILE", "").strip()
    if key_file:
        v = _read_key_from_file(key_file)
        if v:
            return v

    default_kf = Path.home() / ".node_agent_maas_key"
    if default_kf.exists():
        v = _read_key_from_file(str(default_kf))
        if v:
            return v
    return ""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8077)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--model", default="google/gemma-4-31b-it")
    ap.add_argument("--token", default="demo-key-change-me")
    ap.add_argument("--no-llm", action="store_true", help="serve retrieve-only")
    ap.add_argument("--single-model", action="store_true",
                    help="Route ALL three seats to --model (demo/cheap). Default: 3 prod MaaS seats.")
    args = ap.parse_args()

    _load_dotenv(Path(__file__).resolve().parent)

    if not args.no_llm:
        key = _load_maas_key()
        if not key:
            raise SystemExit("No MaaS key. Set NODE_AGENT_KEY_FILE or NODE_AGENT_API_KEY.")
        os.environ["NODE_AGENT_BASE_URL"] = MAAS_BASE_URL
        os.environ["NODE_AGENT_API_KEY"] = key
        os.environ["NODE_AGENT_MODEL"] = args.model
        # Wire the THREE production seats (qwen orchestrator / minimax thinker /
        # gemma writer). Without this, roles.resolve() falls back to the single
        # NODE_AGENT_MODEL for every seat -> 100% of traffic hits one model.
        # Respect any seat env already exported; only fill the gaps.
        from node_agent import roles as _R
        if not args.single_model:
            for _role in _R.ROLES:
                _envname = _R._ENV_MAP[_role][0]
                if not os.environ.get(_envname, "").strip():
                    os.environ[_envname] = _R._DEFAULTS[_role]
            _seats = {r: _R.resolve(r).model for r in _R.ROLES}
            print(f"[serve] seats orchestrator={_seats['orchestrator']} "
                  f"thinker={_seats['thinker']} writer={_seats['writer']}")
        else:
            print(f"[serve] single-model mode: all seats -> {args.model}")
        print(f"[serve] MaaS {MAAS_BASE_URL}")
        print(f"[serve] model {args.model}")
    else:
        print("[serve] retrieve-only mode (no LLM)")

    os.environ["NODE_AGENT_DASH_TOKEN"] = args.token
    if not os.environ.get("NODE_AGENT_SEARXNG_URL"):
        os.environ["NODE_AGENT_SEARXNG_URL"] = "http://localhost:8888"
        print(f"[serve] SearXNG: {os.environ['NODE_AGENT_SEARXNG_URL']}")

    import uvicorn

    print(f"[serve] dashboard http://{args.host}:{args.port} token={args.token}")
    uvicorn.run("node_agent.api:app", host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
