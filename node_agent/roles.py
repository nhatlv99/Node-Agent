"""Tier 0.5 ROLE HARNESS — Node Agent Assistant (multi-model orchestration).

WHY FILE EXISTS
---------------
The production deployment runs THREE light models via VNG Cloud MaaS
(qwen/qwen3-5-27b, minimax/minimax-m2.5, google/gemma-4-31b-it).

No single light model is strong at everything; don't pick one — give
each ROLE what it's good at and let them work as a team ("harness").
The harness talks in LOGICAL ROLES, not raw model names. Deploy by
remapping roles → models in ONE place (env vars). Nothing else in
the codebase should mention a model id directly.

 LOGICAL ROLE    JOB                          PROD (VNG Cloud MaaS)
 ──────────────────────────────────────────────────────────────────────
 orchestrator    triage input, critique output  qwen/qwen3-5-27b
 thinker         deep reasoning when needed     minimax/minimax-m2.5
 writer          grounded final answer           google/gemma-4-31b-it

DEPLOY: set 3 env vars on the VPS, code untouched:
  NODE_AGENT_MODEL_ORCHESTRATOR=qwen/qwen3-5-27b
  NODE_AGENT_MODEL_THINKER=minimax/minimax-m2.5
  NODE_AGENT_MODEL_WRITER=google/gemma-4-31b-it

If a role env var is unset, falls back to NODE_AGENT_MODEL (single-model
mode) — useful for cheap demos when only one MaaS model is provisioned.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

# Logical roles (the ONLY names the rest of the code should use) ───
ORCHESTRATOR = "orchestrator"  # default driver: input triage, output critique loop
THINKER = "thinker"           # deep reasoning, invoked only when triage decides
WRITER = "writer"             # grounded final-answer generation

ROLES = (ORCHESTRATOR, THINKER, WRITER)

# Production defaults (VNG Cloud MaaS models) ──────────────────────
_DEFAULTS = {
    ORCHESTRATOR: "qwen/qwen3-5-27b",
    THINKER: "minimax/minimax-m2.5",
    WRITER: "google/gemma-4-31b-it",
}

# Env var names per role (priority order) ──────────────────────────
_ENV_MAP = {
    ORCHESTRATOR: ("NODE_AGENT_MODEL_ORCHESTRATOR",),
    THINKER: ("NODE_AGENT_MODEL_THINKER",),
    WRITER: ("NODE_AGENT_MODEL_WRITER",),
}


@dataclass(frozen=True)
class Seat:
    role: str
    model: str


def resolve(role: str) -> Seat:
    """Resolve a logical role to its concrete model id.

    Priority: role-specific env var > NODE_AGENT_MODEL > _DEFAULTS.
    """
    for env_name in _ENV_MAP.get(role, ()):
        v = os.environ.get(env_name, "").strip()
        if v:
            return Seat(role=role, model=v)

    v = os.environ.get("NODE_AGENT_MODEL", "").strip()
    if v:
        return Seat(role=role, model=v)

    return Seat(role=role, model=_DEFAULTS.get(role, ""))


def seat_map() -> dict[str, Seat]:
    """Return {role: Seat} for all logical roles."""
    return {r: resolve(r) for r in ROLES}
