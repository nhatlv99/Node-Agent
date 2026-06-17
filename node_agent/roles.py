"""Tier 0.5 — ROLE HARNESS for Node Agent Assistant (multi-model orchestration).

WHY THIS FILE EXISTS
--------------------
The production deployment runs THREE light open-weight models on VNG MaaS:

    gemma-4-31b-it      ·  qwen-3.5-24b      ·  minimax-m2.5

No single light model is strong at everything, so we don't pick one — we give
each a ROLE it's good at and let them work as a team (a "harness"). The harness
talks in LOGICAL ROLES, never raw model names. Deploy = remap roles → models in
ONE place (env or this file). Nothing else in the codebase mentions a model id.

    LOGICAL ROLE         JOB                                  DEV STAND-IN    →  PROD (VNG MaaS)
    ────────────────────────────────────────────────────────────────────────────────────────
    orchestrator (def.)  triage input + critique output loop  haiku-4.5       →  minimax-m2.5
    thinker              deep reasoning when triage says so    opus-4.6        →  qwen-3.5-24b
    writer               ground the final grounded answer      sonnet (gemma)  →  gemma-4-31b-it

Anh's mapping (2026-06-15):
    opus 4.6 (thinking)  = Qwen 3.5 24b     → ROLE_THINKER
    haiku (default)      = Minimax 2.5      → ROLE_ORCHESTRATOR  (default driver)
    sonnet               = Gemma 4 31b      → ROLE_WRITER

DEPLOY: set 3 env vars on the VPS, code untouched:
    NODE_AGENT_MODEL_ORCHESTRATOR=minimax/minimax-m2.5
    NODE_AGENT_MODEL_THINKER=qwen/qwen3-5-24b
    NODE_AGENT_MODEL_WRITER=google/gemma-4-31b-it

If a role env is unset it falls back to NODE_AGENT_MODEL (single-model mode), so
the harness degrades to "one model does everything" with zero config — useful
for a cheap demo or when only one MaaS model is provisioned.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

# ── Logical roles (the ONLY names the rest of the code should use) ───────────
ORCHESTRATOR = "orchestrator"  # default driver: input triage + output critique loop
THINKER = "thinker"            # deep reasoning, invoked only when triage decides
WRITER = "writer"              # grounded final-answer generation

ROLES = (ORCHESTRATOR, THINKER, WRITER)

# ── Dev stand-ins (gateway models) — swapped at deploy via env ───────────────
# These are the CURRENT local gateway models that best approximate each prod
# light model, so tuning here reflects what the light models will actually do.
_DEV_DEFAULTS = {
    # PROD wiring on VNG MaaS (contest models). Mapping by technical behaviour,
    # not raw size (2026-06-16, verified against the live MaaS endpoint):
    #  • orchestrator gọi NHIỀU lần ở token thấp (triage 120 / critique 200) →
    #    cần model nhanh, ra content ngay. qwen3 TẮT được thinking
    #    (enable_thinking=False) nên hợp ghế này.
    #  • thinker gọi ÍT, cần suy luận sâu ở token cao → minimax-m2.5 (mạnh nhất,
    #    #1 open-source agentic) nhưng LUÔN thinking & không tắt được → đặt đúng
    #    chỗ cần nghĩ + token cao để nó kịp ra content.
    #  • writer sinh câu trả lời B2B tiếng Việt → gemma-4-31b-it (non-reasoning,
    #    instruction-following + viết mượt).
    ORCHESTRATOR: "qwen/qwen3-5-27b",       # triage + critique (default driver, nhanh)
    THINKER:      "minimax/minimax-m2.5",   # deep reasoning / ReAct (mạnh nhất)
    WRITER:       "google/gemma-4-31b-it",  # ground the final answer (viết B2B)
}

# ── Prod targets (documentation only — set these as env on the VPS) ──────────
PROD_TARGETS = {
    ORCHESTRATOR: "qwen/qwen3-5-27b",
    THINKER:      "minimax/minimax-m2.5",
    WRITER:       "google/gemma-4-31b-it",
}

_ENV_KEY = {
    ORCHESTRATOR: "NODE_AGENT_MODEL_ORCHESTRATOR",
    THINKER:      "NODE_AGENT_MODEL_THINKER",
    WRITER:       "NODE_AGENT_MODEL_WRITER",
}


@dataclass(frozen=True)
class RoleModel:
    role: str
    model: str
    source: str  # "env" | "dev-default" | "single-fallback"


def resolve(role: str) -> RoleModel:
    """Map a logical role → concrete model id, honouring deploy-time env.

    Precedence:
      1. NODE_AGENT_MODEL_<ROLE>   (per-role override — the deploy knob)
      2. NODE_AGENT_MODEL          (single-model fallback: one model, all roles)
      3. dev stand-in default      (local gateway approximation)
    """
    if role not in ROLES:
        raise ValueError(f"unknown role {role!r}; valid: {ROLES}")
    env = os.environ.get(_ENV_KEY[role])
    if env:
        return RoleModel(role, env, "env")
    single = os.environ.get("NODE_AGENT_MODEL")
    if single:
        return RoleModel(role, single, "single-fallback")
    return RoleModel(role, _DEV_DEFAULTS[role], "dev-default")


def model_for(role: str) -> str:
    """Shorthand: just the model id for a role."""
    return resolve(role).model


def current_mapping() -> dict[str, RoleModel]:
    """Full role→model snapshot — surfaced to the dashboard so anh can SEE
    which model is driving which seat at runtime (baseline transparency)."""
    return {r: resolve(r) for r in ROLES}


if __name__ == "__main__":
    # Offline self-check: print the active harness wiring + source of each seat.
    print("=== Node Agent — role harness wiring ===")
    for r, rm in current_mapping().items():
        tag = {"env": "DEPLOY", "dev-default": "dev", "single-fallback": "SINGLE"}[rm.source]
        print(f"  {r:<13} → {rm.model:<32} [{tag}]")
    print("\nProd targets (set as env on VPS):")
    for r, m in PROD_TARGETS.items():
        print(f"  {_ENV_KEY[r]}={m}")
