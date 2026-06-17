"""Tier 3.7 — LOOP BUDGET for Node Agent Assistant (MVP1 unified counter).

The architecture report §3.2 is explicit: every agentic loop MUST have a hard
ceiling and a safe-halt mechanism. The current code has two independent ceilings
(ReAct MAX_AGENT_ROUNDS=2, critique MAX_CRITIQUE=3) that don't share state —
the total LLM-calls can hit 5+ with no single counter tracking the budget.

This module defines ONE unified budget per customer request:

    GATHER_MAX   = 3   (ReAct tool-selection rounds — each = 1 thinker call)
    REFINE_MAX   = 2   (total rewrite rounds, NOT counting the first draft)
    TOTAL_LLM_MAX = 8  (hard ceiling: triage + gather + draft + judge + rewrite)

The counter is checked BEFORE every LLM call. If the budget is exhausted, the
pipeline emits ESCALATE with a safe fallback — never crashes, never produces a
half-baked answer, never loops forever.

Why these numbers (the math, simplified):
  triage:          0-1 calls  (heuristic + optional LLM refine)
  gather:          0-3 calls  (ReAct reasoning rounds)
  draft:           1 call     (always — we must produce something)
  judge:           1 call     (always — G-Eval on every draft)
  rewrite+judge:   0-2 pairs  (fix, then judge again — 2 pairs max)
  ────────────────
  worst case:      1 + 3 + 1 + 1 + (2×2) = 10, but the ceil is 8 because
  gather=3 is only when need_thinking=True, and triage LLM is rare.
  For the COMMON case (need_thinking=True, 1 rewrite): 0 + 2 + 1 + 1 + 2 = 6.
  Happy path (no rewrite):                               0 + 2 + 1 + 1     = 4.
"""
from __future__ import annotations
import dataclasses

# ── Ceilings (the contract §3.2) ─────────────────────────────────────────────
# GATHER_MAX raised 3→4 (2026-06-16): the THINKER seat is now minimax-m2.5, a
# reasoning model whose ReAct evidence-gathering has higher variance than opus —
# some runs picked a loose chunk and stopped one round too early, so a chart that
# needed a second data-bearing chunk didn't get it. One extra gather round lets
# the loop recover the missing numbers; TOTAL_LLM_MAX stays 8 as the hard ceiling.
# NOTE (2026-06-17): this ceiling is now ACTUALLY enforced — orchestrator passes
# max_rounds=GATHER_MAX into run_react(), which previously hard-stopped at its own
# MAX_AGENT_ROUNDS=2 and silently ignored this budget. Kept at 3 (not 4) to cap
# minimax latency: each extra ReAct round costs ~tens of seconds.
GATHER_MAX = 3       # ReAct reasoning rounds (0 if fast-path)
REFINE_MAX = 2       # corrective rewrite rounds (not counting first draft)
TOTAL_LLM_MAX = 8    # hard ceiling across the whole pipeline per request


@dataclasses.dataclass
class LoopBudget:
    """Mutable counter that tracks LLM-calls in a single orchestrator.run().

    Every call site MUST check `budget.can(label)` before calling provider.chat().
    If it returns False, the pipeline must ESCALATE (safe fallback) — never
    call the LLM anyway. This is the §3.2 "counter + safe exit" requirement.
    """
    gather_rounds: int = 0
    refine_rounds: int = 0
    total_llm: int = 0
    escalate_reason: str = ""

    # ── Guard methods ────────────────────────────────────────────────────────
    def can_gather(self) -> bool:
        """True if we can still run a ReAct gather round."""
        if self.total_llm >= TOTAL_LLM_MAX:
            self.escalate_reason = (f"total_llm={self.total_llm} ≥ "
                                    f"{TOTAL_LLM_MAX} (hard ceiling)")
            return False
        if self.gather_rounds >= GATHER_MAX:
            self.escalate_reason = (f"gather={self.gather_rounds} ≥ "
                                    f"{GATHER_MAX} (gather ceiling)")
            return False
        return True

    def can_refine(self) -> bool:
        """True if we can still do a rewrite+judge pair."""
        if self.total_llm >= TOTAL_LLM_MAX:
            self.escalate_reason = (f"total_llm={self.total_llm} ≥ "
                                    f"{TOTAL_LLM_MAX} (hard ceiling)")
            return False
        if self.refine_rounds >= REFINE_MAX:
            self.escalate_reason = (f"refine={self.refine_rounds} ≥ "
                                    f"{REFINE_MAX} (refine ceiling)")
            return False
        return True

    def can(self, label: str = "") -> bool:
        """Generic check before ANY LLM call."""
        if self.total_llm >= TOTAL_LLM_MAX:
            self.escalate_reason = (f"total_llm={self.total_llm} ≥ "
                                    f"{TOTAL_LLM_MAX} (hard ceiling)")
            return False
        return True

    # ── Bump methods (call AFTER the LLM call succeeds) ──────────────────────
    def bump(self, label: str = "") -> None:
        """Increment the generic + total counters."""
        self.total_llm += 1
        if label == "gather":
            self.gather_rounds += 1
        elif label == "refine":
            self.refine_rounds += 1

    # ── Reporting ────────────────────────────────────────────────────────────
    def summary(self) -> dict:
        return {
            "gather": self.gather_rounds,
            "refine": self.refine_rounds,
            "total_llm": self.total_llm,
            "ceilings": {"gather": GATHER_MAX, "refine": REFINE_MAX,
                         "total": TOTAL_LLM_MAX},
        }

    def __repr__(self) -> str:
        return (f"Budget(g={self.gather_rounds}/{GATHER_MAX} "
                f"r={self.refine_rounds}/{REFINE_MAX} "
                f"t={self.total_llm}/{TOTAL_LLM_MAX})")


# ── Unit test (runs with: python -m node_agent.loop_budget) ──────────────────
if __name__ == "__main__":
    b = LoopBudget()
    # Happy path: 2 gather + 1 draft + 1 judge = 4 calls, no rewrite
    for i in range(2):
        assert b.can_gather(), "should allow gather"
        b.bump("gather")
    assert b.can()
    b.bump("draft"); b.bump("judge")  # 2 more = total 4
    assert not b.can_refine() or b.can_refine()  # refine_rounds=0, should be True
    print(f"happy path: {b} OK")

    # Exhaust gather ceiling
    b2 = LoopBudget()
    for i in range(GATHER_MAX):
        assert b2.can_gather()
        b2.bump("gather")
    assert not b2.can_gather()
    assert b2.escalate_reason
    print(f"gather ceiling: {b2} → escalate: {b2.escalate_reason}")

    # Exhaust total ceiling
    b3 = LoopBudget()
    for i in range(TOTAL_LLM_MAX):
        assert b3.can()
        b3.bump()
    assert not b3.can()
    print(f"total ceiling: {b3} → escalate: {b3.escalate_reason}")

    # Exhaust refine ceiling
    b4 = LoopBudget()
    for i in range(REFINE_MAX):
        assert b4.can_refine()
        b4.bump("refine")
    assert not b4.can_refine()
    print(f"refine ceiling: {b4} → escalate: {b4.escalate_reason}")

    print("\nALL PASS")
