"""Tier 5 — MVP2 TRADING AGENT (Bifurcated Architecture skeleton).

The architecture report §6 is explicit: a trading bot is NOT a chatbot. A wrong
decision or a few hundred ms of lag is a financial loss. So the report mandates
a BIFURCATED (luồng kép) design — two loops that NEVER mix:

    ┌─ SLOW LOOP (Strategic, LLM)  ──────────────────────────────┐
    │  analyse market / news / chart → reason (ReAct, refine 3-5) │
    │  → emit a TRADING INTENT (structured JSON), seconds OK       │
    │  → RISK GATE (Critic) validates: capital, stop-loss, size    │
    └──────────────────────┬─────────────────────────────────────┘
                           │ approved intent (signed elsewhere)
                           ▼
    ┌─ FAST LOOP (Tactical, NO LLM)  ────────────────────────────┐
    │  Rust/Go engine holds gRPC/WS to SSI/TCBS/Binance           │
    │  → when live price matches the approved matrix → fire <100ms │
    └─────────────────────────────────────────────────────────────┘

WHY this file is a SKELETON (honest scope for MVP2):
  • The LLM analysis loop REUSES the MVP1 machinery (ReAct + LoopBudget + Judge)
    — same hardcore pipeline, different output schema. We don't rebuild it.
  • The TRADING INTENT contract + RISK GATE are the genuinely new, safety-
    critical pieces, so those are real and unit-tested here.
  • The TACTICAL ENGINE is a STUB: real execution needs Rust/gRPC + signed
    orders + a broker account (out of scope for a hackathon skeleton). We model
    its INTERFACE so the slow loop is testable end-to-end offline.

SAFETY INVARIANTS baked in (report §6.2 / §6.3):
  1. The LLM NEVER holds a private key and NEVER executes — it only PROPOSES an
     intent. Signing + sending lives in the (stubbed) execution engine.
  2. Every intent passes a deterministic RISK GATE before it can reach the
     engine. Fail = DROP the order (never rewrite it to sneak past — unlike the
     support agent which rewrites to improve, a risk failure is a hard stop).
  3. Orders above a size threshold require explicit human CONFIRMATION (halt).

Everything here is stdlib-only + offline-verifiable (the __main__ self-test runs
the risk gate over good/bad intents — no network, no broker, no model).
"""

from __future__ import annotations

import dataclasses
import json
import re
from typing import Any, Optional

from .contract import validate, extract_json


# ── TRADING INTENT contract (LLM structured output for MVP2) ─────────────────
# This is the slow loop's ONLY output: a proposal, not an execution. Mirrors the
# report's "Ý định Giao dịch" (§6.2). The execution engine consumes this AFTER
# the risk gate approves it.
TRADING_INTENT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "action":      {"type": "string", "enum": ["buy", "sell", "hold"]},
        "symbol":      {"type": "string", "minLength": 1, "maxLength": 20},
        "market":      {"type": "string", "enum": ["HOSE", "HNX", "UPCOM", "BINANCE"]},
        "order_type":  {"type": "string", "enum": ["limit", "market"]},
        "entry_low":   {"type": "number", "minimum": 0},
        "entry_high":  {"type": "number", "minimum": 0},
        "quantity":    {"type": "integer", "minimum": 0},
        "stop_loss":   {"type": "number", "minimum": 0},
        "take_profit": {"type": "number", "minimum": 0},
        "confidence":  {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "reasoning":   {"type": "string"},
    },
    # An actionable intent must name what/where/how + carry a stop-loss (no
    # naked positions) + a confidence. "hold" intents skip the price fields.
    "required": ["action", "symbol", "market", "confidence"],
}


# ── RISK LIMITS (the deterministic gate's policy) ────────────────────────────
@dataclasses.dataclass
class RiskLimits:
    """Per-account risk policy. The gate is LLM-free + deterministic so it can
    never be 'talked into' approving a bad order (report §6.3 least-privilege)."""
    max_order_value: float = 50_000_000      # VND (or quote ccy) per single order
    min_confidence: float = 0.6              # below this → never trade
    require_stop_loss: bool = True           # no naked positions
    max_stop_loss_pct: float = 0.10          # stop-loss must be within 10% of entry
    confirm_above_value: float = 20_000_000  # orders above this need human confirm


@dataclasses.dataclass
class RiskVerdict:
    approved: bool
    needs_confirmation: bool
    reasons: list[str]

    def __bool__(self) -> bool:
        return self.approved


def risk_gate(intent: dict, limits: RiskLimits) -> RiskVerdict:
    """Deterministically validate a TRADING INTENT against the risk policy.

    Returns RiskVerdict(approved, needs_confirmation, reasons). A failed gate
    DROPS the order — the slow loop must NOT rewrite-to-pass (that's the support
    agent's behaviour; here a risk failure is a hard stop, §6.2).
    """
    reasons: list[str] = []

    # 0. schema first — a malformed intent is an automatic reject.
    errs = validate(intent, TRADING_INTENT_SCHEMA)
    if errs:
        return RiskVerdict(False, False, [f"schema: {e}" for e in errs[:3]])

    action = intent.get("action")
    if action == "hold":
        # A 'hold' proposes no trade — always safe, nothing to execute.
        return RiskVerdict(True, False, ["hold: no order to place"])

    conf = float(intent.get("confidence") or 0.0)
    if conf < limits.min_confidence:
        reasons.append(f"confidence {conf:.2f} < min {limits.min_confidence}")

    # stop-loss presence + sanity
    sl = intent.get("stop_loss")
    entry = intent.get("entry_high") or intent.get("entry_low")
    if limits.require_stop_loss and not sl:
        reasons.append("missing stop_loss (no naked positions)")
    if sl and entry:
        dist = abs(entry - sl) / entry if entry else 1.0
        if dist > limits.max_stop_loss_pct:
            reasons.append(f"stop_loss {dist:.0%} away > max {limits.max_stop_loss_pct:.0%}")

    # order value ceiling
    qty = intent.get("quantity") or 0
    price = entry or 0
    order_value = qty * price
    if order_value > limits.max_order_value:
        reasons.append(f"order value {order_value:,.0f} > max {limits.max_order_value:,.0f}")

    # buy/sell must carry a tradeable quantity + price
    if action in ("buy", "sell"):
        if qty <= 0:
            reasons.append("quantity must be > 0 for a buy/sell")
        if price <= 0:
            reasons.append("entry price required for a buy/sell")

    approved = not reasons
    # large (but valid) orders still require a human in the loop (§6.3).
    needs_confirm = approved and order_value >= limits.confirm_above_value
    if needs_confirm:
        reasons.append(f"order value {order_value:,.0f} ≥ confirm threshold "
                       f"{limits.confirm_above_value:,.0f} → HUMAN CONFIRM required")
    return RiskVerdict(approved, needs_confirm, reasons or ["ok"])


# ── TACTICAL ENGINE (STUB — interface only, no real execution) ───────────────
@dataclasses.dataclass
class ExecutionResult:
    placed: bool
    detail: str
    intent: dict


class TacticalEngineStub:
    """Models the fast-loop execution engine's INTERFACE without real I/O.

    A real implementation is a separate Rust/Go process holding gRPC/WS to the
    broker, signing each order with the user's private key. Here we only record
    that an APPROVED + CONFIRMED intent WOULD be sent — so the slow loop is
    end-to-end testable offline. The LLM never reaches this class directly; the
    orchestrator calls it only after risk_gate approves.
    """
    def __init__(self):
        self.sent: list[dict] = []

    def execute(self, intent: dict, verdict: RiskVerdict,
                human_confirmed: bool = False) -> ExecutionResult:
        if not verdict.approved:
            return ExecutionResult(False, "risk gate rejected → order DROPPED", intent)
        if verdict.needs_confirmation and not human_confirmed:
            return ExecutionResult(False, "awaiting HUMAN CONFIRMATION → held", intent)
        if intent.get("action") == "hold":
            return ExecutionResult(False, "hold → nothing to place", intent)
        # STUB: a real engine would sign + send via gRPC here.
        self.sent.append(intent)
        return ExecutionResult(True, f"[STUB] would place {intent['action']} "
                               f"{intent.get('quantity')} {intent['symbol']} "
                               f"@ {intent.get('entry_low')}-{intent.get('entry_high')} "
                               f"on {intent['market']}", intent)


# ── Coercion: model output → safe TradingIntent (the contract's fallback) ────
def coerce_trading_intent(raw: dict | None) -> dict:
    """Coerce a (possibly malformed) model reply into a schema-shaped intent.

    Defaults to a safe 'hold' when the model output is unusable — the system
    boundary never emits a half-parsed trade.
    """
    raw = raw or {}
    action = str(raw.get("action") or "hold").lower()
    if action not in ("buy", "sell", "hold"):
        action = "hold"
    out = {
        "action": action,
        "symbol": str(raw.get("symbol") or "").upper()[:20],
        "market": str(raw.get("market") or "HOSE").upper(),
        "confidence": _safe_float(raw.get("confidence"), 0.0),
        "reasoning": str(raw.get("reasoning") or "")[:500],
    }
    if action in ("buy", "sell"):
        out["order_type"] = str(raw.get("order_type") or "limit")
        out["entry_low"] = _safe_float(raw.get("entry_low"), 0.0)
        out["entry_high"] = _safe_float(raw.get("entry_high") or raw.get("entry_low"), 0.0)
        out["quantity"] = _safe_int(raw.get("quantity"), 0)
        out["stop_loss"] = _safe_float(raw.get("stop_loss"), 0.0)
        out["take_profit"] = _safe_float(raw.get("take_profit"), 0.0)
    if out["market"] not in ("HOSE", "HNX", "UPCOM", "BINANCE"):
        out["market"] = "HOSE"
    return out


def _safe_float(v, default: float) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _safe_int(v, default: int) -> int:
    try:
        return int(v)
    except (TypeError, ValueError):
        return default


if __name__ == "__main__":
    # Offline self-test: risk gate over good/bad intents (no network/broker/model).
    limits = RiskLimits()
    engine = TacticalEngineStub()

    cases = [
        # (intent, expect_approved, note)
        ({"action": "hold", "symbol": "FPT", "market": "HOSE", "confidence": 0.8},
         True, "hold → always safe"),
        ({"action": "buy", "symbol": "FPT", "market": "HOSE", "order_type": "limit",
          "entry_low": 100000, "entry_high": 102000, "quantity": 100,
          "stop_loss": 96000, "confidence": 0.75},
         True, "valid small buy with stop-loss (102000*100=10.2M < 20M confirm)"),
        ({"action": "buy", "symbol": "VCB", "market": "HOSE", "order_type": "limit",
          "entry_low": 90000, "entry_high": 92000, "quantity": 100, "confidence": 0.8},
         False, "missing stop-loss → reject"),
        ({"action": "buy", "symbol": "SSI", "market": "HOSE", "order_type": "limit",
          "entry_low": 30000, "entry_high": 31000, "quantity": 100,
          "stop_loss": 20000, "confidence": 0.8},
         False, "stop-loss 35% away > 10% → reject"),
        ({"action": "buy", "symbol": "HPG", "market": "HOSE", "order_type": "limit",
          "entry_low": 28000, "entry_high": 28000, "quantity": 100, "stop_loss": 26000,
          "confidence": 0.4},
         False, "confidence 0.4 < 0.6 → reject"),
        ({"action": "sell", "symbol": "BTCUSDT", "market": "BINANCE", "order_type": "market",
          "entry_low": 95000, "entry_high": 95000, "quantity": 400, "stop_loss": 99000,
          "confidence": 0.9},
         True, "valid order 38M (≥20M confirm, <50M max → approved + needs confirmation)"),
    ]

    fails = 0
    for intent, expect, note in cases:
        v = risk_gate(intent, limits)
        mark = "✓" if v.approved == expect else "✗ MISMATCH"
        if v.approved != expect:
            fails += 1
        print(f"{mark} approved={v.approved} confirm={v.needs_confirmation} :: {note}")
        for r in v.reasons[:2]:
            print(f"      - {r}")

    # end-to-end: a large approved order is HELD until human confirms.
    big = cases[-1][0]
    v = risk_gate(big, limits)
    r1 = engine.execute(big, v, human_confirmed=False)
    assert not r1.placed, "large order must be held without confirmation"
    r2 = engine.execute(big, v, human_confirmed=True)
    assert r2.placed, "large order must place after confirmation"
    print(f"\nexec without confirm: {r1.detail}")
    print(f"exec with confirm:    {r2.detail}")

    # coercion: garbage → safe hold
    safe = coerce_trading_intent({"action": "YOLO ALL IN", "confidence": "high"})
    assert safe["action"] == "hold", "unparseable action must coerce to hold"
    print(f"\ncoerce garbage → action={safe['action']} (safe default)")

    print("\nALL PASS" if fails == 0 else f"\n{fails} FAILED")
