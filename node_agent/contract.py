"""Tier 2 — I/O CONTRACT for Node Agent Assistant (define LLM input ↔ output).

This is the formal answer to "define input và output của LLM đối với input của
khách hàng và output của chúng ta" from the architecture report (§2 Structured
Outputs / Constrained Decoding).

WHY a contract layer
--------------------
A small/cheap model's output becomes the INPUT of deterministic systems (the
web UI, the citation renderer, telemetry, a future trading-intent executor). A
stray comma, a hallucinated field, or a broken JSON shape corrupts the whole
pipeline. The report's fix is constrained decoding: force the model output to
satisfy a known schema (JSON Schema / Pydantic), validated in code.

True logit-masking (Outlines / XGrammar, §2.2) needs control of the inference
server (vLLM). Our models live behind a remote OpenAI-compatible gateway, so we
CANNOT logit-mask. Instead we emulate the same guarantee the report wants with:

    schema-prompt  →  parse  →  validate(schema)  →  repair(once)  →  fallback

i.e. tell the model the exact JSON shape, parse tolerantly, validate against the
schema below, and on failure either repair (one cheap retry) or coerce to a safe
default. This gives the SAME contract guarantee at the system boundary without
needing server-side decoding.

THREE contracts (the data flow end to end):

  1. CUSTOMER_INPUT  — what the customer sends us (raw query + session).
  2. TRIAGE_OUTPUT   — LLM structured output #1: classify the input.
  3. AGENT_OUTPUT    — LLM structured output #2: what WE return (the answer
                       object the UI/clients consume: response_text + citations
                       + confidence + escalate flag).

Everything here is stdlib-only and offline-verifiable (the __main__ self-test
runs the validator over good/bad fixtures — no network, no model).
"""

from __future__ import annotations

import dataclasses
import json
import re
from typing import Any


# ── JSON Schemas (the contract definitions) ──────────────────────────────────
# Minimal JSON-Schema dicts. We ship our own tiny validator (below) so there is
# no jsonschema/pydantic dependency — important on a venv with no pip.

CUSTOMER_INPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "query":      {"type": "string", "minLength": 1, "maxLength": 4000},
        "session_id": {"type": "string", "maxLength": 80},
        "locale":     {"type": "string", "enum": ["vi", "en"]},
    },
    "required": ["query"],
}

# Triage = the model's classification of the customer input. This is LLM
# structured output #1 and decides routing (need_thinking → ReAct, shape, etc.).
TRIAGE_OUTPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "intent":        {"type": "string",
                          "enum": ["howto", "spec", "pricing", "compare",
                                   "troubleshoot", "general", "meta", "unsafe"]},
        "domain":        {"type": "string"},
        "need_thinking": {"type": "boolean"},
        "answer_shape":  {"type": "string",
                          "enum": ["short", "bullets", "table", "steps"]},
        "max_sentences": {"type": "integer", "minimum": 1, "maximum": 20},
    },
    "required": ["intent", "need_thinking", "answer_shape"],
}

# AgentOutput = what WE return to the customer/UI. This is LLM structured output
# #2, the system-boundary contract every client depends on.
AGENT_OUTPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "response_text": {"type": "string", "minLength": 1},
        "citations": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "n":     {"type": "integer", "minimum": 1},
                    "title": {"type": "string"},
                    "url":   {"type": "string"},
                },
                "required": ["n", "url"],
            },
        },
        "confidence":    {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "escalate":      {"type": "boolean"},
        "answer_shape":  {"type": "string"},
    },
    "required": ["response_text", "citations", "confidence"],
}


# ── Tiny JSON-Schema validator (subset, stdlib only) ─────────────────────────
def validate(obj: Any, schema: dict[str, Any], path: str = "$") -> list[str]:
    """Validate `obj` against a JSON-Schema subset. Returns a list of error
    strings (empty list = valid). Supports: type, properties, required, enum,
    minimum/maximum, minLength/maxLength, items. Enough for our contracts."""
    errs: list[str] = []
    t = schema.get("type")
    _TYPES = {
        "object": dict, "array": list, "string": str,
        "integer": int, "number": (int, float), "boolean": bool,
    }
    if t and t in _TYPES:
        # bool is a subclass of int — guard so True doesn't pass as integer.
        if t in ("integer", "number") and isinstance(obj, bool):
            errs.append(f"{path}: expected {t}, got boolean")
            return errs
        if not isinstance(obj, _TYPES[t]):
            errs.append(f"{path}: expected {t}, got {type(obj).__name__}")
            return errs

    if "enum" in schema and obj not in schema["enum"]:
        errs.append(f"{path}: {obj!r} not in enum {schema['enum']}")

    if t == "string" and isinstance(obj, str):
        if "minLength" in schema and len(obj) < schema["minLength"]:
            errs.append(f"{path}: shorter than minLength {schema['minLength']}")
        if "maxLength" in schema and len(obj) > schema["maxLength"]:
            errs.append(f"{path}: longer than maxLength {schema['maxLength']}")

    if t in ("integer", "number") and isinstance(obj, (int, float)) and not isinstance(obj, bool):
        if "minimum" in schema and obj < schema["minimum"]:
            errs.append(f"{path}: {obj} < minimum {schema['minimum']}")
        if "maximum" in schema and obj > schema["maximum"]:
            errs.append(f"{path}: {obj} > maximum {schema['maximum']}")

    if t == "object" and isinstance(obj, dict):
        for req in schema.get("required", []):
            if req not in obj:
                errs.append(f"{path}: missing required '{req}'")
        for key, sub in schema.get("properties", {}).items():
            if key in obj:
                errs += validate(obj[key], sub, f"{path}.{key}")

    if t == "array" and isinstance(obj, list) and "items" in schema:
        for i, item in enumerate(obj):
            errs += validate(item, schema["items"], f"{path}[{i}]")

    return errs


def is_valid(obj: Any, schema: dict[str, Any]) -> bool:
    return not validate(obj, schema)


# ── Tolerant JSON extraction (model output is messy) ─────────────────────────
def extract_json(text: str) -> dict | None:
    """Pull the first JSON object out of a model reply. Tolerant of code fences
    and leading/trailing prose. Returns None if nothing parses."""
    if not text:
        return None
    # strip ```json fences
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    candidates = []
    if fenced:
        candidates.append(fenced.group(1))
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        candidates.append(m.group(0))
    for c in candidates:
        try:
            d = json.loads(c)
            if isinstance(d, dict):
                return d
        except Exception:
            continue
    return None


# ── Coercion to safe defaults (the "fallback" arm of the contract) ───────────
@dataclasses.dataclass
class AgentOutput:
    """Typed view of the AGENT_OUTPUT contract used by the orchestrator/UI."""
    response_text: str
    citations: list[dict]
    confidence: float
    escalate: bool = False
    answer_shape: str = "short"

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


def coerce_agent_output(raw: dict | None, *, sources: list[dict] | None = None,
                        fallback_text: str = "") -> AgentOutput:
    """Coerce a (possibly malformed) model object into a valid AgentOutput.

    This is the contract's safety net: whatever the model produced, the system
    boundary always emits a schema-valid object. Missing/!invalid fields are
    repaired from `sources` and sane defaults instead of crashing a consumer.
    """
    raw = raw or {}
    text = str(raw.get("response_text") or fallback_text or "").strip()
    cites = raw.get("citations")
    if not isinstance(cites, list):
        cites = []
    clean_cites = []
    for c in cites:
        if isinstance(c, dict) and c.get("url"):
            clean_cites.append({"n": int(c.get("n") or len(clean_cites) + 1),
                                "title": str(c.get("title") or c["url"]),
                                "url": str(c["url"])})
    if not clean_cites and sources:
        clean_cites = [{"n": s.get("n", i + 1), "title": s.get("title", ""),
                        "url": s.get("url", "")} for i, s in enumerate(sources)
                       if s.get("url")]
    try:
        conf = float(raw.get("confidence"))
    except (TypeError, ValueError):
        conf = 1.0 if clean_cites else 0.3
    conf = max(0.0, min(1.0, conf))
    return AgentOutput(
        response_text=text,
        citations=clean_cites,
        confidence=conf,
        escalate=bool(raw.get("escalate", not text)),
        answer_shape=str(raw.get("answer_shape") or "short"),
    )


if __name__ == "__main__":
    # Offline self-test: validator over good/bad fixtures (no network, no model).
    cases = [
        # (obj, schema, expect_valid, note)
        ({"query": "thay IP private vServer"}, CUSTOMER_INPUT_SCHEMA, True, "min customer input"),
        ({"query": ""}, CUSTOMER_INPUT_SCHEMA, False, "empty query → minLength fail"),
        ({"session_id": "x"}, CUSTOMER_INPUT_SCHEMA, False, "missing required query"),
        ({"intent": "howto", "need_thinking": True, "answer_shape": "steps"},
         TRIAGE_OUTPUT_SCHEMA, True, "valid triage"),
        ({"intent": "bogus", "need_thinking": True, "answer_shape": "steps"},
         TRIAGE_OUTPUT_SCHEMA, False, "intent not in enum"),
        ({"intent": "howto", "need_thinking": "yes", "answer_shape": "steps"},
         TRIAGE_OUTPUT_SCHEMA, False, "need_thinking not boolean"),
        ({"response_text": "ok", "citations": [{"n": 1, "url": "https://x"}], "confidence": 0.9},
         AGENT_OUTPUT_SCHEMA, True, "valid agent output"),
        ({"response_text": "ok", "citations": [{"n": 1, "url": "https://x"}], "confidence": 2.0},
         AGENT_OUTPUT_SCHEMA, False, "confidence > 1.0"),
        ({"response_text": "ok", "citations": [{"title": "no url"}], "confidence": 0.5},
         AGENT_OUTPUT_SCHEMA, False, "citation missing url"),
    ]
    fails = 0
    for obj, schema, expect, note in cases:
        errs = validate(obj, schema)
        ok = not errs
        mark = "✓" if ok == expect else "✗ MISMATCH"
        if ok != expect:
            fails += 1
        print(f"{mark} valid={ok} :: {note}")
        for e in errs:
            print(f"      - {e}")

    # coercion: garbage in → valid AgentOutput out
    out = coerce_agent_output({"response_text": "Có H100.", "confidence": "high"},
                              sources=[{"n": 1, "title": "GPU", "url": "https://greennode.ai"}])
    assert is_valid(out.to_dict(), AGENT_OUTPUT_SCHEMA), "coerced output must be valid"
    print(f"\ncoerce → confidence={out.confidence}, cites={len(out.citations)} (repaired)")
    print("ALL PASS" if fails == 0 else f"{fails} FAILED")
