"""Tier 4 — Quality / Verify gate for Node Agent Assistant.

This is the "hardcore" tier: light open-weight models (gemma-4-31b-it,
qwen-3-27b) hallucinate and drop citations more than frontier models, so we
gate their output with deterministic, LLM-free checks before it reaches the
user. Everything here is verifiable with `python3 -m py_compile` + unit-style
asserts — no model call needed.

Checks performed on a drafted answer against the CONTEXT actually given:

  1. citation_format   — every [n] marker refers to a real source index.
  2. citation_present  — a fact-bearing answer must cite at least one source
                          (unless it is an explicit "not enough info" reply).
  3. no_invented_urls  — any http(s) URL in the answer must appear in sources.
  4. grounding         — sentences carrying product specifics (numbers, specs,
                          prices) should sit near a citation, else flagged.

The gate returns a verdict + reasons. The orchestrator decides what to do:
pass through, append a soft disclaimer, or fall back to the safe
"insufficient info" template. None of this needs the network.
"""

from __future__ import annotations

import dataclasses
import re

# A bare [3] style marker. We intentionally ignore [n] inside fenced code.
_CITE_RE = re.compile(r"\[(\d+)\]")
_URL_RE = re.compile(r"https?://[^\s)\]\"'>]+")
# Markers that the model is honestly declining — citations not required.
_REFUSAL_MARKERS = (
    "chưa có đủ thông tin",
    "không có đủ thông tin",
    "mình chưa có",
    "không tìm thấy",
    "liên hệ greennode",
    "info@greennode",
    "i don't have enough",
    "not enough information",
)
# Tokens that signal a concrete product claim worth grounding.
_SPECIFIC_RE = re.compile(
    r"(\d[\d.,]*\s*(gb|tb|tflops?|gbps|%|vnd|usd|\$|giờ|hour|core|vcpu|node))"
    r"|\b(h100|h200|hgx|nvlink|hbm3e?|sxm|pcie)\b",
    re.IGNORECASE,
)


@dataclasses.dataclass
class Verdict:
    ok: bool
    reasons: list[str]
    score: float  # 0..1, soft quality score for telemetry/tuning

    def __bool__(self) -> bool:
        return self.ok


def _strip_code_fences(text: str) -> str:
    """Remove fenced code blocks so [n]/URLs inside code aren't mis-scored."""
    return re.sub(r"```.*?```", " ", text, flags=re.DOTALL)


def _split_sentences(text: str) -> list[str]:
    # Lightweight VN/EN sentence split on terminal punctuation + newlines.
    parts = re.split(r"(?<=[.!?…])\s+|\n+", text)
    return [p.strip() for p in parts if p.strip()]


def is_refusal(answer: str) -> bool:
    low = answer.lower()
    return any(m in low for m in _REFUSAL_MARKERS)


def verify(answer: str, *, n_sources: int) -> Verdict:
    """Validate a drafted answer against the number of available sources.

    `n_sources` is len(sources) handed to the model (citations must be in
    1..n_sources). Returns a Verdict; callers gate on `.ok`.
    """
    reasons: list[str] = []
    body = _strip_code_fences(answer)
    cites = [int(m) for m in _CITE_RE.findall(body)]
    refusal = is_refusal(answer)

    # 1. citation format — no out-of-range / zero indices.
    bad = sorted({c for c in cites if c < 1 or c > n_sources})
    if bad:
        reasons.append(
            f"citation_format: marker(s) {bad} out of range 1..{n_sources}"
        )

    # 2. citation present — non-refusal answers must cite something.
    if not refusal and n_sources > 0 and not cites:
        reasons.append("citation_present: answer states facts but cites no source")

    # 3. no invented URLs — every URL must be a known source URL is enforced
    #    by the caller (it knows source URLs); here we just surface URLs so the
    #    orchestrator can cross-check. We flag obvious fabricationrisk: a URL
    #    that is not greennode/vngcloud domain.
    urls = _URL_RE.findall(body)
    foreign = [u for u in urls if not re.search(r"(greennode|vngcloud)\.", u)]
    if foreign:
        reasons.append(f"no_invented_urls: non-GreenNode URL(s) {foreign[:3]}")

    # 4. grounding — specific product claims should be near a citation.
    ungrounded = 0
    for sent in _split_sentences(body):
        if _SPECIFIC_RE.search(sent) and not _CITE_RE.search(sent):
            ungrounded += 1
    if ungrounded and not refusal:
        reasons.append(
            f"grounding: {ungrounded} specific claim(s) without a nearby [n]"
        )

    # Soft score: start at 1, subtract per reason class (floor 0).
    score = max(0.0, 1.0 - 0.25 * len(reasons))
    # Hard fail only on format errors or fabricated URLs (the dangerous ones).
    hard_fail = bool(bad) or bool(foreign)
    ok = not hard_fail
    return Verdict(ok=ok, reasons=reasons, score=score)


def verify_urls(answer: str, source_urls: set[str]) -> list[str]:
    """Return URLs in the answer that are NOT among the provided sources."""
    body = _strip_code_fences(answer)
    return [u.rstrip(".,)") for u in _URL_RE.findall(body) if u.rstrip(".,)") not in source_urls]


# Emoji + pictographic ranges — stripped for the enterprise B2B tone (no icons).
_EMOJI_RE = re.compile(
    "[\U0001F000-\U0001FAFF\U00002600-\U000027BF\U0001F1E6-\U0001F1FF"
    "\U00002190-\U000021FF\U00002B00-\U00002BFF\U0000FE00-\U0000FE0F\U00002700-\U000027BF]+",
    flags=re.UNICODE,
)


def sanitize_for_enterprise(answer: str, sources: list) -> str:
    """Post-process a draft into the enterprise house style (deterministic).

    Small models forget the prompt rules, so we ENFORCE them in code — the same
    'validate + repair' pattern as the contract layer:
      1. strip emoji / pictographs (B2B tone — no icons).
      2. replace any raw URL with its [n] citation marker when that URL matches
         a known source (so the answer carries [n], not a long link); unknown
         URLs are dropped to a bare '(nguồn)' rather than leaking a long link.
      3. collapse the leftover whitespace the removals create.

    `sources` is the list of source objects (each with .n and .url) so we can
    map url → [n]. Returns the cleaned answer; never raises.
    """
    if not answer:
        return answer
    text = _EMOJI_RE.sub("", answer)

    # Build url → [n] map (longest urls first so the most specific wins).
    url_to_n = {}
    for s in sources or []:
        u = getattr(s, "url", None) or (s.get("url") if isinstance(s, dict) else None)
        n = getattr(s, "n", None) or (s.get("n") if isinstance(s, dict) else None)
        if u and n:
            url_to_n[u.rstrip("/")] = n

    def _repl(m: re.Match) -> str:
        raw = m.group(0).rstrip(".,)]\"'")
        key = raw.rstrip("/")
        # exact or prefix match against a known source
        for su, n in url_to_n.items():
            if key == su or key.startswith(su) or su.startswith(key):
                return f"[{n}]"
        return "(nguồn)"  # unknown url → don't leak a long link

    # Protect markdown links [text](url): these are ALREADY the clean enterprise
    # form (show text, hide URL), so the frontend renders them as <a>. We must
    # NOT let the raw-URL stripper eat the url inside the parens. Swap each to a
    # placeholder, strip bare URLs, then restore.
    md_links: list[str] = []
    def _stash(m: re.Match) -> str:
        md_links.append(m.group(0))
        return f"\x00MDLINK{len(md_links) - 1}\x00"
    text = re.sub(r"\[[^\]]+\]\(https?://[^\s)]+\)", _stash, text)

    # Don't touch URLs inside code fences.
    parts = re.split(r"(```.*?```)", text, flags=re.DOTALL)
    for i, part in enumerate(parts):
        if not part.startswith("```"):
            parts[i] = _URL_RE.sub(_repl, part)
    text = "".join(parts)

    # restore protected markdown links
    for i, link in enumerate(md_links):
        text = text.replace(f"\x00MDLINK{i}\x00", link)

    # tidy: collapse the double spaces emoji removal leaves mid-sentence, plus
    # trailing spaces + 3+ newlines.
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"[ \t]+([.,;:!?])", r"\1", text)  # space before punctuation
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


if __name__ == "__main__":
    # Inline self-test (offline, deterministic) — doubles as the verify step.
    tests = [
        # (answer, n_sources, expect_ok, note)
        ("GreenNode cung cấp GPU H100 [1] với HBM3 80GB [2].", 2, True, "good cited"),
        ("GPU H100 có 80GB HBM3 và giá 9.99 USD/giờ.", 2, True, "specific, no cite → grounding flag (soft), score drops but not hard-fail"),
        ("Xem chi tiết tại [5].", 2, False, "out-of-range citation → hard fail"),
        ("Thông tin tại https://aws.amazon.com/ec2.", 1, False, "foreign URL → hard fail"),
        ("Mình chưa có đủ thông tin về phần này, vui lòng liên hệ info@greennode.vn.", 0, True, "refusal ok"),
        ("Chi tiết tại https://greennode.ai/pricing [1].", 1, True, "own URL ok"),
    ]
    fails = 0
    for ans, n, expect_ok, note in tests:
        v = verify(ans, n_sources=n)
        mark = "✓" if v.ok == expect_ok else "✗ MISMATCH"
        if v.ok != expect_ok:
            fails += 1
        print(f"{mark} ok={v.ok} score={v.score:.2f} :: {note}")
        for r in v.reasons:
            print(f"      - {r}")
    print(f"\n{'ALL PASS' if fails == 0 else f'{fails} FAILED'}")
