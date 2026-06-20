"""Tier 0 — LLM Provider Layer for Node Agent Assistant.

Config-driven OpenAI-compatible chat client. This is the HARDCODE SLOT:
the contest spirit is light open-weight models (VNG MaaS gemma-4-31b-it /
qwen-3-27b), but the provider is swappable via env so we can develop against
any OpenAI-compatible endpoint (currently the in-house gateway stand-in)
while the MaaS key is pending.

Env (read at construction, never hardcoded):
  NODE_AGENT_BASE_URL   e.g. http://192.168.100.103:20128/v1
  NODE_AGENT_API_KEY    bearer key
  NODE_AGENT_MODEL      e.g. qwen/qwen3-5-27b  (stand-in)
                        → swap to google/gemma-4-31b-it once MaaS key lands

Quirk handled (verified 2026-06-14): the in-house gateway appends
`data: [DONE]\\n\\n` AFTER a valid JSON chat.completion body (non-spec SSE
tail on a non-streaming response). Plain json.loads() then throws
"Extra data". We cut at the first balanced top-level object before decoding.

stdlib-only (urllib + json) so it runs on a bare interpreter and is
verifiable without installing an SDK.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass


@dataclass
class ChatResult:
    text: str
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0


def _first_json_object(raw: str) -> str:
    """Return the first balanced {...} object from a string.

    Handles the gateway's `{...valid json...}data: [DONE]` tail by scanning
    brace depth while respecting strings/escapes, and stopping at the close
    of the first top-level object.
    """
    start = raw.find("{")
    if start == -1:
        raise ValueError(f"no JSON object in response: {raw[:200]!r}")
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(raw)):
        c = raw[i]
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
            continue
        if c == '"':
            in_str = True
        elif c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return raw[start : i + 1]
    raise ValueError("unbalanced JSON object in response")


# VNG MaaS serves reasoning models (qwen3, minimax) that, by default, spend the
# token budget on a hidden chain-of-thought and return content=None when the
# budget runs out before the visible answer. For a customer-support bot we want
# the ANSWER, fast — so we ask the server to disable thinking. qwen3/vLLM honour
# `chat_template_kwargs.enable_thinking=False`; gemma (non-reasoning) and minimax
# (ignores the flag) are unaffected, so sending it unconditionally is safe.
_THINK_OFF = {"chat_template_kwargs": {"enable_thinking": False}}


def _with_thinking_off(body: dict) -> dict:
    body.update(_THINK_OFF)
    return body


def _content_of(message: dict) -> str:
    """Pull the visible answer from a chat message, tolerating reasoning models.

    Some models (minimax) split a turn into `reasoning_content` (the thinking)
    and `content` (the final answer). When the model exhausts max_tokens mid-
    thought, `content` is None/empty — we fall back to reasoning_content so the
    pipeline gets SOMETHING rather than crashing on None, then strip.
    """
    txt = message.get("content")
    if txt:
        return txt
    # last resort: a reasoning field, so the caller isn't handed None
    for k in ("reasoning_content", "reasoning"):
        v = message.get(k)
        if v:
            return v
    return ""


class Provider:
    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
        timeout: int | None = None,
    ) -> None:
        self.base_url = (base_url or os.environ.get("NODE_AGENT_BASE_URL", "")).rstrip("/")
        self.api_key = api_key or os.environ.get("NODE_AGENT_API_KEY", "")
        self.model = model or os.environ.get("NODE_AGENT_MODEL", "")
        # Gateway can be slow (multi-chart / long answers run 120-180s+). Default
        # 300s, overridable via NODE_AGENT_TIMEOUT so it's tunable without a code edit.
        if timeout is None:
            try:
                timeout = int(os.environ.get("NODE_AGENT_TIMEOUT", "300"))
            except ValueError:
                timeout = 300
        self.timeout = timeout
        if not self.base_url or not self.model:
            raise ValueError(
                "Provider needs NODE_AGENT_BASE_URL + NODE_AGENT_MODEL "
                "(set env or pass args)."
            )

    def chat(
        self,
        messages: list[dict],
        *,
        temperature: float = 0.2,
        max_tokens: int = 1024,
        model: str | None = None,
    ) -> ChatResult:
        # Per-request model override lets one Provider serve every model the
        # dashboard exposes without re-instantiating (same base_url/key).
        use_model = model or self.model
        body = json.dumps(
            _with_thinking_off(
                {
                    "model": use_model,
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "stream": False,
                }
            )
        ).encode("utf-8")
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=body,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )
        # Retry with exponential backoff on transient gateway errors. The shared
        # router rate-limits (429) and sheds load (503) under contention; a bare
        # single attempt made every seat fail the whole turn. We retry a few
        # times with growing waits so a brief spike doesn't kill the request.
        import time as _t
        last_err = None
        for attempt in range(4):
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    raw = resp.read().decode("utf-8", errors="replace")
                break
            except urllib.error.HTTPError as e:
                last_err = e
                if e.code in (429, 503, 502, 504) and attempt < 3:
                    _t.sleep(2 * (attempt + 1))  # 2s, 4s, 6s
                    continue
                raise
            except Exception as e:
                last_err = e
                if attempt < 3:
                    _t.sleep(2 * (attempt + 1))
                    continue
                raise
        else:
            raise last_err
        data = json.loads(_first_json_object(raw))
        choice = _content_of(data["choices"][0]["message"])
        usage = data.get("usage") or {}
        return ChatResult(
            text=choice,
            model=data.get("model", self.model),
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
        )


    def chat_stream(
        self,
        messages: list[dict],
        *,
        temperature: float = 0.2,
        max_tokens: int = 1024,
        model: str | None = None,
        on_delta=None,
    ) -> ChatResult:
        """Streaming chat: sends stream:True and feeds each token delta to
        `on_delta(text)` as it arrives, then returns the FULL ChatResult once
        the stream closes (so CRITIQUE / judge still get the whole answer).

        Streaming does NOT make the gateway faster — the model emits the same
        tokens either way. It makes the WAIT feel live: the customer watches the
        answer build instead of staring at a 2-3 minute spinner. We fall back to
        the non-stream `chat()` if the gateway rejects streaming or sends nothing.
        """
        use_model = model or self.model
        body = json.dumps(
            _with_thinking_off(
                {
                    "model": use_model,
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "stream": True,
                }
            )
        ).encode("utf-8")
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=body,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
                "Accept": "text/event-stream",
            },
            method="POST",
        )
        import time as _t
        last_err = None
        for attempt in range(4):
            parts: list[str] = []
            model_name = use_model
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    for raw_line in resp:
                        line = raw_line.decode("utf-8", errors="replace").strip()
                        if not line or not line.startswith("data:"):
                            continue
                        payload = line[5:].strip()
                        if payload == "[DONE]":
                            break
                        try:
                            obj = json.loads(payload)
                        except Exception:
                            continue
                        model_name = obj.get("model", model_name)
                        for ch in obj.get("choices", []):
                            delta = (ch.get("delta") or {}).get("content")
                            if delta:
                                parts.append(delta)
                                if on_delta:
                                    on_delta(delta)
                text = "".join(parts)
                if not text:
                    # Gateway accepted the request but streamed nothing usable —
                    # fall back to a normal blocking call so we still answer.
                    return self.chat(messages, temperature=temperature,
                                     max_tokens=max_tokens, model=model)
                # Usage isn't reliably sent on streamed responses; estimate cheaply.
                return ChatResult(text=text, model=model_name,
                                  prompt_tokens=0,
                                  completion_tokens=len(text) // 4)
            except urllib.error.HTTPError as e:
                last_err = e
                if e.code in (429, 503, 502, 504) and attempt < 3:
                    _t.sleep(2 * (attempt + 1))
                    continue
                # Streaming not supported / other HTTP error → blocking fallback.
                return self.chat(messages, temperature=temperature,
                                 max_tokens=max_tokens, model=model)
            except Exception as e:
                last_err = e
                if attempt < 3:
                    _t.sleep(2 * (attempt + 1))
                    continue
                # Last resort: one blocking attempt before giving up.
                return self.chat(messages, temperature=temperature,
                                 max_tokens=max_tokens, model=model)
        raise last_err


if __name__ == "__main__":
    import sys

    p = Provider()
    q = " ".join(sys.argv[1:]) or "Trả lời 1 từ: thủ đô Việt Nam?"
    r = p.chat([{"role": "user", "content": q}], max_tokens=64)
    print(f"model={r.model} tok={r.prompt_tokens}+{r.completion_tokens}")
    print(r.text)
