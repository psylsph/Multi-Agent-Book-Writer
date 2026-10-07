"""Shared configuration and LLM client.

Single entry point for all LLM calls. Works with any OpenAI-compatible
endpoint (LM Studio, llama.cpp server, vLLM, Ollama's /v1 API, OpenAI,
OpenRouter, ...). Reads config.yaml once, applies per-agent temperature,
and adds timeouts + retries with backoff.
"""

import json
import os
import re
import time
from pathlib import Path

import requests
import yaml

from shared import config_schema

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.yaml"

_config = None

# Web search for the researcher (off unless enabled in config.yaml).
WEB_SEARCH_DEFAULTS = {
    "enabled": False,
    "searxng_url": "http://localhost:8888",
    "auto_start": "ask",            # ask | yes | no: offer a Docker container
    "docker_image": "searxng/searxng",
    "queries_per_chapter": 3,
    "results_per_query": 4,
    "snippet_chars": 300,
    "categories": "general",
    "timeout": 15,
    "double_check": True,           # second model pass over each verdict
    "banned_terms": [],            # never allowed in a search query
}


class EndpointUnavailable(RuntimeError):
    """The LLM endpoint is down, unreachable, or not ready (connection
    refused, timeout, or an HTTP 5xx such as 'Loading model'). Retrying
    once the server is back should succeed."""


# Strip leaked reasoning blocks. Some servers put the model's internal
# thinking into 'content' instead of (or in addition to) a 'reasoning'
# field; never let it reach downstream text.
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
_THINK_OPEN_RE = re.compile(r"<think>.*\Z", re.DOTALL)  # cut off mid-thought


def load_config(path=None):
    """Load (or reload) the configuration. Called once by main.py at startup."""
    global _config
    config_path = Path(path) if path else DEFAULT_CONFIG_PATH
    if not config_path.exists():
        raise FileNotFoundError(
            f"Config file not found: {config_path}. Copy config.example.yaml "
            "to config.yaml (or pass --config FILE) and edit it.")
    with open(config_path, "r", encoding="utf-8") as f:
        _config = yaml.safe_load(f) or {}

    for message in config_schema.problems(_config):
        print(f"[CONFIG] {message}")

    # A section whose keys are all commented out parses as None; normalise.
    for section in ("book", "llm", "agents", "output", "web_search"):
        if not isinstance(_config.get(section), dict):
            _config[section] = {}
    for name, agent_cfg in list(_config["agents"].items()):
        if not isinstance(agent_cfg, dict):
            _config["agents"][name] = {}

    # sane defaults so a partial config file doesn't crash the pipeline
    _config["book"].setdefault("num_chapters", 5)
    _config["book"].setdefault("words_per_chapter", 800)
    _config.setdefault("llm", {})
    llm_cfg = _config["llm"]

    # migrate legacy 'ollama:' section if present and 'llm:' is unset
    legacy = _config.get("ollama") or {}
    if "base_url" not in llm_cfg and legacy.get("api_url"):
        llm_cfg["base_url"] = legacy["api_url"]
    if "model" not in llm_cfg and legacy.get("model"):
        llm_cfg["model"] = legacy["model"]

    llm_cfg.setdefault("base_url", "http://localhost:11434")
    llm_cfg.setdefault("api_key", "")
    llm_cfg.setdefault("model", "mistral")
    llm_cfg.setdefault("timeout", 300)
    llm_cfg.setdefault("retries", 2)
    llm_cfg.setdefault("endpoint_wait", 300)  # s to wait for a downed server
    llm_cfg.setdefault("reasoning_effort", "")  # "", low, medium, xhigh
    llm_cfg.setdefault("enable_thinking", None)  # None = don't send it
    llm_cfg.setdefault("stream", False)          # stream responses (progress)
    llm_cfg.setdefault("context_window", None)   # tokens; None = no guard
    llm_cfg.setdefault("json_mode", False)       # ask for JSON-object output
    for key, value in WEB_SEARCH_DEFAULTS.items():
        _config["web_search"].setdefault(key, value)
    _config["output"].setdefault("directory", "output")
    _config["output"].setdefault("filename", "draft.md")
    _config["output"].setdefault("overwrite", True)
    _config["output"].setdefault("log", True)
    return _config


def get_config():
    """Return the loaded config, lazily loading the default one if needed."""
    if _config is None:
        load_config()
    return _config


def _resolve_api_key(cfg):
    """Resolve the API key: 'env:VAR' syntax, plain value, or LLM_API_KEY."""
    raw = str(cfg.get("api_key") or "").strip()
    if raw.startswith("env:"):
        raw = os.environ.get(raw[4:].strip(), "")
    if not raw:
        raw = os.environ.get("LLM_API_KEY", "")
    return raw


def _base_url(llm_cfg):
    """Base URL without a trailing slash or '/v1' (the client appends
    /v1/...), so both 'http://host:8080' and 'https://api.openai.com/v1'
    work."""
    base = str(llm_cfg["base_url"]).rstrip("/")
    if base.endswith("/v1"):
        base = base[:-3].rstrip("/")
    return base


def _headers(api_key):
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


# ------------------------------------------------- per-agent thinking effort

# Reasoning models think for a very long time at their highest effort. These
# stages need little of it (notes, JSON extraction) or some (continuity
# judgement), so they default to less. The other stages (writer, editor,
# architect, planner) use llm.reasoning_effort as set. Applied ONLY when
# llm.reasoning_effort is set: an empty value means "send no thinking
# controls", and the defaults must not start sending them. Override per agent
# with agents.<name>.reasoning_effort.
AGENT_REASONING_DEFAULTS = {
    "researcher": "low",
    "verifier": "low",
    "extractor": "low",
    "reviewer": "medium",
}


def _template_kwargs(llm_cfg, agent_cfg, agent):
    """chat_template_kwargs for one agent's request ({} when none apply).

    reasoning_effort: agents.<name>.reasoning_effort if the key is present
    (an empty value sends nothing); else, when llm.reasoning_effort is set,
    the built-in default for the agent or the global value.
    enable_thinking: agents.<name>.enable_thinking, else llm.enable_thinking.
    """
    kwargs = {}
    if "reasoning_effort" in agent_cfg:
        effort = str(agent_cfg["reasoning_effort"] or "").strip().lower()
    else:
        effort = str(llm_cfg.get("reasoning_effort") or "").strip().lower()
        if effort:
            effort = AGENT_REASONING_DEFAULTS.get(agent, effort)
    if effort:
        kwargs["reasoning_effort"] = effort
    thinking = (agent_cfg["enable_thinking"] if "enable_thinking" in agent_cfg
                else llm_cfg.get("enable_thinking"))
    if thinking is not None:
        kwargs["enable_thinking"] = bool(thinking)
    return kwargs


# A model that loops while writing JSON never stops: measured on a thinking
# model, a single review ran past 9,800 tokens (about 10 minutes) before it was
# cancelled, and an unparseable reply is then silently treated as a pass. These
# agents only ever produce short structured replies (plus, at most, modest
# thinking), so they get a generous ceiling by default. The writer, editor,
# architect and planner are NOT capped: their thinking at xhigh can be long.
# Used only when neither agents.<name>.max_tokens nor llm.max_tokens is set.
AGENT_MAX_TOKENS_DEFAULTS = {
    "researcher": 6000,
    "extractor": 6000,
    "verifier": 2000,
    "reviewer": 8000,
}


def _max_tokens(llm_cfg, agent_cfg, agent=None):
    """agents.<name>.max_tokens, else llm.max_tokens, else the built-in
    ceiling for short-reply agents (AGENT_MAX_TOKENS_DEFAULTS), else None."""
    return (agent_cfg.get("max_tokens") or llm_cfg.get("max_tokens")
            or AGENT_MAX_TOKENS_DEFAULTS.get(agent) or None)


# ---------------------------------------------------------------- run stats

PROGRESS_SECONDS = 20          # streaming progress line interval
STATS = {}                     # agent -> counters, since reset_stats()
_CHARS = {"chars": 0, "tokens": 0}   # calibrates characters per token
_WARNED = set()                # agents already warned about context size
_STATE = {"json_mode_broken": False}


def reset_stats():
    STATS.clear()
    _CHARS.update(chars=0, tokens=0)
    _WARNED.clear()
    _STATE["json_mode_broken"] = False


def _record(agent, seconds, usage, prompt_chars):
    s = STATS.setdefault(agent, {"calls": 0, "prompt_tokens": 0,
                                 "completion_tokens": 0, "seconds": 0.0,
                                 "unreported": 0})
    s["calls"] += 1
    s["seconds"] += seconds
    if isinstance(usage, dict) and usage.get("prompt_tokens") is not None:
        prompt_tokens = int(usage.get("prompt_tokens") or 0)
        s["prompt_tokens"] += prompt_tokens
        s["completion_tokens"] += int(usage.get("completion_tokens") or 0)
        if prompt_tokens > 0:
            _CHARS["chars"] += prompt_chars
            _CHARS["tokens"] += prompt_tokens
    else:
        s["unreported"] += 1


def stats_data(phases=None):
    """Per-agent and total LLM usage (and phase timings) as plain data."""
    agents = {a: dict(v) for a, v in STATS.items()}
    total = {k: sum(v[k] for v in agents.values())
             for k in ("calls", "prompt_tokens", "completion_tokens",
                       "seconds", "unreported")}
    return {"agents": agents, "total": total,
            "phases": [{"phase": n, "seconds": s} for n, s in phases or []]}


def stats_markdown(phases=None):
    """The run's LLM usage as a markdown report (printed and saved)."""
    data = stats_data(phases)

    def row(name, v):
        rate = (f"{v['completion_tokens'] / v['seconds']:.1f}"
                if v["seconds"] and v["completion_tokens"] else "-")
        return (f"| {name} | {v['calls']} | {v['prompt_tokens']:,} | "
                f"{v['completion_tokens']:,} | {v['seconds'] / 60:.1f} | "
                f"{rate} |")

    lines = ["# Run statistics", "",
             "| agent | calls | prompt tokens | output tokens | minutes | "
             "output tok/s |", "|---|---|---|---|---|---|"]
    lines += [row(a, v) for a, v in sorted(data["agents"].items())]
    lines.append(row("**total**", data["total"]))
    if data["total"]["unreported"]:
        lines += ["", f"{data['total']['unreported']} call(s) did not report "
                      "token usage; their tokens are not counted."]
    if phases:
        lines += ["", "| phase | minutes |", "|---|---|"]
        lines += [f"| {n} | {s / 60:.1f} |" for n, s in phases]
    return "\n".join(lines) + "\n"


def _chars_per_token():
    """Observed characters per prompt token (3.5 until enough is known)."""
    if _CHARS["tokens"] >= 500:
        return max(_CHARS["chars"] / _CHARS["tokens"], 1.5)
    return 3.5


def _check_context(messages, agent, llm_cfg, agent_cfg=None):
    """Warn (once per agent) when a prompt nears the configured window."""
    window = llm_cfg.get("context_window")
    if not window or agent in _WARNED:
        return
    chars = sum(len(m["content"]) for m in messages)
    estimate = chars / _chars_per_token()
    reserve = int(_max_tokens(llm_cfg, agent_cfg or {}, agent) or 0)
    if estimate + reserve > 0.9 * int(window):
        _WARNED.add(agent)
        print(f"[LLM] Warning: the {agent} prompt is about {estimate:,.0f} "
              f"tokens, close to the {int(window):,}-token context window "
              "(llm.context_window). A longer book may overflow it: lower "
              "book.summary_window, or raise the server's context size.")


def _read_stream(response, agent):
    """Assemble a streamed (SSE) chat completion.

    Returns (content, finish_reason, usage). Prints a progress line every
    PROGRESS_SECONDS. Transport errors propagate to the caller's retry
    handling; the partial text is discarded.
    """
    response.encoding = "utf-8"
    parts, finish, usage, chunks = [], None, None, 0
    started = last_print = time.time()
    try:
        for raw in response.iter_lines(decode_unicode=True):
            if isinstance(raw, bytes):
                raw = raw.decode("utf-8", "replace")
            line = (raw or "").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            try:
                chunk = json.loads(data)
            except ValueError:
                continue
            if chunk.get("usage"):
                usage = chunk["usage"]
            choice = (chunk.get("choices") or [{}])[0]
            piece = (choice.get("delta") or {}).get("content")
            if piece:
                parts.append(piece)
                chunks += 1
            if choice.get("finish_reason"):
                finish = choice["finish_reason"]
            now = time.time()
            if now - last_print >= PROGRESS_SECONDS:
                print(f"[LLM] {agent}: ~{chunks} tokens so far "
                      f"({chunks / max(now - started, 1e-9):.0f} tok/s)")
                last_print = now
    finally:
        response.close()
    return "".join(parts), finish, usage


def _request(messages, agent="writer", model=None, json_mode=False):
    """POST one chat-completions request with retries.

    Returns (text, finish_reason); finish_reason is None when the server
    doesn't report one. Raises like generate(). json_mode asks the server
    for a JSON object when llm.json_mode is on (object-returning calls only).
    """
    cfg = get_config()
    llm_cfg = cfg["llm"]
    base_url = _base_url(llm_cfg)
    # `or {}` tolerates an agent section whose keys are all commented out.
    agent_cfg = (cfg.get("agents") or {}).get(agent) or {}
    # An agent can use its own model (agents.<name>.model), e.g. a different
    # one for the reviewer than the one that wrote the chapter.
    model = model or agent_cfg.get("model") or llm_cfg["model"]
    # Only send a temperature the user configured; otherwise leave it to the
    # server/model (some models have their preferred sampling baked in).
    temperature = agent_cfg.get("temperature")
    timeout = llm_cfg.get("timeout", 300)
    retries = int(llm_cfg.get("retries", 2))
    headers = _headers(_resolve_api_key(llm_cfg))
    stream = bool(llm_cfg.get("stream"))

    payload = {
        "model": model,
        "messages": messages,
        "stream": stream,
    }
    if temperature is not None:
        payload["temperature"] = float(temperature)
    max_tokens = _max_tokens(llm_cfg, agent_cfg, agent)
    if max_tokens:
        payload["max_tokens"] = int(max_tokens)
    if (json_mode and llm_cfg.get("json_mode")
            and not _STATE["json_mode_broken"]):
        payload["response_format"] = {"type": "json_object"}

    # Chat-template controls for reasoning models (Qwen3.x etc.): the
    # template's built-in default is the highest effort, so set
    # llm.reasoning_effort (low/medium/xhigh) to choose. See _template_kwargs
    # for the per-agent defaults and overrides.
    chat_template_kwargs = _template_kwargs(llm_cfg, agent_cfg, agent)
    if chat_template_kwargs:
        payload["chat_template_kwargs"] = chat_template_kwargs

    prompt_chars = sum(len(m["content"]) for m in messages)
    _check_context(messages, agent, llm_cfg, agent_cfg)

    def post():
        return requests.post(f"{base_url}/v1/chat/completions", json=payload,
                             headers=headers, timeout=timeout, stream=stream)

    last_error = None
    server_side = False  # True when the endpoint itself is down/unready
    for attempt in range(retries + 1):
        started = time.time()
        try:
            response = post()
            if (response.status_code == 400 and "response_format" in payload
                    and "response_format" in (response.text or "").lower()):
                # the server doesn't do JSON mode: stop asking, retry now
                payload.pop("response_format")
                _STATE["json_mode_broken"] = True
                print("[LLM] The server rejected response_format; JSON mode "
                      "is off for the rest of this run.")
                response = post()
            if response.status_code == 200:
                if stream:
                    content, finish, usage = _read_stream(response, agent)
                    if not content and finish is None:
                        raise ValueError("the stream contained no content")
                else:
                    data = response.json()
                    choice = data.get("choices", [{}])[0]
                    content = choice.get("message", {}).get("content")
                    if content is None:
                        raise ValueError("response has no message content")
                    finish, usage = choice.get("finish_reason"), \
                        data.get("usage")
                # Some servers leak the reasoning block into 'content';
                # never let it reach the book text.
                content = _THINK_RE.sub("", content)
                content = _THINK_OPEN_RE.sub("", content)
                _record(agent, time.time() - started, usage, prompt_chars)
                return content.strip(), finish
            last_error = RuntimeError(
                f"LLM endpoint returned HTTP {response.status_code}: "
                f"{response.text[:200]}"
            )
            if response.status_code == 401:
                last_error = RuntimeError(
                    "LLM endpoint returned HTTP 401 (unauthorized). "
                    "Set llm.api_key in config.yaml or the LLM_API_KEY "
                    "environment variable."
                )
                break  # auth errors won't improve on retry
            if response.status_code == 400 and re.search(
                    r"context|too long|too many tokens|exceed",
                    response.text or "", re.IGNORECASE):
                last_error = RuntimeError(
                    f"{last_error} (the prompt may exceed the model's "
                    "context window: set llm.context_window for an early "
                    "warning, lower book.summary_window, or raise the "
                    "server's context size)")
            if response.status_code != 429 and response.status_code < 500:
                break  # other client errors won't improve on retry
            server_side = True  # 5xx (e.g. 'Loading model') or 429
        except requests.exceptions.ConnectionError:
            server_side = True
            last_error = ConnectionError(
                f"Could not connect to the LLM endpoint at {base_url}. "
                "Is the server running?"
            )
        except requests.exceptions.Timeout:
            server_side = True
            last_error = TimeoutError(
                f"LLM call timed out after {timeout}s"
                + (" without receiving data" if stream else "")
            )
        except requests.exceptions.ChunkedEncodingError:
            server_side = True
            last_error = ConnectionError("the response stream was "
                                         "interrupted")
        except (KeyError, IndexError, ValueError) as e:
            last_error = RuntimeError(f"Malformed response from endpoint: {e}")

        if attempt < retries:
            wait = min(2 ** (attempt + 1), 30)  # 2s, 4s, 8s... capped at 30s
            print(f"[LLM] {last_error} -- retrying in {wait}s "
                  f"({attempt + 1}/{retries})")
            time.sleep(wait)

    if server_side and last_error is not None:
        raise EndpointUnavailable(str(last_error)) from last_error
    raise last_error


def _messages(prompt, system):
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    return messages


def generate(prompt, system=None, agent="writer", model=None,
             json_mode=False):
    """Call the OpenAI-compatible chat completions API with retries.

    Args:
        prompt: user prompt text
        system: optional system prompt
        agent: agent name, used to look up temperature from config
        model: optional model override (defaults to config llm.model)
        json_mode: the reply is a JSON *object*; ask the server to enforce
            that when llm.json_mode is on

    Returns:
        The model's response text (stripped).

    Raises:
        EndpointUnavailable for connection errors, timeouts, and HTTP 5xx
        (the server is down or reloading; retrying later should work).
        ConnectionError / TimeoutError / RuntimeError otherwise, after the
        final retry.
    """
    text, reason = _request(_messages(prompt, system), agent, model,
                            json_mode)
    if reason == "length":
        print(f"[LLM] Warning: the {agent} reply was cut off "
              "(finish_reason=length): it hit max_tokens or the server's "
              "limit. If it was looping, fix the prompt or lower the "
              f"temperature; if it needs more room, raise "
              f"agents.{agent}.max_tokens.")
    return text


def endpoint_ready(timeout=5):
    """True when the endpoint answers /v1/models without a server error."""
    cfg = get_config()
    base_url = _base_url(cfg["llm"])
    headers = _headers(_resolve_api_key(cfg["llm"]))
    try:
        response = requests.get(f"{base_url}/v1/models",
                                headers=headers, timeout=timeout)
        return response.status_code < 500
    except requests.exceptions.RequestException:
        return False


def wait_for_endpoint(max_wait=None, poll=5):
    """Poll the endpoint until it is ready, up to max_wait seconds.

    Used when a call failed because the server was down or reloading its
    model (HTTP 503 'Loading model'). Returns True as soon as the endpoint
    responds, False once max_wait elapses.
    """
    if max_wait is None:
        max_wait = int(get_config()["llm"].get("endpoint_wait", 300))
    deadline = time.time() + max_wait
    while time.time() < deadline:
        if endpoint_ready():
            return True
        time.sleep(poll)
    return endpoint_ready()


def _call_with_wait(call):
    """Run call(); if the endpoint is down, wait for it and retry once.

    Local servers routinely drop connections or return 503 'Loading model'
    while swapping models. Instead of failing the phase, poll the endpoint
    for up to llm.endpoint_wait seconds (default 300). Raises
    EndpointUnavailable if the server never comes back.
    """
    try:
        return call()
    except EndpointUnavailable:
        wait = int(get_config()["llm"].get("endpoint_wait", 300))
        print(f"[LLM] Endpoint unavailable; waiting up to {wait}s for it "
              "to come back...")
        if not wait_for_endpoint(wait):
            raise
        print("[LLM] Endpoint is back; retrying.")
        return call()


def generate_with_wait(prompt, system=None, agent="writer", model=None,
                       json_mode=False):
    """generate() that waits for a downed endpoint instead of failing fast."""
    if json_mode:
        return _call_with_wait(lambda: generate(
            prompt, system=system, agent=agent, model=model, json_mode=True))
    return _call_with_wait(
        lambda: generate(prompt, system=system, agent=agent, model=model))


def generate_prose(prompt, system=None, agent="writer", model=None,
                   max_continuations=3):
    """generate_with_wait() for long-form text: if the model stops because it
    hit a length limit, ask it to continue and stitch the pieces together,
    so a chapter is never silently cut off mid-sentence."""
    messages = _messages(prompt, system)
    text, reason = _call_with_wait(lambda: _request(messages, agent, model))
    parts = [text]
    for _ in range(max_continuations):
        if reason != "length":
            return "".join(parts).strip()
        print("[LLM] Response hit the length limit; asking the model to "
              "continue...")
        messages = messages + [
            {"role": "assistant", "content": "".join(parts)},
            {"role": "user", "content": "Your reply was cut off. Continue "
             "exactly where it stopped. Do not repeat anything and do not "
             "add commentary."},
        ]
        text, reason = _call_with_wait(
            lambda: _request(messages, agent, model))
        parts.append(text if text.startswith((" ", "\n")) else " " + text)
    if reason == "length":
        print("[LLM] Warning: the response is still cut off after "
              f"{max_continuations} continuations.")
    return "".join(parts).strip()


def _model_matches(listed, wanted):
    """Exact match, or Ollama-style 'mistral' == 'mistral:latest'."""
    return listed == wanted or (":" not in wanted
                                and listed == f"{wanted}:latest")


def _configured_models(default_model):
    """[(who, model)] for the default model and every per-agent override of
    an agent that is switched on (a disabled agent's model isn't needed)."""
    found = [("llm.model", default_model)]
    for name, agent_cfg in (get_config().get("agents") or {}).items():
        agent_cfg = agent_cfg or {}
        if agent_cfg.get("model") and agent_cfg.get("enabled", True):
            found.append((f"agents.{name}.model", str(agent_cfg["model"])))
    return found


def preflight(model=None):
    """Verify the endpoint is reachable and the model is available.

    Returns a list of available model names on success (empty if the
    endpoint does not expose a model list).
    Raises ConnectionError or RuntimeError with a friendly message.
    """
    cfg = get_config()
    llm_cfg = cfg["llm"]
    base_url = _base_url(llm_cfg)
    model = model or llm_cfg["model"]
    headers = _headers(_resolve_api_key(llm_cfg))
    try:
        response = requests.get(f"{base_url}/v1/models",
                                headers=headers, timeout=10)
    except requests.exceptions.RequestException as e:
        raise ConnectionError(
            f"Could not reach the LLM endpoint at {base_url} ({e}). "
            "Start the server and check llm.base_url in config.yaml."
        ) from e
    if response.status_code == 404 or response.status_code == 405:
        # server doesn't expose a model list; nothing more to check
        print(f"[LLM] {base_url} does not expose /v1/models; skipping "
              "model check.")
        return []
    if response.status_code != 200:
        raise RuntimeError(
            f"LLM endpoint health check failed with HTTP "
            f"{response.status_code}: {response.text[:200]}"
        )
    try:
        names = [m.get("id", "") for m in response.json().get("data", [])]
    except ValueError:
        names = []
    if names:
        missing = [(who, m) for who, m in _configured_models(model)
                   if not any(_model_matches(n, m) for n in names)]
        if missing:
            wanted = "; ".join(f"'{m}' ({who})" for who, m in missing)
            raise RuntimeError(
                f"Not available on the endpoint: {wanted}. "
                f"Available models: {', '.join(names)}. "
                "Fix llm.model / agents.<name>.model in config.yaml or pass "
                "--model."
            )
    return names
