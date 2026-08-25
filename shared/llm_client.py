"""Shared configuration and LLM client.

Single entry point for all LLM calls. Works with any OpenAI-compatible
endpoint (LM Studio, llama.cpp server, vLLM, Ollama's /v1 API, OpenAI,
OpenRouter, ...). Reads config.yaml once, applies per-agent temperature,
and adds timeouts + retries with backoff.
"""

import os
import time
from pathlib import Path

import requests
import yaml

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.yaml"

_config = None


def load_config(path=None):
    """Load (or reload) the configuration. Called once by main.py at startup."""
    global _config
    config_path = Path(path) if path else DEFAULT_CONFIG_PATH
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        _config = yaml.safe_load(f) or {}

    # sane defaults so a partial config file doesn't crash the pipeline
    _config.setdefault("book", {})
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
    _config.setdefault("agents", {})
    _config.setdefault("output", {})
    _config["output"].setdefault("directory", "output")
    _config["output"].setdefault("filename", "draft.md")
    _config["output"].setdefault("overwrite", True)
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


def _headers(api_key):
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


def generate(prompt, system=None, agent="writer", model=None):
    """Call the OpenAI-compatible chat completions API with retries.

    Args:
        prompt: user prompt text
        system: optional system prompt
        agent: agent name, used to look up temperature from config
        model: optional model override (defaults to config llm.model)

    Returns:
        The model's response text (stripped).

    Raises:
        ConnectionError / TimeoutError / RuntimeError after final retry.
    """
    cfg = get_config()
    llm_cfg = cfg["llm"]
    base_url = llm_cfg["base_url"].rstrip("/")
    model = model or llm_cfg["model"]
    temperature = cfg.get("agents", {}).get(agent, {}).get("temperature", 0.7)
    timeout = llm_cfg.get("timeout", 300)
    retries = int(llm_cfg.get("retries", 2))
    headers = _headers(_resolve_api_key(llm_cfg))

    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "stream": False,
    }

    last_error = None
    for attempt in range(retries + 1):
        try:
            response = requests.post(
                f"{base_url}/v1/chat/completions", json=payload,
                headers=headers, timeout=timeout,
            )
            if response.status_code == 200:
                content = (
                    response.json()
                    .get("choices", [{}])[0]
                    .get("message", {})
                    .get("content")
                )
                if content is None:
                    raise ValueError("response has no message content")
                return content.strip()
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
            if response.status_code != 429 and response.status_code < 500:
                break  # other client errors won't improve on retry
        except requests.exceptions.ConnectionError:
            last_error = ConnectionError(
                f"Could not connect to the LLM endpoint at {base_url}. "
                "Is the server running?"
            )
        except requests.exceptions.Timeout:
            last_error = TimeoutError(
                f"LLM call timed out after {timeout}s"
            )
        except (KeyError, IndexError, ValueError) as e:
            last_error = RuntimeError(f"Malformed response from endpoint: {e}")

        if attempt < retries:
            wait = 2 * (attempt + 1)
            print(f"[LLM] {last_error} -- retrying in {wait}s "
                  f"({attempt + 1}/{retries})")
            time.sleep(wait)

    raise last_error


def preflight(model=None):
    """Verify the endpoint is reachable and the model is available.

    Returns a list of available model names on success (empty if the
    endpoint does not expose a model list).
    Raises ConnectionError or RuntimeError with a friendly message.
    """
    cfg = get_config()
    llm_cfg = cfg["llm"]
    base_url = llm_cfg["base_url"].rstrip("/")
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
    if names and not any(n == model for n in names):
        available = ", ".join(names) or "(none)"
        raise RuntimeError(
            f"Model '{model}' is not available on the endpoint. "
            f"Available models: {available}. "
            "Set llm.model in config.yaml or pass --model."
        )
    return names
