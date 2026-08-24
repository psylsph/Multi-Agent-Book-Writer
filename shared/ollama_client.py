"""Shared configuration and Ollama client.

Single entry point for all LLM calls. Reads config.yaml once, applies
per-agent temperature, and adds timeouts + retries with backoff.
"""

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
    _config.setdefault("ollama", {})
    _config["ollama"].setdefault("api_url", "http://localhost:11434")
    _config["ollama"].setdefault("model", "mistral")
    _config["ollama"].setdefault("timeout", 300)
    _config["ollama"].setdefault("retries", 2)
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


def generate(prompt, system=None, agent="writer", model=None):
    """Call Ollama's generate API with retries.

    Args:
        prompt: user prompt text
        system: optional system prompt
        agent: agent name, used to look up temperature from config
        model: optional model override (defaults to config ollama.model)

    Returns:
        The model's response text (stripped).

    Raises:
        ConnectionError / TimeoutError / RuntimeError after final retry.
    """
    cfg = get_config()
    ollama_cfg = cfg["ollama"]
    api_url = ollama_cfg["api_url"].rstrip("/")
    model = model or ollama_cfg["model"]
    temperature = cfg.get("agents", {}).get(agent, {}).get("temperature", 0.7)
    timeout = ollama_cfg.get("timeout", 300)
    retries = int(ollama_cfg.get("retries", 2))

    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": temperature},
    }
    if system:
        payload["system"] = system

    last_error = None
    for attempt in range(retries + 1):
        try:
            response = requests.post(
                f"{api_url}/api/generate", json=payload, timeout=timeout
            )
            if response.status_code == 200:
                return response.json()["response"].strip()
            last_error = RuntimeError(
                f"Ollama returned HTTP {response.status_code}: {response.text[:200]}"
            )
            if response.status_code < 500:
                break  # client errors won't improve on retry
        except requests.exceptions.ConnectionError:
            last_error = ConnectionError(
                f"Could not connect to Ollama at {api_url}. "
                "Is `ollama serve` running?"
            )
        except requests.exceptions.Timeout:
            last_error = TimeoutError(f"Ollama call timed out after {timeout}s")
        except (KeyError, ValueError) as e:
            last_error = RuntimeError(f"Malformed response from Ollama: {e}")

        if attempt < retries:
            wait = 2 * (attempt + 1)
            print(f"[OLLAMA] {last_error} -- retrying in {wait}s "
                  f"({attempt + 1}/{retries})")
            time.sleep(wait)

    raise last_error


def preflight(model=None):
    """Verify Ollama is reachable and the model is available.

    Returns a list of available model names on success.
    Raises ConnectionError or RuntimeError with a friendly message.
    """
    cfg = get_config()
    api_url = cfg["ollama"]["api_url"].rstrip("/")
    model = model or cfg["ollama"]["model"]
    try:
        response = requests.get(f"{api_url}/api/tags", timeout=10)
    except requests.exceptions.RequestException as e:
        raise ConnectionError(
            f"Could not reach Ollama at {api_url} ({e}). "
            "Start it with `ollama serve` and try again."
        ) from e
    if response.status_code != 200:
        raise RuntimeError(
            f"Ollama health check failed with HTTP {response.status_code}"
        )
    names = [m.get("name", "") for m in response.json().get("models", [])]
    base = model.split(":")[0]
    if not any(n == model or n.split(":")[0] == base for n in names):
        available = ", ".join(names) or "(none)"
        raise RuntimeError(
            f"Model '{model}' is not available in Ollama. "
            f"Available models: {available}. "
            f"Pull it with `ollama pull {model}` or set ollama.model in config.yaml."
        )
    return names
