"""The set of options config.yaml understands, and a checker that reports the
rest. A misspelled key (`revision_round`) is otherwise silently ignored, so
the setting the user thinks they made does nothing.

tests/test_config_files.py keeps this in step with config.example.yaml: every
option here must be documented there, and every key the shipped configs use
must be known here.
"""

import difflib

KNOWN_KEYS = {
    "book": {
        "auto_chapters", "min_chapters", "max_chapters", "num_chapters",
        "words_per_chapter", "word_count_tolerance", "revision_rounds",
        "extra_length_rounds", "review_as_you_go", "review_checks",
        "repetition_lint", "name_lint_ignore", "summary_window",
        "extraction_checks", "seed_review", "seed_questions",
    },
    "llm": {
        "base_url", "model", "api_key", "timeout", "retries",
        "endpoint_wait", "max_tokens", "reasoning_effort",
        "enable_thinking", "stream", "context_window", "json_mode",
    },
    "output": {"directory", "filename", "overwrite", "interim", "log"},
    "web_search": {
        "enabled", "searxng_url", "auto_start", "docker_image",
        "queries_per_chapter", "results_per_query", "snippet_chars",
        "categories", "timeout", "double_check", "banned_terms",
    },
}

# The value every option takes when config.yaml leaves it out. load_config()
# applies these, so code reads cfg["book"]["words_per_chapter"] directly and a
# default lives in exactly one place. None means "unset": the feature is off
# or the server decides (e.g. no max_tokens is sent).
DEFAULTS = {
    "book": {
        "auto_chapters": True,
        "min_chapters": 3,
        "max_chapters": 30,
        "num_chapters": 5,
        "words_per_chapter": 800,
        "word_count_tolerance": 0.8,
        "revision_rounds": 2,
        "extra_length_rounds": 2,
        "review_as_you_go": False,
        "review_checks": None,          # None = every check
        "repetition_lint": True,
        "name_lint_ignore": [],
        "summary_window": 8,
        "extraction_checks": True,
        "seed_review": "ask",
        "seed_questions": 5,
    },
    "llm": {
        "base_url": "http://localhost:11434",
        "api_key": "",
        "model": "mistral",
        "timeout": 300,
        "retries": 2,
        "endpoint_wait": 300,           # s to wait for a downed server
        "max_tokens": None,
        "reasoning_effort": "",         # "", low, medium, xhigh
        "enable_thinking": None,        # None = don't send it
        "stream": False,
        "context_window": None,         # tokens; None = no guard
        "json_mode": False,
    },
    "output": {
        "directory": "output",
        "filename": "draft.md",
        "overwrite": True,
        "interim": True,
        "log": True,
    },
    "web_search": {
        "enabled": False,
        "searxng_url": "http://localhost:8888",
        "auto_start": "ask",            # ask | yes | no: offer a container
        "docker_image": "searxng/searxng",
        "queries_per_chapter": 3,
        "results_per_query": 4,
        "snippet_chars": 300,
        "categories": "general",
        "timeout": 15,
        "double_check": True,           # second model pass over each verdict
        "banned_terms": [],             # never allowed in a search query
    },
}

AGENTS = ("architect", "planner", "researcher", "verifier", "writer",
          "extractor",
          "reviewer", "editor")
AGENT_KEYS = {"model", "temperature", "enabled", "reasoning_effort",
              "enable_thinking", "max_tokens"}
# Only these stages are optional, so only they have an off switch.
ENABLE_HONOURED = {"researcher", "reviewer", "editor"}

LEGACY_SECTIONS = {"ollama"}  # migrated by load_config
# Keys that were renamed or removed: say what to do instead of guessing.
RETIRED_KEYS = {
    ("web_search", "allow_terms"): (
        "no longer used: every query is allowed unless it mentions a term in "
        "web_search.banned_terms"),
}


def _suggest(word, choices):
    close = difflib.get_close_matches(word, sorted(choices), n=1, cutoff=0.6)
    return f" (did you mean '{close[0]}'?)" if close else ""


def problems(cfg):
    """Human-readable warnings about a raw (parsed) config dict."""
    out = []
    sections = set(KNOWN_KEYS) | {"agents"} | LEGACY_SECTIONS
    for section, values in cfg.items():
        if section not in sections:
            out.append(f"unknown section '{section}' is ignored"
                       + _suggest(section, sections))
            continue
        if section in LEGACY_SECTIONS or not isinstance(values, dict):
            continue
        if section == "agents":
            for agent, agent_cfg in values.items():
                if agent not in AGENTS:
                    out.append(f"unknown agent 'agents.{agent}' is ignored"
                               + _suggest(agent, AGENTS))
                    continue
                for key in (agent_cfg or {}) if isinstance(
                        agent_cfg, dict) else ():
                    if key not in AGENT_KEYS:
                        out.append(f"unknown key 'agents.{agent}.{key}' is "
                                   "ignored" + _suggest(key, AGENT_KEYS))
                    elif key == "enabled" and agent not in ENABLE_HONOURED:
                        out.append(
                            f"'agents.{agent}.enabled' has no effect: this "
                            "stage always runs (only "
                            + ", ".join(sorted(ENABLE_HONOURED))
                            + " can be switched off)")
            continue
        for key in values:
            if (section, key) in RETIRED_KEYS:
                out.append(f"'{section}.{key}' is {RETIRED_KEYS[(section, key)]}")
            elif key not in KNOWN_KEYS[section]:
                out.append(f"unknown key '{section}.{key}' is ignored"
                           + _suggest(key, KNOWN_KEYS[section]))
    return out
