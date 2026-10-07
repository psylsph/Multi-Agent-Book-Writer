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
        "categories", "timeout", "double_check", "allow_terms",
    },
}

AGENTS = ("architect", "planner", "researcher", "writer", "extractor",
          "reviewer", "editor")
AGENT_KEYS = {"model", "temperature", "enabled"}
# Only these stages are optional, so only they have an off switch.
ENABLE_HONOURED = {"researcher", "reviewer", "editor"}

LEGACY_SECTIONS = {"ollama"}  # migrated by load_config


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
            if key not in KNOWN_KEYS[section]:
                out.append(f"unknown key '{section}.{key}' is ignored"
                           + _suggest(key, KNOWN_KEYS[section]))
    return out
