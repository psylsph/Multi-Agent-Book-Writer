"""The set of options config.yaml understands, and a checker that reports the
rest. A misspelled key (`revision_round`) is otherwise silently ignored, so
the setting the user thinks they made does nothing.

tests/test_config_files.py keeps this in step with config.example.yaml: every
option here must be documented there, and every key the shipped configs use
must be known here.
"""

import difflib
import re

KNOWN_KEYS = {
    "book": {
        "auto_chapters", "min_chapters", "max_chapters", "num_chapters",
        "words_per_chapter", "word_count_tolerance", "revision_rounds",
        "extra_length_rounds", "max_length_ratio", "review_as_you_go",
        "review_checks",
        "repetition_lint", "name_lint_ignore", "summary_window",
        "extraction_checks", "seed_review", "seed_questions", "seed_auto_rounds",
        "style",
    },
    "llm": {
        "base_url", "model", "api_key", "timeout", "retries",
        "endpoint_wait", "max_tokens", "reasoning_effort",
        "enable_thinking", "stream", "stream_max_seconds", "context_window",
        "json_mode", "sampling",
    },
    "output": {"directory", "filename", "overwrite", "interim", "log"},
    "web_search": {
        "enabled", "searxng_url", "auto_start", "docker_image",
        "queries_per_chapter", "results_per_query", "snippet_chars",
        "categories", "timeout", "double_check", "banned_terms",
        "block_character_names", "min_interval", "providers",
        "parallel_url", "parallel_api_key", "tinyfish_url", "tinyfish_api_key",
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
        "max_length_ratio": 1.5,
        "review_as_you_go": False,
        "review_checks": None,          # None = every check
        "repetition_lint": True,
        "name_lint_ignore": [],
        "summary_window": 8,
        "extraction_checks": True,
        "seed_review": "ask",
        "seed_questions": 5,
        "seed_auto_rounds": 5,
        "style": {},                    # {} = no style checks (see README)
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
        "stream_max_seconds": None,     # cap on one streamed reply; None = no cap
        "context_window": None,         # tokens; None = no guard
        "json_mode": False,
        "sampling": {},                 # {} = send no sampler fields
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
        "block_character_names": True,  # also block the bible's full names
        "min_interval": 1.5,            # seconds between queries
        "providers": ["searxng"],       # searxng | parallel, merged in order
        "parallel_url": "https://mcp.parallel.ai/mcp",
        "parallel_api_key": "",         # "" = the free, keyless rate limit
        "tinyfish_url": "https://api.search.tinyfish.ai/",
        "tinyfish_api_key": "",         # "" = TINYFISH_API_KEY (required)
    },
}

AGENTS = ("seed_reviewer", "architect", "planner", "researcher", "verifier", "writer",
          "extractor",
          "reviewer", "editor")
AGENT_KEYS = {"model", "temperature", "enabled", "reasoning_effort",
              "enable_thinking", "max_tokens", "sampling"}
# Only these stages are optional, so only they have an off switch.
ENABLE_HONOURED = {"researcher", "reviewer", "editor"}

# Nested option names for the two mapping-valued settings, so a typo inside
# them is reported instead of silently doing nothing.
STYLE_KEYS = {
    "enabled", "banned_phrases", "max_phrase_hits",
    "max_hedges_per_1000", "min_sentence_cv", "pov", "min_pov_share",
    "tense", "min_tense_share", "spelling",
}
SAMPLING_KEYS = (
    "top_p", "top_k", "min_p", "min_keep",
    "repeat_penalty", "repeat_last_n",
    "presence_penalty", "frequency_penalty",
    "dry_multiplier", "dry_base", "dry_allowed_length", "dry_penalty_last_n",
    "xtc_probability", "xtc_threshold", "typ_p",
    "dynatemp_range", "dynatemp_exponent",
)

SEARCH_PROVIDERS = ("searxng", "parallel", "tinyfish")   # shared/web_search.PROVIDERS

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
                    elif key == "sampling" and isinstance(agent_cfg[key], dict):
                        for name in agent_cfg[key]:
                            if name not in SAMPLING_KEYS:
                                out.append(
                                    f"unknown key 'agents.{agent}.sampling."
                                    f"{name}' is ignored"
                                    + _suggest(name, SAMPLING_KEYS))
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
        for key, allowed in (("style", STYLE_KEYS),
                             ("sampling", SAMPLING_KEYS)):
            nested = values.get(key)
            if not isinstance(nested, dict):
                continue
            for name in nested:
                if name not in allowed:
                    out.append(f"unknown key '{section}.{key}.{name}' is "
                               "ignored" + _suggest(name, allowed))
        if section == "web_search" and values.get("providers") is not None:
            chosen = values["providers"]
            for name in [chosen] if isinstance(chosen, str) else chosen or ():
                if str(name).strip().lower() not in SEARCH_PROVIDERS:
                    out.append(f"unknown search provider '{name}' in "
                               "web_search.providers is ignored"
                               + _suggest(str(name), SEARCH_PROVIDERS))
    return out


# ------------------------------------------------------------- value types

# Numeric options whose default is None (unset), so DEFAULTS can't say they
# are numbers.
OPTIONAL_NUMBERS = {
    ("llm", "max_tokens"), ("llm", "context_window"),
    ("llm", "stream_max_seconds"),
}
STYLE_NUMBERS = {"max_phrase_hits", "max_hedges_per_1000", "min_sentence_cv",
                 "min_pov_share", "min_tense_share"}
AGENT_NUMBERS = {"temperature", "max_tokens"}
_TRUE, _FALSE = {"true", "yes", "on", "1"}, {"false", "no", "off", "0"}


def _as_number(value):
    """value as an int/float, or None when it is not a number. A bool is not
    a number here (YAML `yes` is True, and int(True) would quietly be 1)."""
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return value
    try:
        text = str(value).strip()
        return int(text) if re.fullmatch(r"[+-]?\d+", text) else float(text)
    except ValueError:
        return None


def _fix_number(holder, key, where, default):
    """Coerce holder[key] to a number in place; report what was done."""
    value = holder[key]
    number = _as_number(value)
    if number is not None:
        if not isinstance(value, (int, float)):
            holder[key] = number            # "800" -> 800, silently fine
        return []
    if default is None:
        del holder[key]
        return [f"'{where}' should be a number, not {value!r}; ignoring it"]
    holder[key] = default
    return [f"'{where}' should be a number, not {value!r}; using the default "
            f"{default!r}"]


def fix_types(cfg):
    """Repair option values of the wrong type IN PLACE and say what was done.

    The code reads options with int()/float()/bool() deep inside a stage, so
    a typo like `revision_rounds: two` used to crash a run hours in, and a
    quoted "false" was truthy. A value that can't be read falls back to its
    default (or is dropped when it has none), with a [CONFIG] warning at
    startup. Expects the sections to be dicts already (load_config does that).
    """
    out = []
    for section, defaults in DEFAULTS.items():
        values = cfg.get(section)
        if not isinstance(values, dict):
            continue
        for key in list(values):
            if values[key] is None or key not in defaults:
                continue
            default = defaults[key]
            where = f"{section}.{key}"
            if isinstance(default, bool):
                value = values[key]
                if isinstance(value, bool):
                    continue
                text = str(value).strip().lower()
                if text in _TRUE or text in _FALSE:
                    values[key] = text in _TRUE
                else:
                    values[key] = default
                    out.append(f"'{where}' should be true or false, not "
                               f"{value!r}; using the default {default}")
            elif (isinstance(default, (int, float))
                  or (section, key) in OPTIONAL_NUMBERS):
                out += _fix_number(values, key, where,
                                   None if default is None else default)
        style = values.get("style") if section == "book" else None
        if isinstance(style, dict):
            for key in list(style):
                if key in STYLE_NUMBERS and style[key] is not None:
                    out += _fix_number(style, key, f"book.style.{key}", None)
    for agent, agent_cfg in (cfg.get("agents") or {}).items():
        if not isinstance(agent_cfg, dict):
            continue
        for key in list(agent_cfg):
            if key in AGENT_NUMBERS and agent_cfg[key] is not None:
                out += _fix_number(agent_cfg, key, f"agents.{agent}.{key}",
                                   None)
    return out
