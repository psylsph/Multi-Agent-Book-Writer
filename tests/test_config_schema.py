"""Config validation: misspelled and meaningless options are reported."""

import re
from pathlib import Path

import yaml

from shared import config_schema
from shared.llm_client import load_config

ROOT = Path(__file__).resolve().parent.parent
EXAMPLE = ROOT / "config.example.yaml"


def test_a_clean_config_has_no_problems():
    cfg = {"book": {"revision_rounds": 2}, "llm": {"model": "m"},
           "agents": {"writer": {"temperature": 1.0},
                      "editor": {"enabled": True, "model": "x"}},
           "output": {"log": True}, "web_search": {"enabled": False},
           "ollama": {"api_url": "http://old"}}            # legacy section
    assert config_schema.problems(cfg) == []


def test_misspelled_keys_get_a_suggestion():
    out = config_schema.problems({"book": {"revision_round": 3},
                                  "llm": {"modle": "x"}})
    assert "unknown key 'book.revision_round' is ignored (did you mean " \
        "'revision_rounds'?)" in out
    assert any("'llm.modle'" in m and "did you mean 'model'" in m for m in out)


def test_unknown_sections_and_agents_are_reported():
    out = config_schema.problems({"outputs": {}, "agents": {"writter": {}}})
    assert any("unknown section 'outputs'" in m and "'output'" in m
               for m in out)
    assert any("unknown agent 'agents.writter'" in m and "'writer'" in m
               for m in out)


def test_enabled_on_an_always_on_stage_is_flagged():
    out = config_schema.problems({"agents": {
        "writer": {"enabled": False}, "reviewer": {"enabled": False},
        "extractor": {"tempreture": 1}}})
    assert any("'agents.writer.enabled' has no effect" in m for m in out)
    assert not any("agents.reviewer.enabled" in m for m in out)   # honoured
    assert any("'agents.extractor.tempreture'" in m for m in out)


def test_the_retired_allow_terms_key_says_what_replaced_it():
    out = config_schema.problems({"web_search": {"allow_terms": ["UK"]}})
    assert len(out) == 1 and "banned_terms" in out[0]
    assert "unknown key" not in out[0]


def test_unrelated_garbage_has_no_suggestion():
    (msg,) = config_schema.problems({"book": {"zzzzzz": 1}})
    assert msg == "unknown key 'book.zzzzzz' is ignored"


def test_load_config_prints_warnings_but_still_loads(tmp_path, capsys):
    path = tmp_path / "config.yaml"
    path.write_text("book:\n  revision_round: 3\n  num_chapters: 7\n")
    cfg = load_config(path)
    assert "[CONFIG] unknown key 'book.revision_round'" in capsys.readouterr().out
    assert cfg["book"]["num_chapters"] == 7            # the rest still applies


def test_sections_with_only_comments_do_not_warn(tmp_path, capsys):
    path = tmp_path / "config.yaml"
    path.write_text("agents:\n  writer:\n    # temperature: 1.0\nweb_search:\n")
    load_config(path)
    assert "[CONFIG]" not in capsys.readouterr().out


# ------------------------------------- the schema and the example stay in step

def _documented_keys():
    return set(re.findall(r"^\s*#?\s*([a-z_]+):", EXAMPLE.read_text(),
                          re.MULTILINE))


def test_every_known_option_is_documented_in_the_example():
    documented = _documented_keys()
    for section, keys in config_schema.KNOWN_KEYS.items():
        for key in keys:
            assert key in documented, f"{section}.{key} missing from the example"
    for key in config_schema.AGENT_KEYS:
        assert key in documented


def test_the_shipped_configs_only_use_known_options():
    for name in ("config.example.yaml", "config.yaml"):
        if not (ROOT / name).exists():
            continue  # config.yaml is local and untracked
        cfg = yaml.safe_load((ROOT / name).read_text()) or {}
        assert config_schema.problems(cfg) == [], name


def test_the_example_lists_every_agent():
    agents = yaml.safe_load(EXAMPLE.read_text())["agents"]
    assert set(agents) == set(config_schema.AGENTS)


def test_every_default_is_a_known_option():
    for section, defaults in config_schema.DEFAULTS.items():
        assert set(defaults) <= config_schema.KNOWN_KEYS[section], section


def test_an_empty_config_gets_every_default(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("")
    cfg = load_config(path)
    for section, defaults in config_schema.DEFAULTS.items():
        for key, value in defaults.items():
            assert cfg[section][key] == value, f"{section}.{key}"


def test_mutable_defaults_are_not_shared(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("")
    load_config(path)["book"]["name_lint_ignore"].append("Zed")
    assert config_schema.DEFAULTS["book"]["name_lint_ignore"] == []
