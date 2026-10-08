"""The shipped config files must load, and stay in sync with each other."""

import re
from pathlib import Path

import pytest
import yaml

from shared.llm_client import load_config

ROOT = Path(__file__).resolve().parent.parent
EXAMPLE = ROOT / "config.example.yaml"
ACTIVE = ROOT / "config.yaml"   # local and untracked: absent in CI
# The checks on config.yaml still catch typos in a developer's own copy.
needs_active = pytest.mark.skipif(not ACTIVE.exists(),
                                  reason="no local config.yaml")


def _raw(path):
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def test_example_config_loads_with_documented_defaults():
    cfg = load_config(EXAMPLE)
    assert cfg["book"]["words_per_chapter"] == 2000
    assert cfg["llm"]["model"]
    assert cfg["output"]["interim"] is True


@needs_active
def test_local_config_loads():
    cfg = load_config(ACTIVE)
    assert cfg["llm"]["base_url"] and cfg["llm"]["model"]


def _documented_keys():
    """Option names in the example, including commented-out ones."""
    return set(re.findall(r"^\s*#?\s*([a-z_]+):", EXAMPLE.read_text(),
                          re.MULTILINE))


@needs_active
def test_config_only_uses_keys_the_example_documents():
    """Catches typos and stale keys in config.yaml (e.g. 'num_chapter')."""
    example, active, documented = _raw(EXAMPLE), _raw(ACTIVE), \
        _documented_keys()
    for section, values in active.items():
        assert section in example, f"unknown section '{section}'"
        for key in (values or {}):
            assert key in documented, \
                f"'{section}.{key}' is not documented in config.example.yaml"


def test_example_does_not_set_a_temperature():
    """Temperatures must stay commented out so none is sent by default."""
    for agent, values in (_raw(EXAMPLE).get("agents") or {}).items():
        assert "temperature" not in (values or {}), agent


def test_missing_config_error_points_at_the_example(tmp_path):
    with pytest.raises(FileNotFoundError, match="config.example.yaml"):
        load_config(tmp_path / "nope.yaml")


def test_enabled_is_only_used_for_agents_that_honour_it():
    """Only these agents read `enabled`; elsewhere it would silently do
    nothing (see main.agent_enabled and the editor's reviewer check)."""
    honoured = {"researcher", "reviewer", "editor"}
    for path in [EXAMPLE] + ([ACTIVE] if ACTIVE.exists() else []):
        for agent, values in (_raw(path).get("agents") or {}).items():
            if "enabled" in (values or {}):
                assert agent in honoured, f"{path.name}: agents.{agent}"
