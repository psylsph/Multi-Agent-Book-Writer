"""Per-agent thinking effort, thinking switch and max_tokens."""

import pytest

from shared import config_schema, llm_client
from shared.llm_client import generate, load_config
from tests.fake_llm import FakeResponse


def _config(tmp_path, llm="", agents=""):
    cfg = tmp_path / "config.yaml"
    cfg.write_text('llm:\n  base_url: "http://localhost:9"\n  model: m\n'
                   f"  retries: 0\n{llm}agents:\n{agents}" if agents else
                   'llm:\n  base_url: "http://localhost:9"\n  model: m\n'
                   f"  retries: 0\n{llm}")
    load_config(cfg)
    llm_client.reset_stats()


@pytest.fixture
def sent(monkeypatch):
    calls = []

    def fake(url, json=None, **kwargs):
        calls.append(json)
        return FakeResponse(data={"choices": [{"message": {"content": "ok"},
                                               "finish_reason": "stop"}]})

    monkeypatch.setattr(llm_client.requests, "post", fake)
    return calls


def kwargs_for(sent, agent):
    generate("hi", agent=agent)
    return sent[-1].get("chat_template_kwargs")


ALL_AGENTS = ("architect", "planner", "researcher", "verifier", "writer",
              "extractor", "reviewer", "editor")


def test_nothing_is_sent_when_no_effort_is_configured(tmp_path, sent):
    _config(tmp_path)
    for agent in ALL_AGENTS:
        assert kwargs_for(sent, agent) is None, agent


def test_the_built_in_defaults_lower_only_three_stages(tmp_path, sent):
    _config(tmp_path, "  reasoning_effort: xhigh\n")
    got = {a: kwargs_for(sent, a)["reasoning_effort"] for a in ALL_AGENTS}
    assert got == {"architect": "xhigh", "planner": "xhigh",
                   "writer": "xhigh", "editor": "xhigh",
                   "researcher": "low", "verifier": "low",
                   "extractor": "low", "reviewer": "medium"}


def test_a_per_agent_value_beats_the_default_and_the_global(tmp_path, sent):
    _config(tmp_path, "  reasoning_effort: xhigh\n",
            "  reviewer:\n    reasoning_effort: xhigh\n"
            "  writer:\n    reasoning_effort: medium\n"
            "  extractor:\n    reasoning_effort: MEDIUM\n")
    assert kwargs_for(sent, "reviewer") == {"reasoning_effort": "xhigh"}
    assert kwargs_for(sent, "writer") == {"reasoning_effort": "medium"}
    assert kwargs_for(sent, "extractor") == {"reasoning_effort": "medium"}
    assert kwargs_for(sent, "researcher") == {"reasoning_effort": "low"}


def test_an_empty_per_agent_value_sends_nothing_for_that_agent(tmp_path, sent):
    _config(tmp_path, "  reasoning_effort: xhigh\n",
            '  extractor:\n    reasoning_effort: ""\n'
            "  reviewer:\n    reasoning_effort:\n")
    assert kwargs_for(sent, "extractor") is None
    assert kwargs_for(sent, "reviewer") is None
    assert kwargs_for(sent, "writer") == {"reasoning_effort": "xhigh"}


def test_defaults_never_start_sending_controls_the_user_left_off(tmp_path,
                                                                 sent):
    _config(tmp_path, '  reasoning_effort: ""\n')
    assert kwargs_for(sent, "researcher") is None
    # ...but an explicit per-agent value is honoured even then
    _config(tmp_path, '  reasoning_effort: ""\n',
            "  researcher:\n    reasoning_effort: low\n")
    assert kwargs_for(sent, "researcher") == {"reasoning_effort": "low"}


def test_enable_thinking_can_be_set_per_agent(tmp_path, sent):
    _config(tmp_path, "  enable_thinking: true\n",
            "  extractor:\n    enable_thinking: false\n")
    assert kwargs_for(sent, "extractor") == {"enable_thinking": False}
    assert kwargs_for(sent, "writer") == {"enable_thinking": True}
    _config(tmp_path, "", "  extractor:\n    enable_thinking: false\n")
    assert kwargs_for(sent, "extractor") == {"enable_thinking": False}
    assert kwargs_for(sent, "writer") is None


def test_max_tokens_can_be_set_per_agent(tmp_path, sent):
    _config(tmp_path, "  max_tokens: 4000\n",
            "  extractor:\n    max_tokens: 1500\n")
    generate("hi", agent="extractor")
    generate("hi", agent="writer")
    assert [c["max_tokens"] for c in sent] == [1500, 4000]
    _config(tmp_path)
    generate("hi", agent="writer")
    assert "max_tokens" not in sent[-1]


def test_the_context_guard_reserves_the_agents_own_max_tokens(tmp_path,
                                                              capsys):
    _config(tmp_path, "  context_window: 1000\n",
            "  extractor:\n    max_tokens: 900\n")
    cfg = llm_client.get_config()
    msgs = [{"role": "user", "content": "x" * 1000}]       # ~285 tokens
    llm_client._check_context(msgs, "writer", cfg["llm"], {})
    assert capsys.readouterr().out == ""
    llm_client._check_context(msgs, "extractor", cfg["llm"],
                              cfg["agents"]["extractor"])
    assert "the extractor prompt" in capsys.readouterr().out


def test_the_schema_knows_the_new_agent_keys(tmp_path, capsys):
    cfg = {"agents": {"extractor": {"reasoning_effort": "low",
                                    "enable_thinking": False,
                                    "max_tokens": 1500}}}
    assert config_schema.problems(cfg) == []
    assert {"reasoning_effort", "enable_thinking", "max_tokens"} <= \
        config_schema.AGENT_KEYS


# ------------------------------------------- ceilings for short-reply agents

@pytest.mark.parametrize("agent,ceiling", [
    ("researcher", 6000), ("extractor", 6000), ("verifier", 2000),
    ("reviewer", 8000)])
def test_short_reply_agents_get_a_built_in_output_ceiling(tmp_path, sent,
                                                          agent, ceiling):
    _config(tmp_path)
    generate("hi", agent=agent)
    assert sent[-1]["max_tokens"] == ceiling


@pytest.mark.parametrize("agent", ["writer", "editor", "architect", "planner"])
def test_long_thinking_agents_are_not_capped_by_default(tmp_path, sent, agent):
    _config(tmp_path)
    generate("hi", agent=agent)
    assert "max_tokens" not in sent[-1]


def test_a_configured_value_always_beats_the_ceiling(tmp_path, sent):
    _config(tmp_path, "  max_tokens: 3000\n",
            "  reviewer:\n    max_tokens: 12000\n")
    generate("hi", agent="reviewer")        # the agent's own value wins
    generate("hi", agent="extractor")       # llm.max_tokens beats the default
    generate("hi", agent="writer")          # ...and applies to everyone
    assert [c["max_tokens"] for c in sent] == [12000, 3000, 3000]


def test_the_truncation_warning_names_the_agent_and_the_remedy(tmp_path,
                                                               monkeypatch,
                                                               capsys):
    _config(tmp_path)
    monkeypatch.setattr(llm_client.requests, "post", lambda *a, **k:
                        FakeResponse(data={"choices": [{
                            "message": {"content": "{\"verdict\": "},
                            "finish_reason": "length"}]}))
    generate("hi", agent="reviewer")
    out = capsys.readouterr().out
    assert "the reviewer reply was cut off" in out
    assert "agents.reviewer.max_tokens" in out and "looping" in out
