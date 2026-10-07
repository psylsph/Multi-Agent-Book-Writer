"""Unit tests for LLM client error classification (no real endpoint)."""

import pytest
import requests

from shared import llm_client
from shared.llm_client import (EndpointUnavailable, generate,
                               generate_with_wait, load_config,
                               wait_for_endpoint)


class _Resp:
    def __init__(self, status_code, text="", content=None):
        self.status_code = status_code
        self.text = text
        self._content = content

    def json(self):
        return {"choices": [{"message": {"content": self._content}}]}


def _use_config(tmp_path, retries=0, endpoint_wait=None):
    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        "llm:\n"
        '  base_url: "http://localhost:9"\n'
        '  model: "test-model"\n'
        "  timeout: 1\n"
        f"  retries: {retries}\n"
        + (f"  endpoint_wait: {endpoint_wait}\n" if endpoint_wait is not None
           else "")
    )
    load_config(cfg)


def test_connection_error_is_endpoint_unavailable(tmp_path, monkeypatch):
    _use_config(tmp_path)
    monkeypatch.setattr(llm_client.requests, "post",
                        lambda *a, **k: (_ for _ in ()).throw(
                            requests.exceptions.ConnectionError("refused")))
    with pytest.raises(EndpointUnavailable):
        generate("hi", agent="writer")


def test_timeout_is_endpoint_unavailable(tmp_path, monkeypatch):
    _use_config(tmp_path)
    monkeypatch.setattr(llm_client.requests, "post",
                        lambda *a, **k: (_ for _ in ()).throw(
                            requests.exceptions.Timeout("too slow")))
    with pytest.raises(EndpointUnavailable):
        generate("hi", agent="writer")


def test_5xx_is_endpoint_unavailable(tmp_path, monkeypatch):
    _use_config(tmp_path)
    monkeypatch.setattr(llm_client.requests, "post",
                        lambda *a, **k: _Resp503())
    with pytest.raises(EndpointUnavailable):
        generate("hi", agent="writer")


class _Resp503:
    status_code = 503
    text = '{"error":{"message":"Loading model"}}'

    def json(self):
        return {}


def test_4xx_is_runtimeerror_not_endpoint_unavailable(tmp_path, monkeypatch):
    _use_config(tmp_path)
    monkeypatch.setattr(llm_client.requests, "post",
                        lambda *a, **k: _Resp400())
    with pytest.raises(RuntimeError) as excinfo:
        generate("hi", agent="writer")
    assert not isinstance(excinfo.value, EndpointUnavailable)


class _Resp400:
    status_code = 400
    text = "bad request"

    def json(self):
        return {}


def test_success_returns_content(tmp_path, monkeypatch):
    _use_config(tmp_path)
    monkeypatch.setattr(llm_client.requests, "post",
                        lambda *a, **k: _Resp200())
    assert generate("hi", agent="writer") == "hello"


class _Resp200:
    status_code = 200
    text = ""

    def json(self):
        return {"choices": [{"message": {"content": "  hello  "}}]}


def test_wait_for_endpoint_true_when_ready(tmp_path, monkeypatch):
    _use_config(tmp_path)
    monkeypatch.setattr(llm_client.requests, "get",
                        lambda *a, **k: _Resp200())
    assert wait_for_endpoint(max_wait=1, poll=0.01) is True


def test_wait_for_endpoint_5xx_is_not_ready(tmp_path, monkeypatch):
    _use_config(tmp_path)
    monkeypatch.setattr(llm_client.requests, "get",
                        lambda *a, **k: _Resp503())
    assert wait_for_endpoint(max_wait=0.05, poll=0.01) is False


def test_wait_for_endpoint_false_when_unreachable(tmp_path, monkeypatch):
    _use_config(tmp_path)
    monkeypatch.setattr(llm_client.requests, "get",
                        lambda *a, **k: (_ for _ in ()).throw(
                            requests.exceptions.ConnectionError("refused")))
    assert wait_for_endpoint(max_wait=0.05, poll=0.01) is False


def test_generate_with_wait_recovers_after_reload(tmp_path, monkeypatch):
    _use_config(tmp_path)
    calls = {"post": 0}

    def fake_post(*a, **k):
        calls["post"] += 1
        if calls["post"] == 1:
            raise requests.exceptions.ConnectionError("down")
        return _Resp200()

    monkeypatch.setattr(llm_client.requests, "post", fake_post)
    monkeypatch.setattr(llm_client.requests, "get",
                        lambda *a, **k: _Resp200())  # endpoint "back"
    assert generate_with_wait("hi", agent="writer") == "hello"
    assert calls["post"] == 2


def test_generate_with_wait_raises_when_server_stays_down(tmp_path,
                                                          monkeypatch):
    _use_config(tmp_path, endpoint_wait=0.05)

    def down(*a, **k):
        raise requests.exceptions.ConnectionError("down")

    monkeypatch.setattr(llm_client.requests, "post", down)
    monkeypatch.setattr(llm_client.requests, "get", down)
    with pytest.raises(EndpointUnavailable):
        generate_with_wait("hi", agent="writer")

# --------------------------------------------------------------- temperature

def _capture_payload(monkeypatch):
    sent = {}

    def fake_post(url, json=None, **kwargs):
        sent.update(json)
        return _Resp(200, content="ok")

    monkeypatch.setattr(llm_client.requests, "post", fake_post)
    return sent


def test_temperature_not_sent_when_unconfigured(tmp_path, monkeypatch):
    _use_config(tmp_path)
    sent = _capture_payload(monkeypatch)
    generate("hi", agent="writer")
    assert "temperature" not in sent


def test_temperature_not_sent_for_empty_agent_section(tmp_path, monkeypatch):
    cfg = tmp_path / "config.yaml"
    cfg.write_text("agents:\n  writer:\n    # temperature: 1.0\n")
    load_config(cfg)
    sent = _capture_payload(monkeypatch)
    generate("hi", agent="writer")  # must not crash on a None section
    assert "temperature" not in sent


def test_configured_temperature_is_sent(tmp_path, monkeypatch):
    cfg = tmp_path / "config.yaml"
    cfg.write_text("agents:\n  writer:\n    temperature: 0.3\n")
    load_config(cfg)
    sent = _capture_payload(monkeypatch)
    generate("hi", agent="writer")
    assert sent["temperature"] == 0.3


# ------------------------------------------------- url / models / truncation

def _post_recorder(monkeypatch, replies):
    """Record request URLs/payloads; return queued (content, finish) replies."""
    calls, queue = [], list(replies)

    class R:
        status_code = 200
        text = ""

        def __init__(self, content, finish):
            self._c, self._f = content, finish

        def json(self):
            return {"choices": [{"message": {"content": self._c},
                                 "finish_reason": self._f}]}

    def fake_post(url, json=None, **kwargs):
        calls.append((url, json))
        return R(*queue.pop(0))

    monkeypatch.setattr(llm_client.requests, "post", fake_post)
    return calls


@pytest.mark.parametrize("base", ["http://h:8080", "http://h:8080/",
                                  "https://api.openai.com/v1",
                                  "https://openrouter.ai/api/v1/"])
def test_base_url_with_or_without_v1(tmp_path, monkeypatch, base):
    cfg = tmp_path / "config.yaml"
    cfg.write_text(f'llm:\n  base_url: "{base}"\n')
    load_config(cfg)
    calls = _post_recorder(monkeypatch, [("ok", "stop")])
    generate("hi")
    assert calls[0][0].endswith("/v1/chat/completions")
    assert "/v1/v1/" not in calls[0][0]


def test_unclosed_think_block_is_stripped(tmp_path, monkeypatch):
    _use_config(tmp_path)
    _post_recorder(monkeypatch, [("Answer.<think>never finished", "length")])
    assert generate("hi") == "Answer."


def test_generate_prose_continues_a_cut_off_reply(tmp_path, monkeypatch):
    _use_config(tmp_path)
    calls = _post_recorder(monkeypatch, [("The door opened and", "length"),
                                         ("she walked in.", "stop")])
    text = llm_client.generate_prose("write")
    assert text == "The door opened and she walked in."
    follow_up = calls[1][1]["messages"]
    assert follow_up[-2] == {"role": "assistant",
                             "content": "The door opened and"}
    assert "cut off" in follow_up[-1]["content"]


def test_generate_prose_gives_up_after_max_continuations(tmp_path,
                                                         monkeypatch):
    _use_config(tmp_path)
    calls = _post_recorder(monkeypatch, [("a", "length")] * 5)
    llm_client.generate_prose("write", max_continuations=2)
    assert len(calls) == 3


def test_generate_prose_waits_for_a_downed_endpoint(tmp_path, monkeypatch):
    _use_config(tmp_path, endpoint_wait=1)
    state = {"n": 0}

    def flaky(*a, **k):
        state["n"] += 1
        if state["n"] == 1:
            raise requests.exceptions.ConnectionError("refused")
        return _Resp(200, content="back")

    monkeypatch.setattr(llm_client.requests, "post", flaky)
    monkeypatch.setattr(llm_client, "wait_for_endpoint", lambda *a, **k: True)
    assert llm_client.generate_prose("hi") == "back"


def test_ollama_latest_tag_matches_bare_model_name():
    assert llm_client._model_matches("mistral:latest", "mistral")
    assert llm_client._model_matches("mistral", "mistral")
    assert not llm_client._model_matches("mistral:7b", "mistral")
    assert not llm_client._model_matches("mistral:latest", "mistral:7b")


# ----------------------------------------------------- per-agent models

def _agents_config(tmp_path, agents):
    cfg = tmp_path / "config.yaml"
    cfg.write_text('llm:\n  base_url: "http://localhost:9"\n'
                   '  model: "main-model"\n  retries: 0\n' + agents)
    load_config(cfg)


def test_agent_model_overrides_the_default(tmp_path, monkeypatch):
    _agents_config(tmp_path, "agents:\n  reviewer:\n    model: judge-model\n")
    calls = _post_recorder(monkeypatch, [("ok", "stop")] * 3)
    generate("hi", agent="reviewer")
    generate("hi", agent="writer")                  # no override
    generate("hi", agent="reviewer", model="explicit")   # argument wins
    assert [c[1]["model"] for c in calls] == ["judge-model", "main-model",
                                              "explicit"]


def _models_response(monkeypatch, names):
    class R:
        status_code = 200
        text = ""

        def json(self):
            return {"data": [{"id": n} for n in names]}

    monkeypatch.setattr(llm_client.requests, "get", lambda *a, **k: R())


def test_preflight_checks_every_configured_agent_model(tmp_path, monkeypatch):
    _agents_config(tmp_path, "agents:\n  reviewer:\n    model: judge-model\n")
    _models_response(monkeypatch, ["main-model"])
    with pytest.raises(RuntimeError) as exc:
        llm_client.preflight()
    assert "judge-model" in str(exc.value)
    assert "agents.reviewer.model" in str(exc.value)
    _models_response(monkeypatch, ["main-model", "judge-model"])
    assert llm_client.preflight() == ["main-model", "judge-model"]


def test_preflight_ignores_the_model_of_a_disabled_agent(tmp_path,
                                                         monkeypatch):
    _agents_config(tmp_path, "agents:\n  reviewer:\n    enabled: false\n"
                             "    model: unused-model\n")
    _models_response(monkeypatch, ["main-model"])
    assert llm_client.preflight() == ["main-model"]


def test_preflight_still_reports_a_missing_default_model(tmp_path,
                                                         monkeypatch):
    _agents_config(tmp_path, "")
    _models_response(monkeypatch, ["something-else"])
    with pytest.raises(RuntimeError, match="main-model"):
        llm_client.preflight()
