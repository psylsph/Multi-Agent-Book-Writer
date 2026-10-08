"""Streaming, run statistics, context guard, JSON mode and the run log."""

import json
import sys

import pytest
import requests

from shared import llm_client, runlog
from shared.llm_client import EndpointUnavailable, generate, load_config
from tests.fake_llm import FakeResponse


def _config(tmp_path, llm=""):
    cfg = tmp_path / "config.yaml"
    cfg.write_text('llm:\n  base_url: "http://localhost:9"\n  model: m\n'
                   f"  retries: 0\n{llm}")
    load_config(cfg)
    llm_client.reset_stats()


def sse(**delta):
    return "data: " + json.dumps(delta)


def chunk(text):
    return sse(choices=[{"delta": {"content": text}}])


def finish(reason="stop", **extra):
    return sse(choices=[{"delta": {}, "finish_reason": reason}], **extra)


def post_returns(monkeypatch, response):
    seen = []

    def fake(url, json=None, **kwargs):
        seen.append({"json": json, **kwargs})
        return response

    monkeypatch.setattr(llm_client.requests, "post", fake)
    return seen


# ------------------------------------------------------------------ streaming

def test_stream_assembles_chunks_finish_reason_and_usage(tmp_path,
                                                         monkeypatch):
    _config(tmp_path, "  stream: true\n")
    response = FakeResponse(lines=[
        ": keep-alive", "", chunk("Hello, "), "not an sse line",
        "data: {broken json", chunk("world."),
        finish("length", usage={"prompt_tokens": 12, "completion_tokens": 3}),
        "data: [DONE]"])
    seen = post_returns(monkeypatch, response)
    text, reason = llm_client._request(
        [{"role": "user", "content": "hi"}], "writer")
    assert (text, reason) == ("Hello, world.", "length")
    assert seen[0]["json"]["stream"] is True and seen[0]["stream"] is True
    assert response.closed
    assert llm_client.STATS["writer"]["prompt_tokens"] == 12
    assert llm_client.STATS["writer"]["completion_tokens"] == 3


def test_stream_strips_thinking_like_a_normal_reply(tmp_path, monkeypatch):
    _config(tmp_path, "  stream: true\n")
    post_returns(monkeypatch, FakeResponse(lines=[
        chunk("<think>hmm"), chunk(" ok</think>Final."), finish()]))
    assert generate("hi") == "Final."


def test_stream_prints_progress(tmp_path, monkeypatch, capsys):
    _config(tmp_path, "  stream: true\n")
    monkeypatch.setattr(llm_client, "PROGRESS_SECONDS", 0)
    post_returns(monkeypatch, FakeResponse(lines=[chunk("a"), chunk("b"),
                                                  finish()]))
    generate("hi", agent="writer")
    assert "[LLM] writer: ~" in capsys.readouterr().out


def test_stream_counts_thinking_and_stays_quiet_when_nothing_arrived(
        tmp_path, monkeypatch, capsys):
    _config(tmp_path, "  stream: true\n")
    monkeypatch.setattr(llm_client, "PROGRESS_SECONDS", 0)
    thinking = 'data: {"choices": [{"delta": {"reasoning_content": "hmm"}}]}'
    post_returns(monkeypatch, FakeResponse(lines=[
        'data: {"choices": [{"delta": {"role": "assistant"}}]}',   # no tokens yet
        thinking, thinking, chunk("Done."), finish()]))
    generate("hi", agent="architect")
    out = capsys.readouterr().out
    assert "~0 tokens" not in out
    assert "architect: ~2 tokens so far" in out


def test_a_stream_that_dies_mid_reply_is_an_outage(tmp_path, monkeypatch):
    _config(tmp_path, "  stream: true\n")

    class Dies(FakeResponse):
        def iter_lines(self, decode_unicode=False):
            yield chunk("partial")
            raise requests.exceptions.ChunkedEncodingError("cut")

    post_returns(monkeypatch, Dies())
    with pytest.raises(EndpointUnavailable, match="interrupted"):
        generate("hi")


def test_a_stalled_stream_times_out_with_a_clear_message(tmp_path,
                                                         monkeypatch):
    _config(tmp_path, "  stream: true\n  timeout: 30\n")

    class Stalls(FakeResponse):
        def iter_lines(self, decode_unicode=False):
            raise requests.exceptions.ReadTimeout("no data")
            yield

    post_returns(monkeypatch, Stalls())
    with pytest.raises(EndpointUnavailable,
                       match="timed out after 30s without receiving data"):
        generate("hi")


def test_an_empty_stream_is_an_error_not_an_empty_chapter(tmp_path,
                                                          monkeypatch):
    _config(tmp_path, "  stream: true\n")
    post_returns(monkeypatch, FakeResponse(lines=["data: [DONE]"]))
    with pytest.raises(RuntimeError, match="no content"):
        generate("hi")


def test_stream_is_off_by_default(tmp_path, monkeypatch):
    _config(tmp_path)
    seen = post_returns(monkeypatch, FakeResponse(data={
        "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}]}))
    generate("hi")
    assert seen[0]["json"]["stream"] is False


# ---------------------------------------------------------------------- stats

def test_stats_accumulate_per_agent_and_render(tmp_path):
    _config(tmp_path)
    llm_client._record("writer", 10.0, {"prompt_tokens": 100,
                                        "completion_tokens": 400}, 350)
    llm_client._record("writer", 10.0, {"prompt_tokens": 50,
                                        "completion_tokens": 100}, 175)
    llm_client._record("reviewer", 5.0, None, 99)          # no usage reported
    data = llm_client.stats_data([("writer", 20.0)])
    assert data["agents"]["writer"] == {
        "calls": 2, "prompt_tokens": 150, "completion_tokens": 500,
        "seconds": 20.0, "unreported": 0}
    assert data["total"]["calls"] == 3 and data["total"]["unreported"] == 1
    assert data["phases"] == [{"phase": "writer", "seconds": 20.0}]
    text = llm_client.stats_markdown([("writer", 90.0)])
    assert "| writer | 2 | 150 | 500 | 0.3 | 25.0 |" in text
    assert "| reviewer | 1 | 0 | 0 | 0.1 | - |" in text
    assert "1 call(s) did not report token usage" in text
    assert "| writer | 1.5 |" in text                      # phase minutes


def test_reset_stats_clears_everything(tmp_path):
    _config(tmp_path)
    llm_client._record("writer", 1.0, {"prompt_tokens": 5,
                                       "completion_tokens": 5}, 20)
    llm_client._STATE["json_mode_broken"] = True
    llm_client.reset_stats()
    assert llm_client.STATS == {} and not llm_client._STATE["json_mode_broken"]
    assert llm_client._chars_per_token() == 3.5


# -------------------------------------------------------------- context guard

def test_chars_per_token_calibrates_from_real_usage(tmp_path):
    _config(tmp_path)
    assert llm_client._chars_per_token() == 3.5            # no data yet
    llm_client._record("writer", 1.0, {"prompt_tokens": 1000,
                                       "completion_tokens": 1}, 4000)
    assert llm_client._chars_per_token() == 4.0


def _msgs(chars):
    return [{"role": "user", "content": "x" * chars}]


def test_context_guard_is_silent_without_a_window(tmp_path, capsys):
    _config(tmp_path)
    llm_client._check_context(_msgs(10 ** 6), "writer",
                              llm_client.get_config()["llm"])
    assert capsys.readouterr().out == ""


def test_context_guard_warns_once_per_agent_and_only_when_close(tmp_path,
                                                                capsys):
    _config(tmp_path, "  context_window: 1000\n")
    llm = llm_client.get_config()["llm"]
    llm_client._check_context(_msgs(1000), "writer", llm)    # ~285 tokens
    assert capsys.readouterr().out == ""
    llm_client._check_context(_msgs(3500), "writer", llm)    # ~1000 tokens
    llm_client._check_context(_msgs(3500), "writer", llm)    # not repeated
    llm_client._check_context(_msgs(3500), "reviewer", llm)  # other agent
    out = capsys.readouterr().out
    assert out.count("the writer prompt") == 1
    assert out.count("the reviewer prompt") == 1
    assert "lower book.summary_window" in out


def test_context_guard_counts_the_reserved_output(tmp_path, capsys):
    _config(tmp_path, "  context_window: 1000\n  max_tokens: 800\n")
    llm_client._check_context(_msgs(1000), "writer",
                              llm_client.get_config()["llm"])  # 285 + 800
    assert "the writer prompt" in capsys.readouterr().out


def test_a_context_overflow_error_gets_a_hint(tmp_path, monkeypatch):
    _config(tmp_path)
    post_returns(monkeypatch, FakeResponse(
        400, text="request exceeds the available context size"))
    with pytest.raises(RuntimeError) as exc:
        generate("hi")
    assert "context window" in str(exc.value)
    assert "book.summary_window" in str(exc.value)


# ------------------------------------------------------------------ JSON mode

def _ok():
    return FakeResponse(data={"choices": [{"message": {"content": "{}"},
                                           "finish_reason": "stop"}]})


def test_json_mode_needs_both_the_setting_and_the_flag(tmp_path, monkeypatch):
    _config(tmp_path, "  json_mode: true\n")
    seen = post_returns(monkeypatch, _ok())
    generate("hi", json_mode=True)
    generate("hi")                                   # prose: never JSON mode
    assert seen[0]["json"]["response_format"] == {"type": "json_object"}
    assert "response_format" not in seen[1]["json"]
    _config(tmp_path)                                # setting off
    seen = post_returns(monkeypatch, _ok())
    generate("hi", json_mode=True)
    assert "response_format" not in seen[0]["json"]


def test_json_mode_falls_back_when_the_server_rejects_it(tmp_path,
                                                         monkeypatch, capsys):
    _config(tmp_path, "  json_mode: true\n")
    calls = []

    def fake(url, json=None, **kwargs):
        calls.append(dict(json))
        if "response_format" in json:
            return FakeResponse(400, text="bad field response_format")
        return _ok()

    monkeypatch.setattr(llm_client.requests, "post", fake)
    assert generate("hi", json_mode=True) == "{}"
    generate("hi", json_mode=True)                    # remembered: no retry
    assert ["response_format" in c for c in calls] == [True, False, False]
    assert capsys.readouterr().out.count("rejected response_format") == 1


def test_generate_with_wait_forwards_json_mode(tmp_path, monkeypatch):
    _config(tmp_path, "  json_mode: true\n")
    seen = post_returns(monkeypatch, _ok())
    llm_client.generate_with_wait("hi", json_mode=True)
    assert "response_format" in seen[0]["json"]


# -------------------------------------------------------------------- run log

def test_the_log_captures_stdout_and_stderr_and_restores_them(tmp_path):
    original = (sys.stdout, sys.stderr)
    path = runlog.start_log(tmp_path / "logs")
    try:
        print("hello from stdout")
        print("hello from stderr", file=sys.stderr)
        assert path.parent == tmp_path / "logs" and path.suffix == ".log"
        assert path.name.startswith("run-")
    finally:
        runlog.stop_log()
    assert (sys.stdout, sys.stderr) == original
    text = path.read_text()
    assert "hello from stdout" in text and "hello from stderr" in text


def test_starting_the_log_twice_does_not_nest_streams(tmp_path):
    original = sys.stdout
    runlog.start_log(tmp_path / "a")
    runlog.start_log(tmp_path / "b")
    runlog.stop_log()
    assert sys.stdout is original


def test_an_unwritable_log_directory_never_stops_the_run(tmp_path, capsys):
    blocker = tmp_path / "file"
    blocker.write_text("not a directory")
    assert runlog.start_log(blocker / "logs") is None
    assert "Could not open a log file" in capsys.readouterr().out
    runlog.stop_log()


def test_the_tee_delegates_stream_attributes(tmp_path):
    runlog.start_log(tmp_path)
    try:
        assert isinstance(sys.stdout, runlog._Tee)
        assert isinstance(sys.stdout.isatty(), bool)
        assert sys.stdout.writable() in (True, False)     # delegated
    finally:
        runlog.stop_log()
