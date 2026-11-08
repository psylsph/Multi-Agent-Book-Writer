"""Tests for the browser question channel (shared/webanswer.py).

The channel is plain files: the pipeline writes question.json and polls
answer.json. These tests exercise the protocol and the seam swapping; the web
app's side of it is in tests/test_web_app.py.
"""

import json
import threading
import time
from pathlib import Path

import pytest

from shared import webanswer


@pytest.fixture
def seams(monkeypatch):
    """The two prompts install() replaces, put back after the test."""
    from agents import seed_review
    from shared import web_search
    monkeypatch.setattr(seed_review, "_ask", seed_review._ask)
    monkeypatch.setattr(seed_review, "interactive", lambda: False)
    monkeypatch.setitem(seed_review._INPUT, "closed", True)
    monkeypatch.setattr(web_search, "_ask", lambda question: False)


def question(directory):
    return json.loads((Path(directory) / webanswer.QUESTION).read_text())


def answer_next(directory, text, poll=0.01, timeout=5):
    """A stand-in browser: answer the next question that appears."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            pending = question(directory)
        except OSError:
            time.sleep(poll)
            continue
        webanswer.write_answer(directory, pending["id"], text)
        return
    raise AssertionError("no question was asked")


def test_question_and_answer_round_trip(tmp_path):
    asked = webanswer.write_question(tmp_path, "  > ", default="d")
    assert question(tmp_path)["id"] == asked["id"]
    assert webanswer.read_answer(tmp_path, asked["id"]) is None

    webanswer.write_answer(tmp_path, asked["id"], "yes")
    assert webanswer.read_answer(tmp_path, asked["id"]) == "yes"

    # both files are consumed, so the page shows nothing pending and the next
    # poll cannot re-deliver the same answer
    assert webanswer.read_question(tmp_path) is None
    assert not (tmp_path / webanswer.ANSWER).exists()


def test_a_stale_answer_is_discarded(tmp_path):
    """Answers are tied to a question id: a page left open across a rerun
    must not answer the new question with the old text."""
    old = webanswer.write_question(tmp_path, ">")
    webanswer.write_answer(tmp_path, old["id"], "the old answer")
    new = webanswer.write_question(tmp_path, ">")
    assert new["id"] != old["id"]

    assert webanswer.read_answer(tmp_path, new["id"]) is None
    assert not (tmp_path / webanswer.ANSWER).exists()
    assert question(tmp_path)["id"] == new["id"]     # still waiting


def test_a_half_written_answer_is_ignored_not_fatal(tmp_path):
    asked = webanswer.write_question(tmp_path, ">")
    (tmp_path / webanswer.ANSWER).write_text('{"id": ', encoding="utf-8")
    assert webanswer.read_answer(tmp_path, asked["id"]) is None


def test_ask_blocks_until_answered(tmp_path):
    threading.Thread(target=answer_next, args=(tmp_path, "banana")).start()
    assert webanswer.ask(">", directory=tmp_path, timeout=5, poll=0.01) == "banana"


def test_ask_keeps_the_question_visible_while_waiting(tmp_path):
    threading.Thread(target=answer_next, args=(tmp_path, "x")).start()
    webanswer.ask("the question", directory=tmp_path, timeout=5, poll=0.01)
    # the answer consumes the question file too
    assert webanswer.read_question(tmp_path) is None


def test_ask_times_out_as_eof_and_clears_the_question(tmp_path):
    """A closed browser must not hang the run (the seed review treats EOF as
    'keep the seed as it is'), and no stale question may stay on the page."""
    with pytest.raises(EOFError):
        webanswer.ask(">", directory=tmp_path, timeout=0.05, poll=0.01)
    assert webanswer.read_question(tmp_path) is None


def test_install_swaps_both_prompts(tmp_path, seams):
    from agents import seed_review
    from shared import web_search

    original_seed_ask, original_web_ask = seed_review._ask, web_search._ask
    assert seed_review.interactive() is False
    webanswer.install(directory=tmp_path, timeout=5)

    assert seed_review._ask is not original_seed_ask
    assert web_search._ask is not original_web_ask
    assert seed_review.interactive() is True          # keeps seed_review: ask
    assert seed_review._INPUT["closed"] is False


def test_the_installed_prompt_goes_through_the_files(tmp_path, seams):
    from agents import seed_review

    webanswer.install(directory=tmp_path, timeout=5)
    threading.Thread(target=answer_next, args=(tmp_path, "banana")).start()

    # _ask(question, default) returns the answer, not the default
    assert seed_review._ask("  > ", "default") == "banana"
    # Enter (an empty answer) falls through to the default, as on a terminal
    threading.Thread(target=answer_next, args=(tmp_path, "")).start()
    assert seed_review._ask("  > ", "d") == "d"


def test_a_silent_browser_takes_the_default_and_stops_asking(tmp_path, seams):
    """No answer must not hang a run: the review takes its default and, like a
    closed stdin, stops rather than expanding the seed forever."""
    from agents import seed_review

    webanswer.install(directory=tmp_path, timeout=0.05)
    seed_review._INPUT["closed"] = False
    assert seed_review._ask("  > ", "d") == "d"
    assert seed_review._INPUT["closed"] is True


def test_a_yes_no_prompt_is_a_boolean(tmp_path, seams):
    from shared import web_search

    webanswer.install(directory=tmp_path, timeout=5)
    threading.Thread(target=answer_next, args=(tmp_path, "y")).start()
    assert web_search._ask("Download the image?") is True
    threading.Thread(target=answer_next, args=(tmp_path, "no")).start()
    assert web_search._ask("Download the image?") is False


def test_install_if_enabled_does_nothing_without_the_env_var(monkeypatch, seams):
    from agents import seed_review
    original = seed_review._ask
    monkeypatch.delenv(webanswer.ENV_DIR, raising=False)
    assert webanswer.install_if_enabled() is None
    assert seed_review._ask is original


def test_install_if_enabled_uses_the_env_directory(tmp_path, monkeypatch, seams):
    from agents import seed_review
    directory = tmp_path / "ask"
    monkeypatch.setenv(webanswer.ENV_DIR, str(directory))
    monkeypatch.setenv(webanswer.ENV_TIMEOUT, "5")
    assert webanswer.install_if_enabled() == directory
    assert seed_review.interactive() is True
    assert directory.is_dir()


def test_main_installs_the_browser_channel_before_parsing(monkeypatch):
    """main.main() must swap the prompts whatever the other arguments are."""
    import sys

    import main
    calls = []
    monkeypatch.setattr(main.webanswer, "install_if_enabled",
                        lambda: calls.append(True))
    monkeypatch.setattr(sys, "argv", ["main.py", "--help"])
    with pytest.raises(SystemExit):
        main.main()
    assert calls == [True]
