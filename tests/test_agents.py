"""Agent and pipeline-control tests with a stubbed LLM (temp dirs only)."""

import argparse
import json

import pytest

import main as pipeline
from agents import editor, writer
from shared.context import context, reset_context, update_context
from shared.llm_client import load_config
from shared.story_state import minimal_state

CHAPTERS = [{"number": n, "title": f"Ch{n}", "summary": f"s{n}", "part": ""}
            for n in (1, 2, 3)]
STATE_JSON = json.dumps({"summary": "Things happened.", "time": "day",
                         "location": "town", "present": ["Aria"],
                         "events": []})


def _config(tmp_path, book="", agents=""):
    path = tmp_path / "config.yaml"
    path.write_text(f'output:\n  directory: "{tmp_path / "out"}"\n'
                    f"book:\n  words_per_chapter: 100\n{book}"
                    f"agents:\n  reviewer:\n    enabled: false\n{agents}")
    load_config(path)
    reset_context()
    return tmp_path / "out"


def _words(n, prefix="alpha"):
    return " ".join(f"{prefix}{i % 7}" for i in range(n))


# --------------------------------------------------------------------- writer

@pytest.fixture
def stub_writer(monkeypatch):
    prose_prompts = []

    def prose(prompt, **kwargs):
        prose_prompts.append(prompt)
        return _words(120)

    monkeypatch.setattr(writer, "generate_prose", prose)
    monkeypatch.setattr(writer, "generate_with_wait",
                        lambda prompt, **k: STATE_JSON)
    return prose_prompts


def _writer_context():
    update_context("chapters", CHAPTERS)
    update_context("bible", {"title": "T", "characters": []})
    update_context("title", "T")


def test_writer_drafts_every_chapter_and_records_state(tmp_path, stub_writer):
    out = _config(tmp_path)
    _writer_context()
    writer.run_writer()
    assert sorted(context["drafts"]) == [1, 2, 3]
    assert sorted(context["chronology"]) == [1, 2, 3]
    assert context["chronology"][1]["summary"] == "Things happened."
    assert (out / "chapters" / "chapter_03.md").exists()
    assert (out / "interim" / "chronology.json").exists()


def test_writer_recap_only_includes_earlier_chapters(tmp_path, stub_writer):
    _config(tmp_path)
    _writer_context()
    update_context("summaries", {1: "ONE-SUMMARY", 3: "THREE-SUMMARY"})
    update_context("chronology", {1: minimal_state(1, "Ch1", "x")})
    update_context("drafts", {1: "d1"})
    writer.run_writer()
    second = stub_writer[0]            # first prompt is chapter 2
    assert "ONE-SUMMARY" in second
    assert "THREE-SUMMARY" not in second


def test_writer_rebuilds_missing_state_without_redrafting(tmp_path,
                                                          stub_writer):
    _config(tmp_path)
    _writer_context()
    update_context("drafts", {1: "kept draft", 2: "kept draft 2",
                              3: "kept draft 3"})
    writer.run_writer()                 # chronology missing for all three
    assert stub_writer == []            # nothing was redrafted
    assert context["drafts"][1] == "kept draft"
    assert sorted(context["chronology"]) == [1, 2, 3]


def test_writer_stops_at_first_failed_chapter(tmp_path, monkeypatch):
    _config(tmp_path)
    _writer_context()
    calls = []

    def prose(prompt, **kwargs):
        calls.append(prompt)
        if len(calls) == 2:
            raise RuntimeError("HTTP 400: context too long")
        return _words(120)

    monkeypatch.setattr(writer, "generate_prose", prose)
    monkeypatch.setattr(writer, "generate_with_wait",
                        lambda prompt, **k: STATE_JSON)
    writer.run_writer()
    assert sorted(context["drafts"]) == [1]      # chapter 3 not attempted
    assert len(calls) == 2


def test_writer_extraction_retries_once_then_falls_back(tmp_path,
                                                        monkeypatch):
    _config(tmp_path)
    replies = iter(["not json", "still not json"])
    monkeypatch.setattr(writer, "generate_with_wait",
                        lambda prompt, **k: next(replies))
    summary, state = writer._summarize_and_extract(1, "One", _words(300))
    assert state["events"] == [] and state["summary"] == summary
    assert not summary.endswith("alpha")         # cut at a word boundary


def test_writer_extraction_uses_its_own_agent_key(tmp_path, monkeypatch):
    _config(tmp_path)
    seen = {}

    def fake(prompt, **kwargs):
        seen.update(kwargs)
        return STATE_JSON

    monkeypatch.setattr(writer, "generate_with_wait", fake)
    writer._summarize_and_extract(1, "One", "draft")
    assert seen["agent"] == "extractor"


# --------------------------------------------------------------------- editor

BANNED = ['Never "foo".', 'Never "bar".']


def _editor_context(draft):
    update_context("chapters", [CHAPTERS[0]])
    update_context("bible", {"title": "T", "characters": [],
                             "constraints": BANNED})
    update_context("title", "T")
    update_context("drafts", {1: draft})
    update_context("chronology", {})


def _run_editor(monkeypatch, revise_reply, polish_reply=None):
    """Run the editor with scripted revise/polish replies."""
    def prose(prompt, **kwargs):
        if "You are revising" in prompt:
            return revise_reply
        return polish_reply if polish_reply is not None else \
            _body_of(prompt)

    monkeypatch.setattr(editor, "generate_prose", prose)
    editor.run_editor()
    return context["final"][1]


def _body_of(prompt):
    """Echo the chapter back from the polish prompt (a no-op polish)."""
    start = prompt.index("## Chapter")
    return prompt[start:prompt.index("\n\nReturn ONLY")]


def test_editor_fixes_a_lint_finding(tmp_path, monkeypatch):
    _config(tmp_path)
    _editor_context(_words(120) + " foo")
    final = _run_editor(monkeypatch, _words(120))
    assert "foo" not in final


def test_editor_rejects_a_revision_that_adds_findings(tmp_path, monkeypatch):
    _config(tmp_path)
    draft = _words(120) + " foo"                  # 1 finding
    _editor_context(draft)
    worse = _words(120) + " foo bar"              # 2 findings
    final = _run_editor(monkeypatch, worse)
    assert final.endswith(draft)                  # previous version kept
    assert "bar" not in final


def test_editor_rejects_a_revision_that_shrinks_the_chapter(tmp_path,
                                                            monkeypatch):
    _config(tmp_path)
    draft = _words(150) + " foo"
    _editor_context(draft)
    shrunk = _words(105)                          # fixes foo, loses a third
    final = _run_editor(monkeypatch, shrunk)
    assert final.endswith(draft)


def test_editor_rejects_a_polish_that_truncates(tmp_path, monkeypatch):
    _config(tmp_path)
    draft = _words(150)
    _editor_context(draft)
    final = _run_editor(monkeypatch, "unused",
                        polish_reply="## Chapter 1: Ch1\n\n" + _words(90))
    assert final.endswith(draft)


def test_editor_accepts_a_normal_polish(tmp_path, monkeypatch):
    _config(tmp_path)
    _editor_context(_words(150))
    polished = _words(148, prefix="beta")
    final = _run_editor(monkeypatch, "unused",
                        polish_reply="## Chapter 1: Ch1\n\n" + polished)
    assert final.endswith(polished)


# ------------------------------------------------------------- pipeline: main

def _args(**kw):
    base = dict(demo=False, seed=None, prompt=None)
    base.update(kw)
    return argparse.Namespace(**base)


def _state(seed="SEED", chapters=3, finals=0):
    return {"bible": {"title": "T", "seed": seed},
            "chapters": CHAPTERS[:chapters], "final": {n: "x" for n in range(1, finals + 1)}}


def test_resume_reuses_the_saved_seed_when_none_given():
    assert pipeline.check_resume(_args(), _state(seed="MY SEED"), None) == \
        "MY SEED"


def test_resume_accepts_the_same_seed_ignoring_whitespace():
    seed = pipeline.check_resume(_args(prompt="  SEED\r\n"), _state(), None)
    assert seed.strip() == "SEED"


def test_resume_refuses_a_different_seed():
    with pytest.raises(SystemExit) as exc:
        pipeline.check_resume(_args(prompt="A NEW BOOK"), _state(), None)
    assert "--no-resume" in str(exc.value)


def test_resume_refuses_a_different_chapter_count():
    with pytest.raises(SystemExit):
        pipeline.check_resume(_args(), _state(chapters=3), 5)
    pipeline.check_resume(_args(), _state(chapters=3), 3)   # same count is fine


def test_resume_refuses_unreadable_bible():
    with pytest.raises(SystemExit):
        pipeline.check_resume(_args(), {"bible": None, "chapters": None,
                                        "final": {}}, None)


def test_incomplete_chapters_reports_unfinished_work():
    ctx = {"chapters": CHAPTERS, "final": {1: "edited"},
           "drafts": {1: "a", 2: "b"}}
    assert pipeline._incomplete_chapters(ctx, editor_on=True) == [2, 3]
    assert pipeline._incomplete_chapters(ctx, editor_on=False) == [3]
