"""System prompts, the writer's previous-chapter ending and its draft guard."""

import json

import pytest

from agents import architect, editor, planner, researcher, reviewer, writer
from shared import prompts, web_search
from shared.context import context, reset_context, update_context
from shared.llm_client import load_config
from shared.story_state import minimal_state
from shared.prompts import unfence

BIBLE = {"title": "T", "genre": "Gothic", "tone": "Dry and wry",
         "premise": "p", "world": "A salt marsh", "constraints": [],
         "characters": [{"name": "Aria Vance", "role": "protagonist",
                         "description": "a surveyor"}]}
CHAPTERS = [{"number": n, "title": f"Ch{n}", "summary": f"s{n}", "part": ""}
            for n in (1, 2, 3)]
STATE_JSON = json.dumps({"summary": "It happened.", "time": "day",
                         "location": "town", "present": ["Aria Vance"],
                         "events": []})


def _config(tmp_path, extra=""):
    path = tmp_path / "config.yaml"
    path.write_text(f'output:\n  directory: "{tmp_path / "out"}"\n'
                    f"book:\n  words_per_chapter: 100\n{extra}"
                    "agents:\n  reviewer:\n    enabled: false\n")
    load_config(path)
    reset_context()
    update_context("bible", BIBLE)
    update_context("chapters", CHAPTERS)
    update_context("title", "T")


def _words(n, prefix="alpha"):
    return " ".join(f"{prefix}{i % 7}" for i in range(n))


class Recorder:
    """A stub LLM that records every call's system prompt and agent."""

    def __init__(self, reply):
        self.reply, self.calls = reply, []

    def __call__(self, prompt, **kwargs):
        self.calls.append({"prompt": prompt, **kwargs})
        return self.reply(prompt) if callable(self.reply) else self.reply


# ------------------------------------------------- every agent has a system prompt

def test_architect_has_a_json_system_prompt(tmp_path, monkeypatch):
    _config(tmp_path)
    rec = Recorder(json.dumps({"title": "X", "characters": []}))
    monkeypatch.setattr(architect, "generate_with_wait", rec)
    architect.run_architect("A seed.")
    assert rec.calls[0]["system"] == prompts.ARCHITECT
    assert "JSON" in prompts.ARCHITECT


def test_planner_has_a_system_prompt(tmp_path, monkeypatch):
    _config(tmp_path)
    rec = Recorder(json.dumps([{"title": "A", "summary": "x"}]))
    monkeypatch.setattr(planner, "generate_with_wait", rec)
    planner._plan_with_llm(BIBLE, 1)
    planner.suggest_chapter_count("seed", BIBLE)
    assert [c["system"] for c in rec.calls] == [prompts.PLANNER] * 2


def test_researcher_gets_the_bible_in_the_system_prompt(tmp_path, monkeypatch):
    _config(tmp_path)
    rec = Recorder("BRIEF")
    monkeypatch.setattr(researcher, "generate_with_wait", rec)
    researcher.run_researcher()
    systems = {c["system"] for c in rec.calls}
    assert len(systems) == 1                       # identical for every chapter
    (system,) = systems
    assert system.startswith(prompts.RESEARCHER)
    assert "A salt marsh" in system and "Aria Vance" in system
    assert "A salt marsh" not in rec.calls[0]["prompt"]


def test_fact_checker_prompt_forbids_leaking_the_story(tmp_path, monkeypatch):
    _config(tmp_path, "")
    (tmp_path / "config.yaml").write_text(
        f'output:\n  directory: "{tmp_path / "out"}"\nweb_search:\n  enabled: true\n'
        "agents:\n  reviewer:\n    enabled: false\n")
    load_config(tmp_path / "config.yaml")
    update_context("bible", BIBLE)
    rec = Recorder(lambda p: '["salt marsh tides"]'
                   if "fact-check" in p else "BRIEF")
    monkeypatch.setattr(researcher, "generate_with_wait", rec)
    monkeypatch.setattr(web_search, "is_available", lambda: True)
    monkeypatch.setattr(web_search, "search", lambda q, **k: [])
    update_context("chapters", CHAPTERS[:1])
    researcher.run_researcher()
    planning = next(c for c in rec.calls if "fact-check" in c["prompt"])
    assert planning["system"] == prompts.FACT_CHECKER
    assert "character name" in prompts.FACT_CHECKER


def test_extractor_has_a_system_prompt(tmp_path, monkeypatch):
    _config(tmp_path)
    rec = Recorder(STATE_JSON)
    monkeypatch.setattr(writer, "generate_with_wait", rec)
    writer._summarize_and_extract(1, "One", "draft text")
    assert rec.calls[0]["system"] == prompts.EXTRACTOR
    assert rec.calls[0]["agent"] == "extractor"


def test_reviewer_gets_the_bible_in_the_system_prompt(tmp_path, monkeypatch):
    _config(tmp_path)
    update_context("chronology", {})
    rec = Recorder(json.dumps({"verdict": "pass", "issues": []}))
    monkeypatch.setattr(reviewer, "generate_with_wait", rec)
    reviewer.run_reviewer(1, "One", "A draft.")
    system = rec.calls[0]["system"]
    assert system.startswith(prompts.REVIEWER) and "A salt marsh" in system
    assert "A salt marsh" not in rec.calls[0]["prompt"]


def test_editor_revise_and_polish_have_system_prompts(tmp_path, monkeypatch):
    _config(tmp_path, "  revision_rounds: 1\n")
    update_context("bible", {**BIBLE, "constraints": ['Never "foo".']})
    update_context("drafts", {1: _words(120) + " foo"})
    update_context("chapters", CHAPTERS[:1])
    update_context("chronology", {})

    def reply(prompt):
        if "You are revising" in prompt:
            return _words(120)
        return unfence(prompt, "chapter")

    rec = Recorder(reply)
    monkeypatch.setattr(editor, "generate_prose", rec)
    editor.run_editor()
    revise = next(c for c in rec.calls if "You are revising" in c["prompt"])
    polish = next(c for c in rec.calls if "You are revising" not in c["prompt"])
    assert revise["system"].startswith(prompts.REVISER)
    assert polish["system"].startswith(prompts.POLISHER)
    assert "Aria Vance" in revise["system"]
    assert "STORY BIBLE REFERENCE" not in polish["prompt"]


def test_revise_prompt_allows_the_length_exception():
    import inspect
    assert "LENGTH finding" in inspect.getsource(editor._revise)


# ------------------------------------------------------- writer system prompt

def _writer_run(tmp_path, monkeypatch, reply=None):
    _config(tmp_path)
    rec = Recorder(reply or (lambda p: _words(120)))
    monkeypatch.setattr(writer, "generate_prose", rec)
    monkeypatch.setattr(writer, "generate_with_wait",
                        lambda prompt, **k: STATE_JSON)
    return rec


def test_writer_system_prompt_carries_the_bible_and_is_stable(tmp_path,
                                                              monkeypatch):
    rec = _writer_run(tmp_path, monkeypatch)
    writer.run_writer()
    systems = {c["system"] for c in rec.calls}
    assert len(systems) == 1                       # same prefix every chapter
    (system,) = systems
    assert system.startswith(prompts.WRITER)
    assert "Dry and wry" in system and "A salt marsh" in system
    assert "A salt marsh" not in rec.calls[0]["prompt"]


def test_writer_prompt_includes_previous_chapter_ending(tmp_path, monkeypatch):
    rec = _writer_run(tmp_path, monkeypatch)
    update_context("drafts", {1: "Opening. " + _words(300, "tail")})
    update_context("chronology", {1: minimal_state(1, "Ch1", "s")})
    writer.run_writer()
    second = rec.calls[0]["prompt"]                 # chapter 2
    assert "PREVIOUS CHAPTER ENDING" in second
    ending = second.split("PREVIOUS CHAPTER ENDING")[1].split("LORE BRIEF")[0]
    assert "tail" in ending and "Opening." not in ending
    assert len(ending.split()) < 220


def test_first_chapter_has_no_previous_ending(tmp_path, monkeypatch):
    rec = _writer_run(tmp_path, monkeypatch)
    writer.run_writer()
    assert "PREVIOUS CHAPTER ENDING" not in rec.calls[0]["prompt"]
    assert "PREVIOUS CHAPTER ENDING" in rec.calls[1]["prompt"]


def test_tail_words_starts_at_a_sentence_break():
    text = " ".join(["word"] * 400) + ". Fresh start here. " + " ".join(
        ["end"] * 150)
    tail = writer._tail_words(text, 170)
    assert tail.startswith("Fresh start here.") or tail.startswith("end")
    assert writer._tail_words("short text", 180) == "short text"


# ----------------------------------------------------- refusal / short drafts

@pytest.mark.parametrize("draft,expected", [
    ("", "nothing"),
    ("I can't help with that request.", "refusal"),
    ("I'm sorry, but I cannot write this.", "refusal"),
    ("As an AI, I won't produce that.", "refusal"),
    ("Sorry, no.", "refusal"),
    (_words(20), "only 20 words"),
    (_words(60), None),                         # short but a real attempt
    ("I can't believe it, she said. " + _words(120), None),  # first person
    (_words(120), None),
])
def test_bad_draft_detection(draft, expected):
    problem = writer._bad_draft(draft, min_words=80)
    if expected is None:
        assert problem is None
    else:
        assert expected in problem


def test_writer_retries_once_after_a_refusal(tmp_path, monkeypatch):
    replies = iter(["I can't write that.", _words(120)])
    rec = _writer_run(tmp_path, monkeypatch, lambda p: next(replies))
    update_context("chapters", CHAPTERS[:1])
    writer.run_writer()
    assert len(rec.calls) == 2
    assert "not a usable chapter" in rec.calls[1]["prompt"]
    assert context["drafts"][1].startswith("alpha0")


def test_writer_stops_when_the_model_keeps_refusing(tmp_path, monkeypatch):
    rec = _writer_run(tmp_path, monkeypatch, lambda p: "I can't write that.")
    writer.run_writer()
    assert len(rec.calls) == 2                     # two attempts, then stop
    assert context["drafts"] == {}                 # the refusal is never saved
    assert not (tmp_path / "out" / "chapters").exists()


# ------------------------------------------------- fences, boundaries, rules

def test_fence_round_trips_and_survives_triple_quotes_and_its_own_tag():
    text = 'She read: """old letter""" and wrote </chapter> twice.'
    wrapped = prompts.fence(text, "chapter")
    assert wrapped.startswith("<chapter>\n") and wrapped.endswith("\n</chapter>")
    assert wrapped.count("</chapter>") == 1          # the inner one escaped
    assert unfence(f"Edit this:\n{wrapped}\n\nReturn it.", "chapter") == text
    assert unfence("no fence here", "chapter") is None


def test_writer_is_told_where_the_chapter_stops(tmp_path, monkeypatch):
    """Regression: the writer saw only its own chapter's summary, so it ran on
    into the next chapter's events or ended short of its own."""
    rec = _writer_run(tmp_path, monkeypatch)
    writer.run_writer()
    first, last = rec.calls[0]["prompt"], rec.calls[-1]["prompt"]
    assert "NEXT CHAPTER" in first and "Chapter 2: Ch2 - s2" in first
    assert "do NOT write any of it" in first
    assert "NEXT CHAPTER" not in last and "LAST chapter" in last
    assert "Hit every plot beat in the lore brief" in first


def test_writer_is_told_the_style_rules_it_is_linted_against(tmp_path,
                                                             monkeypatch):
    rec = Recorder(lambda p: _words(120))
    _config(tmp_path, "  style:\n    banned_phrases: [\"coil of\"]\n"
                      "    tense: past\n")
    monkeypatch.setattr(writer, "generate_prose", rec)
    monkeypatch.setattr(writer, "generate_with_wait", lambda p, **k: STATE_JSON)
    writer.run_writer()
    system = rec.calls[0]["system"]
    assert "STYLE RULES" in system and '"coil of"' in system
    assert "past tense" in system
    assert len({c["system"] for c in rec.calls}) == 1    # still cacheable


def test_no_style_section_without_style_rules(tmp_path, monkeypatch):
    rec = _writer_run(tmp_path, monkeypatch)
    writer.run_writer()
    assert "STYLE RULES" not in rec.calls[0]["system"]


def test_researcher_sees_the_whole_outline(tmp_path, monkeypatch):
    """Regression: the brief must say what to set up for later chapters, but
    the researcher was shown only the chapter it briefed."""
    _config(tmp_path)
    rec = Recorder("brief")
    monkeypatch.setattr(researcher, "generate_with_wait", rec)
    monkeypatch.setattr(researcher.web_search, "enabled", lambda: False)
    researcher.run_researcher()
    systems = {c["system"] for c in rec.calls}
    assert len(systems) == 1                       # same prefix every chapter
    (system,) = systems
    assert "BOOK OUTLINE" in system
    assert all(f"{n}. Ch{n}: s{n}" in system for n in (1, 2, 3))


def _edit_run(tmp_path, monkeypatch, extra="", constraints=(), draft=None):
    _config(tmp_path, "  revision_rounds: 1\n" + extra)
    update_context("bible", {**BIBLE, "constraints": list(constraints)})
    update_context("drafts", {1: draft or _words(120)})
    update_context("chapters", CHAPTERS[:1])
    update_context("chronology", {})

    def reply(prompt):
        if "You are revising" in prompt:
            return _words(120)
        return unfence(prompt, "chapter")

    rec = Recorder(reply)
    monkeypatch.setattr(editor, "generate_prose", rec)
    editor.run_editor()
    return rec


def test_polish_gets_the_body_and_the_rules_the_guard_enforces(tmp_path,
                                                                monkeypatch):
    """Regression: the polish was linted against banned words and style
    phrases it was never shown, and was thrown away whole when it reused one."""
    rec = _edit_run(tmp_path, monkeypatch,
                    "  style:\n    banned_phrases: [\"coil of\"]\n",
                    constraints=['Never the words "unhurried" or "exactly".'])
    polish = next(c for c in rec.calls if c["prompt"].startswith("Copy-edit"))
    prompt = polish["prompt"]
    assert "## Chapter" not in prompt              # no heading to keep or lose
    assert unfence(prompt, "chapter") == _words(120)
    assert '"unhurried"' in prompt and '"coil of"' in prompt
    assert "not a rewrite" in prompt


def test_revise_prompt_repeats_its_task_after_the_chapter(tmp_path,
                                                          monkeypatch):
    rec = _edit_run(tmp_path, monkeypatch,
                    constraints=['Never "foo".'], draft=_words(120) + " foo")
    revise = next(c["prompt"] for c in rec.calls
                  if "You are revising" in c["prompt"])
    after = revise.split("</chapter>", 1)[1]
    assert "fix only these findings (banned_word)" in after
    assert "mirror the input length" not in revise    # advice for us, not it


def test_revise_reminder_states_the_length_floor(monkeypatch):
    rec = Recorder(lambda p: _words(600))
    monkeypatch.setattr(editor, "generate_prose", rec)
    lint = [{"check": "word_count", "minimum": 500, "detail": "short"}]
    editor._revise(1, "T", _words(200), lint, [])
    after = rec.calls[0]["prompt"].split("</chapter>", 1)[1]
    assert "at least 500 words" in after


def test_drafted_chapters_are_timed(tmp_path, monkeypatch):
    from shared.resume import load_state
    _writer_run(tmp_path, monkeypatch)
    writer.run_writer()
    notes = load_state()["chapter_notes"]
    assert sorted(notes) == [1, 2, 3]
    assert notes[1]["drafted_at"] <= notes[3]["drafted_at"]
