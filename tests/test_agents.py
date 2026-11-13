"""Agent and pipeline-control tests with a stubbed LLM (temp dirs only)."""

import argparse
import json

import pytest

import main as pipeline
from agents import editor, writer
from shared.context import context, reset_context, update_context
from shared.llm_client import load_config
from shared.story_state import minimal_state
from shared.prompts import unfence

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
    assert (out / "state" / "chronology.json").exists()


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
    return unfence(prompt, "chapter")


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
    _config(tmp_path, "  word_count_tolerance: 1.2\n")      # min 120 words
    draft = _words(150) + " foo"
    _editor_context(draft)
    shrunk = _words(105)          # fixes foo, loses a third, under the minimum
    final = _run_editor(monkeypatch, shrunk)
    assert final.endswith(draft)


def test_editor_lets_a_long_chapter_be_tightened_to_its_length(tmp_path,
                                                              monkeypatch):
    # 3,189 -> 2,200 words keeps 69% of a chapter: under MIN_REVISION_RATIO,
    # but still over the minimum, so it is tightening, not truncation
    _config(tmp_path)                             # target 100, min 80, max 150
    _editor_context(_words(200))
    tightened = _words(110, prefix="beta")
    final = _run_editor(monkeypatch, tightened)
    assert final.endswith(tightened)


def test_editor_tightens_a_long_chapter_once(tmp_path, monkeypatch):
    _config(tmp_path)
    _editor_context(_words(400))
    calls = []

    def revise(prompt, **kwargs):
        if "You are revising" not in prompt:
            return _body_of(prompt)                # a no-op polish
        calls.append(prompt)
        return _words(380, prefix="gamma")       # tighter, still too long
    monkeypatch.setattr(editor, "generate_prose", revise)
    editor.run_editor()
    assert len(calls) == 1
    assert "TIGHTEN" in calls[0] and "aim for about 100" in calls[0]


def test_editor_leaves_long_chapters_alone_when_max_length_ratio_is_0(
        tmp_path, monkeypatch):
    _config(tmp_path, "  max_length_ratio: 0\n")
    draft = _words(400)
    _editor_context(draft)
    final = _run_editor(monkeypatch, "never asked")
    assert final.endswith(draft)


def test_editor_rejects_a_tightening_that_drops_a_character(tmp_path,
                                                             monkeypatch):
    _config(tmp_path)
    draft = _words(150) + " Mara left. " + _words(100)
    _editor_context(draft)
    context["bible"]["characters"] = [{"name": "Mara Vance"}]
    final = _run_editor(monkeypatch, _words(110, prefix="beta"))
    assert "Mara" in final


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


def test_editor_rejects_a_polish_that_introduces_a_name(tmp_path, monkeypatch):
    """The polish is not reviewed; a renamed character must be caught in code."""
    _config(tmp_path)
    draft = _words(150)
    _editor_context(draft)
    polished = _words(148, prefix="beta") + " and then met Darius"
    final = _run_editor(monkeypatch, "unused",
                        polish_reply="## Chapter 1: Ch1\n\n" + polished)
    assert final.endswith(draft)


def test_editor_stops_after_a_revision_that_fixes_nothing(tmp_path,
                                                          monkeypatch):
    """Regression: a revision with the same severity was accepted and the loop
    spent every remaining round on the same findings."""
    _config(tmp_path, "  revision_rounds: 4\n")
    _editor_context(_words(120) + " foo")
    calls = []

    def prose(prompt, **kwargs):
        if "You are revising" in prompt:
            calls.append(1)
            return _words(120, prefix="gamma") + " foo"    # still has foo
        return _body_of(prompt)

    monkeypatch.setattr(editor, "generate_prose", prose)
    editor.run_editor()
    assert len(calls) == 1


def test_save_book_titles_an_untitled_book(tmp_path):
    _config(tmp_path)
    update_context("chapters", [CHAPTERS[0]])
    update_context("title", "")
    book = editor.save_book({1: "Body."})
    assert book.startswith("# Untitled\n")


def test_short_note_names_what_the_editor_gave_up_on():
    from agents.editor import short_note
    notes = ["lint on the draft: 2 finding(s)\n  - quota",
             "reviewer: unreviewed - the reply could not be read",
             "round 1: revision accepted (findings 2 -> 1)",
             "round 2: revision rejected (severity 3 -> 5)",
             "polish rejected (introduced new names: Zed)"]
    note = short_note(notes, [{"check": "quota"}], [{"type": "timeline"}])
    assert note == ("revision rejected (severity 3 -> 5); polish rejected "
                    "(introduced new names: Zed); 2 finding(s) left: quota, "
                    "timeline")
    assert short_note(["round 1: revision accepted", "polish accepted"],
                      [], []) == ""
    assert short_note([], [], [], unreviewed=True) == "no readable review"


def test_finished_chapters_are_timed_and_noted(tmp_path, monkeypatch):
    """The web GUI's estimate and per-chapter notes come from state/."""
    from shared.resume import load_state
    _config(tmp_path)
    _editor_context(_words(150))
    _run_editor(monkeypatch, "unused",
                polish_reply=_words(148, prefix="beta") + " and then met Darius")
    notes = load_state()["chapter_notes"]
    assert notes[1]["edited_at"] > 0
    assert "introduced new names: Darius" in notes[1]["note"]


def test_a_fused_word_in_the_polish_is_repaired_not_kept(tmp_path, monkeypatch):
    """Regression: DRY made the polish fuse words it was copying."""
    _config(tmp_path)
    draft = _words(150) + " she leaned against her chair"
    _editor_context(draft)
    final = _run_editor(monkeypatch, "unused",
                        polish_reply=_words(150) + " she leaned againsther chair")
    assert final.endswith("she leaned against her chair")
    assert "againsther" not in final


def test_a_polish_with_garbled_words_is_rejected(tmp_path, monkeypatch):
    _config(tmp_path)
    draft = _words(150) + " an unremarkable week"
    _editor_context(draft)
    final = _run_editor(monkeypatch, "unused",
                        polish_reply=_words(150) + " an unremarkableRE week")
    assert final.endswith(draft)


# ------------------------------------------- length and point of view (editor)

THIRD = "He walked to the door and she followed him into the rain. " * 20
FIRST = "I walked to the door and my hands shook in the rain again. " * 20


def test_a_shorter_revision_that_fixes_a_finding_is_kept(tmp_path,
                                                         monkeypatch):
    """Quality over quantity: a revision that fixes a real finding is not
    rejected for dropping under the length minimum (it was: severity 1 -> 3)."""
    _config(tmp_path, "  word_count_tolerance: 1.2\n")      # min 120 words
    _editor_context(_words(130) + " foo")
    final = _run_editor(monkeypatch, _words(105))   # under the min; ratio 0.8
    assert "foo" not in final
    assert final.endswith(_words(105))


def test_a_shorter_revision_that_fixes_nothing_is_rejected(tmp_path,
                                                           monkeypatch):
    _config(tmp_path)
    draft = _words(120) + " foo"
    _editor_context(draft)
    final = _run_editor(monkeypatch, _words(100, prefix="gamma") + " foo")
    assert final.endswith(draft)


def test_editor_flags_a_chapter_that_leaves_the_books_person(tmp_path,
                                                             monkeypatch):
    """Regression: chapters 1-8 in third person, 9 and 10 slipped into "I";
    with no book.style.pov nothing noticed. Third person is the default."""
    _config(tmp_path)
    update_context("chapters", CHAPTERS)
    update_context("bible", {"title": "T", "characters": []})
    update_context("title", "T")
    update_context("drafts", {1: THIRD, 2: THIRD, 3: THIRD + "\n\n" + FIRST})
    update_context("final", {1: THIRD, 2: THIRD})
    prompts_seen = []

    def prose(prompt, **kwargs):
        if "You are revising" in prompt:
            prompts_seen.append(prompt)
            return THIRD + "\n\n" + THIRD
        return _body_of(prompt)

    monkeypatch.setattr(editor, "generate_prose", prose)
    editor.run_editor()
    assert prompts_seen and "pov" in prompts_seen[0]
    assert "the default" in prompts_seen[0]
    assert "I walked" not in context["final"][3]


def test_the_writer_is_told_to_narrate_in_the_third_person(tmp_path,
                                                          stub_writer):
    _config(tmp_path)
    _writer_context()
    writer.run_writer()
    assert all("Narrate in the third person" in p for p in stub_writer)


SCENE_CHARACTERS = [{"name": "Stuart"}, {"name": "Kirsty"},
                    {"name": "James"}, {"name": "Chloe"}]


def test_the_writer_is_told_the_chapters_cast_and_county(tmp_path, stub_writer):
    _config(tmp_path)
    update_context("chapters", [{"number": 1, "title": "Ch1", "part": "",
                                 "summary": "Couples: Stuart & Kirsty. Plot: x"}])
    update_context("bible", {"title": "T", "characters": SCENE_CHARACTERS,
                             "setting": "A cul-de-sac in rural Hampshire."})
    update_context("title", "T")
    writer.run_writer()
    assert "In this chapter's scenes: Stuart, Kirsty." in stub_writer[0]
    assert "set in Hampshire" in stub_writer[0]


def test_editor_sends_back_a_chapter_with_someone_the_outline_left_out(
        tmp_path, monkeypatch):
    """Regression: James and Chloe at a dinner the outline gave to Angela &
    Pete; the reviewer passed it."""
    _config(tmp_path)
    intruded = (_words(100) + " James nodded. James laughed. James stood. "
                "Stuart sat. Kirsty smiled.")
    update_context("chapters", [{"number": 1, "title": "Ch1", "part": "",
                                 "summary": "Couples: Stuart & Kirsty. Plot: x"}])
    update_context("bible", {"title": "T", "characters": SCENE_CHARACTERS})
    update_context("title", "T")
    update_context("drafts", {1: intruded})
    update_context("chronology", {})
    seen = []

    def prose(prompt, **kwargs):
        if "You are revising" in prompt:
            seen.append(prompt)
            return _words(100) + " Stuart sat. Kirsty smiled. They spoke of James."
        return _body_of(prompt)

    monkeypatch.setattr(editor, "generate_prose", prose)
    editor.run_editor()
    assert seen and "[lint] James is in this scene" in seen[0]
    assert "James nodded" not in context["final"][1]


def test_shrink_floor_allows_tightening_only_down_to_the_length():
    assert editor.shrink_floor(3189, 0.75, 1280) == 1280   # 2,200 is allowed
    assert editor.shrink_floor(1806, 0.85, 1600) == 0.85 * 1806   # 438 is not
    assert editor.shrink_floor(1000, 0.75, 0) == 750
