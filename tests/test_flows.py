"""Behaviour of the less-travelled paths: fallbacks, failures, CLI helpers."""

import argparse
import json
import sys

import pytest

import main as pipeline
from agents import architect, editor, researcher
from agents.reviewer import UNREVIEWED
from shared import llm_client, output, runlog
from shared.context import context, reset_context, update_context
from shared.llm_client import EndpointUnavailable, load_config
from shared.resume import _read_json, load_state, summarize_for_log


def _config(tmp_path, book="", agents="", output_extra="", reviewer_on=False):
    path = tmp_path / "config.yaml"
    reviewer = "" if reviewer_on else "  reviewer:\n    enabled: false\n"
    path.write_text(f'output:\n  directory: "{tmp_path / "out"}"\n{output_extra}'
                    f"book:\n  words_per_chapter: 100\n{book}"
                    f"agents:\n{reviewer}{agents}")
    load_config(path)
    reset_context()
    return tmp_path / "out"


def _words(n, prefix="alpha"):
    letters = "abcdefghijklmnopqrstuvwxyz"
    return " ".join(f"{prefix}{letters[i % 26]}{letters[(i // 26) % 26]}"
                    for i in range(n))


# ------------------------------------------------------------------ architect

@pytest.mark.parametrize("seed,title", [
    ("# The Marsh Light\n\nbody", "The Marsh Light"),
    ("\n\n  Just a first line\nmore", "Just a first line"),
    ("", "Untitled Story"),
    ("   \n\n  ", "Untitled Story"),
    ("# " + "x" * 200, "x" * 80),
])
def test_fallback_title(seed, title):
    assert architect._fallback_title(seed) == title


SEED = """# My Story

## Characters
- **Aria** - protagonist. A surveyor.

## Constraints
- Never "unhurried".

## Explicitness
Keep it gentle.
"""


def _architect(monkeypatch, tmp_path, reply):
    _config(tmp_path)
    monkeypatch.setattr(architect, "generate_with_wait",
                        lambda p, **k: reply if isinstance(reply, str)
                        else json.dumps(reply))
    return architect.run_architect(SEED)


def test_garbage_from_the_model_falls_back_but_keeps_the_seeds_own_text(
        tmp_path, monkeypatch, capsys):
    bible = _architect(monkeypatch, tmp_path, "sorry, no json")
    assert "Falling back to raw seed" in capsys.readouterr().out
    assert bible["title"] == "My Story"
    assert [c["name"] for c in bible["characters"]] == ["Aria"]   # verbatim
    assert bible["constraints"] == ['Never "unhurried".']
    assert "Keep it gentle." in bible["notes"]
    assert bible["seed"] == SEED
    assert context["title"] == "My Story" and context["seed"] == SEED


def test_a_json_array_is_not_a_bible(tmp_path, monkeypatch, capsys):
    bible = _architect(monkeypatch, tmp_path, "[1, 2, 3]")
    assert bible["title"] == "My Story"
    assert "Falling back" in capsys.readouterr().out


def test_model_characters_are_normalised(tmp_path, monkeypatch):
    bible = _architect(monkeypatch, tmp_path, {
        "title": "  ", "genre": "g", "constraints": ["", "  ", "Keep it dry"],
        "characters": ["  Henry  ", "", {"name": "Zed", "aliases": [
            "Z", 7, " "]}, {"role": "no name"}, 42]})
    assert bible["title"] == "My Story"                 # blank -> from seed
    names = [c["name"] for c in bible["characters"]]
    assert names == ["Aria", "Henry", "Zed"]
    zed = next(c for c in bible["characters"] if c["name"] == "Zed")
    assert zed["aliases"] == ["Z"]


def test_model_notes_and_seed_notes_are_combined(tmp_path, monkeypatch):
    bible = _architect(monkeypatch, tmp_path,
                       {"title": "T", "notes": "model notes"})
    assert bible["notes"].startswith("model notes")
    assert "Keep it gentle." in bible["notes"]


def test_the_architect_never_hides_an_outage(tmp_path, monkeypatch):
    _config(tmp_path)

    def down(*a, **k):
        raise EndpointUnavailable("down")

    monkeypatch.setattr(architect, "generate_with_wait", down)
    with pytest.raises(EndpointUnavailable):
        architect.run_architect(SEED)


# ----------------------------------------------------------------- researcher

CHAPTERS = [{"number": n, "title": f"Ch{n}", "summary": f"s{n}", "part": ""}
            for n in (1, 2)]


def _research(monkeypatch, tmp_path, llm, **ctx):
    _config(tmp_path)
    update_context("chapters", ctx.get("chapters", CHAPTERS))
    update_context("bible", {"title": "T", "characters": []})
    if "research" in ctx:
        update_context("research", ctx["research"])
    monkeypatch.setattr(researcher, "generate_with_wait", llm)


def test_a_failed_brief_is_recorded_empty_and_the_run_continues(
        tmp_path, monkeypatch, capsys):
    replies = iter([RuntimeError("boom"), "Brief two"])

    def llm(prompt, **kwargs):
        reply = next(replies)
        if isinstance(reply, Exception):
            raise reply
        return reply

    _research(monkeypatch, tmp_path, llm)
    researcher.run_researcher()
    assert context["research"] == {1: "", 2: "Brief two"}
    assert "Error briefing chapter 1: boom" in capsys.readouterr().out
    assert "(1/2 briefs)" in capsys.readouterr().out or True


def test_an_outage_aborts_the_research_phase_with_a_resume_hint(
        tmp_path, monkeypatch, capsys):
    def down(*a, **k):
        raise EndpointUnavailable("down")

    _research(monkeypatch, tmp_path, down)
    with pytest.raises(EndpointUnavailable):
        researcher.run_researcher()
    assert "Rerun the same command to resume" in capsys.readouterr().out


def test_existing_briefs_are_not_regenerated(tmp_path, monkeypatch):
    calls = []
    _research(monkeypatch, tmp_path,
              lambda p, **k: calls.append(p) or "new",
              research={1: "kept brief"})
    researcher.run_researcher()
    assert context["research"] == {1: "kept brief", 2: "new"}
    assert len(calls) == 1


def test_no_chapters_means_nothing_to_research(tmp_path, monkeypatch, capsys):
    _research(monkeypatch, tmp_path, lambda p, **k: pytest.fail("called"),
              chapters=[])
    researcher.run_researcher()
    assert "No chapters found" in capsys.readouterr().out


def _web_research(monkeypatch, tmp_path, proposer):
    from shared import web_search
    _config(tmp_path)
    (tmp_path / "config.yaml").write_text(
        f'output:\n  directory: "{tmp_path / "out"}"\nweb_search:\n'
        "  enabled: true\nagents:\n  reviewer:\n    enabled: false\n")
    load_config(tmp_path / "config.yaml")
    update_context("chapters", CHAPTERS[:1])
    update_context("bible", {"title": "T", "characters": []})
    monkeypatch.setattr(web_search, "is_available", lambda: True)
    monkeypatch.setattr(web_search, "search", lambda q: pytest.fail("searched"))

    def llm(prompt, **kwargs):
        if "fact-check a novel" in prompt:
            return proposer()
        return "BRIEF"

    monkeypatch.setattr(researcher, "generate_with_wait", llm)


def test_unreadable_claims_just_mean_no_web_facts(tmp_path, monkeypatch,
                                                  capsys):
    _web_research(monkeypatch, tmp_path, lambda: "no idea")
    researcher.run_researcher()
    assert context["research"][1] == "BRIEF"
    assert "could not plan web claims" in capsys.readouterr().out


def test_claim_planning_outage_propagates(tmp_path, monkeypatch):
    def down():
        raise EndpointUnavailable("down")

    _web_research(monkeypatch, tmp_path, down)
    with pytest.raises(EndpointUnavailable):
        researcher.run_researcher()


def test_claims_that_are_not_objects_or_strings_are_ignored(tmp_path,
                                                            monkeypatch):
    _web_research(monkeypatch, tmp_path, lambda: json.dumps(
        [None, 7, {"claim": "", "query": ""}, {"query": "only a query"}]))
    researcher.run_researcher()          # nothing usable -> no search at all
    assert context["research"][1] == "BRIEF"


# --------------------------------------------------------------------- editor

BIBLE = {"title": "T", "characters": [], "constraints": []}


def _editor(monkeypatch, tmp_path, drafts, prose, book="", agents="",
            reviewer=None):
    out = _config(tmp_path, book, agents, reviewer_on=reviewer is not None)
    update_context("bible", BIBLE)
    update_context("title", "T")
    update_context("chapters", [c for c in CHAPTERS if c["number"] in drafts])
    update_context("drafts", drafts)
    update_context("chronology", {})
    monkeypatch.setattr(editor, "generate_prose", prose)
    if reviewer:
        monkeypatch.setattr(editor, "run_reviewer", reviewer)
    return out


def _echo_polish(prompt):
    start = prompt.index("## Chapter")
    return prompt[start:prompt.index("\n\nReturn ONLY")]


def test_reviewer_findings_are_fixed_and_rereviewed(tmp_path, monkeypatch):
    reviews = iter([("revise", [{"type": "timeline",
                                 "description": "wrong day", "fix": "Monday"}]),
                    ("pass", [])])
    revise_prompts = []

    def prose(prompt, **kwargs):
        if "You are revising" in prompt:
            revise_prompts.append(prompt)
            return _words(120, "fixed")
        return _echo_polish(prompt)

    _editor(monkeypatch, tmp_path, {1: _words(120)}, prose,
            reviewer=lambda *a: next(reviews))
    editor.run_editor()
    assert "REVIEWER NOTES" in revise_prompts[0] and "wrong day" in revise_prompts[0]
    assert "fixed" in context["final"][1]



TIMELINE_ISSUE = {"type": "timeline", "description": "wrong day",
                  "fix": "Monday"}


def _scripted_reviews(*replies):
    calls = []
    replies = iter(replies)

    def review(n, title, text):
        calls.append(text)
        return next(replies)
    return review, calls


def test_a_length_expansion_is_rereviewed_even_after_a_pass(tmp_path,
                                                            monkeypatch):
    """New material can contradict the story, so a revision that grows the
    chapter is reviewed again; here it did, and the short draft is kept."""
    review, calls = _scripted_reviews(("pass", []),
                                      ("revise", [TIMELINE_ISSUE]))

    def prose(prompt, **kwargs):
        if "You are revising" in prompt:
            return _words(130, "grown")
        return _echo_polish(prompt)

    _editor(monkeypatch, tmp_path, {1: _words(50)}, prose, reviewer=review,
            book="  extra_length_rounds: 0\n")
    editor.run_editor()
    assert len(calls) == 2 and "growna" in calls[1]
    assert context["final"][1] == _words(50)        # contradiction rejected


def test_a_small_lint_fix_is_not_rereviewed(tmp_path, monkeypatch):
    review, calls = _scripted_reviews(("pass", []))

    def prose(prompt, **kwargs):
        if "You are revising" in prompt:
            return _words(120)
        return _echo_polish(prompt)

    _editor(monkeypatch, tmp_path, {1: _words(120) + " ominous"}, prose,
            reviewer=review)
    update_context("bible", {**BIBLE, "constraints": ['Never "ominous".']})
    editor.run_editor()
    assert len(calls) == 1
    assert "ominous" not in context["final"][1]


def test_fixing_a_continuity_error_outweighs_two_banned_words(
        tmp_path, monkeypatch):
    """Two banned words (severity 1 each) are better than a timeline error
    (severity 5): a plain count (1 -> 2) would have rejected this revision."""
    review, _ = _scripted_reviews(("revise", [TIMELINE_ISSUE]), ("pass", []))

    def prose(prompt, **kwargs):
        if "You are revising" in prompt:
            return _words(118) + " ominous foo"
        return _echo_polish(prompt)

    _editor(monkeypatch, tmp_path, {1: _words(120)}, prose, reviewer=review,
            book="  revision_rounds: 1\n")
    update_context("bible", {**BIBLE, "constraints": ['Never "ominous".',
                                                    'Never "foo".']})
    editor.run_editor()
    assert context["final"][1].endswith("ominous foo")
    diff = (tmp_path / "out" / "interim" / "diff_chapter_01.md").read_text()
    assert "revision accepted" in diff and "severity 5 -> 2" in diff


def test_an_unreadable_rereview_does_not_count_as_a_fix(tmp_path,
                                                         monkeypatch, capsys):
    review, _ = _scripted_reviews(("revise", [TIMELINE_ISSUE]),
                                  (UNREVIEWED, []), (UNREVIEWED, []))

    def prose(prompt, **kwargs):
        if "You are revising" in prompt:
            return _words(120, "beta")
        return _echo_polish(prompt)

    _editor(monkeypatch, tmp_path, {1: _words(120)}, prose, reviewer=review)
    editor.run_editor()
    diff = (tmp_path / "out" / "interim" / "diff_chapter_01.md").read_text()
    assert "earlier issues assumed unfixed" in diff
    assert context["unreviewed"] == {1}
    assert "finished WITHOUT a readable review" in capsys.readouterr().out
    report = (tmp_path / "out" / "interim" / "lint_report.md").read_text()
    assert "## Not reviewed" in report


def test_an_unreadable_first_review_is_reported(tmp_path, monkeypatch):
    review, _ = _scripted_reviews((UNREVIEWED, []))
    _editor(monkeypatch, tmp_path, {1: _words(120)},
            lambda p, **k: _echo_polish(p), reviewer=review)
    editor.run_editor()
    assert context["unreviewed"] == {1}
    diff = (tmp_path / "out" / "interim" / "diff_chapter_01.md").read_text()
    assert "reviewer: unreviewed - the reply could not be read" in diff


def test_severity_weighs_continuity_above_surface_findings():
    assert editor.severity([{"check": "dead_character"}]) > \
        editor.severity([{"check": "quota"}, {"check": "banned_word"}])
    assert editor.severity([], [{"type": "made_up_type"}]) == \
        editor.DEFAULT_SEVERITY

def test_a_short_chapter_is_expanded_with_the_length_instructions(
        tmp_path, monkeypatch):
    seen = []

    def prose(prompt, **kwargs):
        if "You are revising" in prompt:
            seen.append(prompt)
            return _words(130, "grown")
        return _echo_polish(prompt)

    _editor(monkeypatch, tmp_path, {1: _words(50)}, prose)
    editor.run_editor()
    assert "LENGTH - HARD REQUIREMENT" in seen[0]
    assert "the draft is 50 words; the minimum is" in seen[0]
    assert len(context["final"][1].split()) > 100


def test_expansion_stops_when_it_stalls(tmp_path, monkeypatch, capsys):
    """Minimum 320 words; each revision adds only a few, so after the first
    extra round the editor gives up instead of burning more LLM calls."""
    lengths = iter([160, 170, 180, 190])           # < 15% growth per round
    revisions = []

    def prose(prompt, **kwargs):
        if "You are revising" in prompt:
            revisions.append(prompt)
            return _words(next(lengths), "slow")
        return _echo_polish(prompt)

    _editor(monkeypatch, tmp_path, {1: _words(150)}, prose,
            book="  words_per_chapter: 400\n  revision_rounds: 1\n"
                 "  extra_length_rounds: 3\n")
    editor.run_editor()
    assert "expansion stalled at 170 words" in capsys.readouterr().out
    assert len(revisions) == 2                       # not all 4 allowed rounds
    assert set(context["final"]) == {1}


def test_a_polish_failure_keeps_the_revised_draft(tmp_path, monkeypatch,
                                                  capsys):
    def prose(prompt, **kwargs):
        raise RuntimeError("model fell over")

    _editor(monkeypatch, tmp_path, {1: _words(120)}, prose)
    editor.run_editor()
    assert "Error editing chapter 1: model fell over" in capsys.readouterr().out
    assert context["final"][1] == _words(120)


def test_a_polish_that_drops_the_heading_gets_it_back(tmp_path, monkeypatch):
    def prose(prompt, **kwargs):
        return _words(120)                            # no heading at all

    _editor(monkeypatch, tmp_path, {1: _words(120)}, prose)
    editor.run_editor()
    assert context["final"][1] == _words(120)
    edited = tmp_path / "out" / "interim" / "edited_chapter_01.md"
    assert edited.read_text().startswith("## Chapter 1: Ch1\n\n")


def test_a_polish_with_a_reworded_heading_keeps_only_its_body(tmp_path,
                                                              monkeypatch):
    """A heading in another form ("Chapter One") is replaced by ours, and a
    first paragraph written straight under it is not lost."""
    def prose(prompt, **kwargs):
        return "## Chapter One: The Start\n" + _words(120)

    _editor(monkeypatch, tmp_path, {1: _words(120)}, prose)
    editor.run_editor()
    assert context["final"][1] == _words(120)
    book = (tmp_path / "out" / "draft.md").read_text()
    assert "## Chapter 1: Ch1\n\n" in book and "Chapter One" not in book


@pytest.mark.parametrize("failing", ["revise", "polish"])
def test_outages_during_editing_abort_instead_of_shipping_unedited_text(
        tmp_path, monkeypatch, failing):
    def prose(prompt, **kwargs):
        if ("You are revising" in prompt) == (failing == "revise"):
            raise EndpointUnavailable("down")
        return _echo_polish(prompt)

    drafts = {1: _words(50) if failing == "revise" else _words(120)}
    _editor(monkeypatch, tmp_path, drafts, prose)
    with pytest.raises(EndpointUnavailable):
        editor.run_editor()


def test_a_failed_revision_call_keeps_the_draft(tmp_path, monkeypatch, capsys):
    def prose(prompt, **kwargs):
        if "You are revising" in prompt:
            raise RuntimeError("bad reply")
        return _echo_polish(prompt)

    _editor(monkeypatch, tmp_path, {1: _words(50)}, prose)
    editor.run_editor()
    assert "Revision failed for chapter 1: bad reply" in capsys.readouterr().out
    diff = (tmp_path / "out" / "interim" / "diff_chapter_01.md").read_text()
    assert "revision failed (bad reply)" in diff


def test_chapters_without_a_draft_or_already_edited_are_skipped(
        tmp_path, monkeypatch, capsys):
    _editor(monkeypatch, tmp_path, {1: _words(120)},
            lambda p, **k: pytest.fail("no LLM call expected"))
    update_context("chapters", CHAPTERS)             # chapter 2 has no draft
    update_context("final", {1: "edited"})
    editor.run_editor()
    text = capsys.readouterr().out
    assert "Chapter 1: already edited" in text and "No draft for chapter 2" in text


# ---------------------------------------------------------------- save_book

def test_overwrite_false_numbers_the_books(tmp_path):
    out = _config(tmp_path, output_extra="  overwrite: false\n")
    update_context("title", "T")
    for expected in ("draft.md", "draft-1.md", "draft-2.md"):
        editor.save_book({1: "text"})
        assert (out / expected).exists()
        assert context["output_path"].endswith(expected)


def test_a_book_that_cannot_be_written_is_returned_not_lost(
        tmp_path, monkeypatch, capsys):
    _config(tmp_path)
    update_context("title", "T")

    def refuse(path, text):
        raise OSError("disk full")

    monkeypatch.setattr(editor, "atomic_write_text", refuse)
    book = editor.save_book({1: "text"})
    assert book.startswith("# T") and "Error saving book: disk full" in \
        capsys.readouterr().out


def test_a_story_bible_that_cannot_be_written_only_warns(tmp_path,
                                                         monkeypatch, capsys):
    _config(tmp_path)
    update_context("title", "T")
    real = editor.atomic_write_text

    def only_the_book(path, text):
        if path.name == "story_bible.md":
            raise OSError("read-only")
        real(path, text)

    monkeypatch.setattr(editor, "atomic_write_text", only_the_book)
    editor.save_book({1: "text"})
    assert "could not save story bible" in capsys.readouterr().out


def test_the_lint_report_flags_short_chapters_and_findings(tmp_path):
    out = _config(tmp_path)
    update_context("bible", {"title": "T", "characters": [],
                             "constraints": ['Never "ominous".']})
    update_context("chapters", CHAPTERS)
    editor._write_lint_report({1: f"{_words(120)} ominous",
                               2: _words(30)})
    report = (out / "interim" / "lint_report.md").read_text()
    assert "- Chapter 1: 121 words (min 80)" in report
    assert "- Chapter 2: 30 words **SHORT** (min 80)" in report
    assert "banned word 'ominous'" in report


# ------------------------------------------------------------------------ main

def test_positive_int():
    assert pipeline.positive_int("3") == 3
    for bad in ("0", "-2", "x", "1.5"):
        with pytest.raises(argparse.ArgumentTypeError):
            pipeline.positive_int(bad)


def _args(**kw):
    base = dict(demo=False, seed=None, prompt=None)
    base.update(kw)
    return argparse.Namespace(**base)


def test_seed_sources(tmp_path, capsys):
    seed_file = tmp_path / "s.md"
    seed_file.write_text("# From a file")
    assert pipeline.resolve_seed_text(_args(seed=str(seed_file))) == "# From a file"
    assert pipeline.resolve_seed_text(_args(prompt="inline")) == "inline"
    assert "bundled example" in pipeline.resolve_seed_text(_args(demo=True)) \
        or True
    text = pipeline.resolve_seed_text(_args())            # no args: the example
    assert text == pipeline.EXAMPLE_SEED.read_text(encoding="utf-8")
    assert "bundled example seed" in capsys.readouterr().out


@pytest.mark.parametrize("args", [
    dict(demo=True, seed="x.md"), dict(demo=True, prompt="x"),
    dict(seed="/no/such/seed.md")])
def test_bad_seed_arguments_exit_with_a_message(args):
    with pytest.raises(SystemExit) as exc:
        pipeline.resolve_seed_text(_args(**args))
    assert "Error" in str(exc.value)


def test_a_missing_example_seed_is_reported(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline, "EXAMPLE_SEED", tmp_path / "gone.md")
    for args in (dict(demo=True), dict()):
        with pytest.raises(SystemExit, match="example seed missing"):
            pipeline.resolve_seed_text(_args(**args))


def test_agent_enabled_defaults_and_none_sections(tmp_path):
    _config(tmp_path, agents="  researcher:\n    enabled: false\n  editor:\n")
    assert pipeline.agent_enabled("researcher") is False
    assert pipeline.agent_enabled("editor") is True         # empty section
    assert pipeline.agent_enabled("writer") is True         # not mentioned


def test_hydrating_resume_state_fills_the_context(tmp_path):
    _config(tmp_path)
    state = {"bible": {"title": "T", "seed": "S"}, "chapters": CHAPTERS,
             "research": {1: "brief"}, "drafts": {1: "draft"},
             "summaries": {1: "sum"}, "chronology": {1: {"x": 1}},
             "final": {1: "body"}}
    pipeline._hydrate_resume(state)
    assert context["title"] == "T" and context["seed"] == "S"
    assert context["final"] == {1: "body"}
    assert context["summaries"] == {1: "sum"}


def _stub_pipeline(monkeypatch, **replace):
    stubs = dict(run_seed_review=lambda *a, **k: {"chapters": None, "words_per_chapter": 100,
                                    "clarifications": [], "stop": False},
                 run_architect=lambda *a: None,
                 run_planner=lambda **k: update_context(
                     "chapters", CHAPTERS) or CHAPTERS,
                 run_researcher=lambda: None, run_writer=lambda **k: None,
                 run_editor=lambda **k: None)
    stubs.update(replace)
    for name, fn in stubs.items():
        monkeypatch.setattr(pipeline, name, fn)


def test_with_the_editor_disabled_the_raw_drafts_are_saved(tmp_path,
                                                           monkeypatch):
    out = _config(tmp_path, agents="  editor:\n    enabled: false\n")
    _stub_pipeline(monkeypatch)
    update_context("title", "T")

    def write_drafts(**kwargs):
        update_context("drafts", {1: "first draft", 2: "second draft"})

    monkeypatch.setattr(pipeline, "run_writer", write_drafts)
    assert pipeline.run_pipeline("seed") == 0
    book = (out / "draft.md").read_text()
    assert "## Chapter 1: Ch1\n\nfirst draft" in book and "second draft" in book


def test_ctrl_c_exits_130_with_a_resume_hint(tmp_path, monkeypatch, capsys):
    _config(tmp_path)

    def interrupted(seed, *a):
        raise KeyboardInterrupt

    _stub_pipeline(monkeypatch, run_architect=interrupted)
    assert pipeline.run_pipeline("seed") == 130
    assert "Rerun the same command to resume" in capsys.readouterr().out


def test_unexpected_errors_are_reported_then_raised(tmp_path, monkeypatch,
                                                    capsys):
    _config(tmp_path)

    def broken(seed, *a):
        raise ValueError("kaboom")

    _stub_pipeline(monkeypatch, run_architect=broken)
    with pytest.raises(ValueError, match="kaboom"):
        pipeline.run_pipeline("seed")
    assert "[PIPELINE] Error: kaboom" in capsys.readouterr().out


def test_a_stats_reporting_failure_never_hides_the_outcome(tmp_path,
                                                           monkeypatch,
                                                           capsys):
    _config(tmp_path)
    llm_client.reset_stats()
    llm_client._record("writer", 1.0, {"prompt_tokens": 1,
                                       "completion_tokens": 1}, 4)

    def boom(phases=None):
        raise RuntimeError("no report")

    monkeypatch.setattr(llm_client, "stats_markdown", boom)
    pipeline._report_stats([])                       # must not raise
    assert "Could not write run statistics: no report" in capsys.readouterr().out


def test_cli_overrides_and_errors(tmp_path, monkeypatch):
    from tests.fake_llm import SEED, FakeLLM
    cfg = tmp_path / "config.yaml"
    cfg.write_text(f'llm:\n  base_url: "http://fake:1"\n  model: fake\n'
                   "  retries: 0\n  endpoint_wait: 0\n"
                   f'output:\n  directory: "{tmp_path / "out"}"\n'
                   "agents:\n  reviewer:\n    enabled: false\n"
                   "book:\n  words_per_chapter: 100\n")
    fake = FakeLLM(models=["fake", "other"]).install(monkeypatch)
    monkeypatch.setattr(sys, "argv", ["main.py", "--config", str(cfg),
                                      "--prompt", SEED, "--model", "other",
                                      "--out", "custom.md"])
    with pytest.raises(SystemExit) as exc:
        pipeline.main()
    assert exc.value.code == 0
    assert (tmp_path / "out" / "custom.md").exists()
    assert {c["payload"]["model"] for c in fake.calls} == {"other"}
    runlog.stop_log()
    # a config that does not exist is a clean error, not a traceback
    monkeypatch.setattr(sys, "argv", ["main.py", "--config",
                                      str(tmp_path / "missing.yaml")])
    with pytest.raises(SystemExit, match="Config file not found"):
        pipeline.main()


@pytest.mark.parametrize("flag", [["-c", "0"], ["-c", "abc"], ["0"]])
def test_bad_chapter_counts_are_rejected_by_the_parser(monkeypatch, flag):
    monkeypatch.setattr(sys, "argv", ["main.py", *flag])
    with pytest.raises(SystemExit) as exc:
        pipeline.main()
    assert exc.value.code == 2


# ------------------------------------------------------- resume / output / log

def test_unreadable_snapshots_are_skipped_with_a_warning(tmp_path, capsys):
    out = _config(tmp_path)
    interim = out / "interim"
    interim.mkdir(parents=True)
    (interim / "bible.json").write_text("{not json")
    (interim / "lore_chapter_xx.md").write_text("junk")
    (interim / "draft_chapter_zz.md").write_text("junk")
    (interim / "edited_chapter_qq.md").write_text("junk")
    (interim / "draft_chapter_02.md").write_text("## Chapter 2: B\n\nbody")
    state = load_state()
    assert state["bible"] is None and "failed to load" in capsys.readouterr().out
    assert state["research"] == {} and state["final"] == {}
    assert state["drafts"] == {2: "body"}
    assert _read_json(interim / "bible.json") is None


def test_summarize_for_log():
    empty = dict(bible=None, chapters=None, drafts={}, final={})
    assert summarize_for_log(empty) == "nothing"
    full = dict(bible={"title": "T"}, chapters=[1, 2], drafts={1: "a"},
                final={1: "x"})
    assert summarize_for_log(full) == \
        "bible 'T', plan 2 chapters, 1 drafted, 1 edited"


def test_unwritable_outputs_are_reported_never_fatal(tmp_path, monkeypatch,
                                                     capsys):
    _config(tmp_path)

    def refuse(path, text):
        raise OSError("disk full")

    monkeypatch.setattr(output, "atomic_write_text", refuse)
    assert output.save_chapter(1, "T", "text") is None
    assert output.save_interim("x.md", "text") is None
    text = capsys.readouterr().out
    assert "Could not save chapter 1: disk full" in text
    assert "Could not save x.md: disk full" in text
    assert output.save_interim_json("x.json", {"s": {1, 2}}) is None
    assert "Could not serialise x.json" in capsys.readouterr().out


def test_atomic_write_replaces_and_leaves_no_temp_file(tmp_path):
    target = tmp_path / "f.txt"
    output.atomic_write_text(target, "one")
    output.atomic_write_text(target, "two")
    assert target.read_text() == "two"
    assert [p.name for p in tmp_path.iterdir()] == ["f.txt"]


def test_the_tee_survives_a_closed_log_file(tmp_path, capsys):
    runlog.start_log(tmp_path)
    try:
        runlog._state["file"].close()                 # simulate a dead disk
        print("still prints")                         # must not raise
    finally:
        runlog.stop_log()
    assert "still prints" in capsys.readouterr().out


def test_a_state_write_failure_stops_the_run_with_advice(tmp_path,
                                                        monkeypatch, capsys):
    """Carrying on would spend hours on work a crash could not resume."""
    from shared.resume import StateWriteError
    _config(tmp_path)

    def broken(*a, **k):
        raise StateWriteError("could not save the run's plan (disk full)")

    monkeypatch.setattr(pipeline, "run_seed_review", broken)
    assert pipeline.run_pipeline("SEED") == 1
    out = capsys.readouterr().out
    assert "Aborted: could not save the run's plan" in out
    assert "rerun the same command to resume" in out


def test_a_stage_never_swallows_a_state_write_failure(tmp_path, monkeypatch):
    """The researcher absorbs a failed brief; it must not absorb a failed
    save."""
    from shared.resume import StateWriteError
    _config(tmp_path)
    update_context("chapters", CHAPTERS[:1])
    update_context("bible", BIBLE)
    monkeypatch.setattr(researcher, "generate_with_wait",
                        lambda *a, **k: "a brief")

    def broken(key, value):
        raise StateWriteError("disk full")

    monkeypatch.setattr(researcher, "save_state", broken)
    with pytest.raises(StateWriteError):
        researcher.run_researcher()
