"""Reviewer checks, cross-chapter repetition lint and edit diffs."""

import json

import pytest

from agents import editor, reviewer
from shared.consistency import (lint_book, phrase_repeats,
                                repeated_phrase_finding,
                                repeated_phrases_in_book)
from shared.context import context, reset_context, update_context
from shared.llm_client import EndpointUnavailable, load_config

BIBLE = {"title": "T", "world": "A marsh", "constraints": [
    "Close third person, past tense, one POV per chapter.",
    'Never "unhurried".'],
    "characters": [{"name": "Aria", "role": "", "description": ""},
                   {"name": "Tom", "role": "", "description": ""}]}
CHAPTERS = [{"number": n, "title": f"Ch{n}", "summary": f"Outline {n}: Aria "
             "finds the hidden letter.", "part": ""} for n in (1, 2, 3)]


def _config(tmp_path, book=""):
    path = tmp_path / "config.yaml"
    path.write_text(f'output:\n  directory: "{tmp_path / "out"}"\n'
                    f"book:\n  words_per_chapter: 100\n{book}"
                    "agents:\n  reviewer:\n    enabled: false\n")
    load_config(path)
    reset_context()
    update_context("bible", BIBLE)
    update_context("chapters", CHAPTERS)
    update_context("chronology", {})
    update_context("research", {1: "- Plot beat: Aria finds the hidden letter."})


def _words(n, prefix="alpha"):
    return " ".join(f"{prefix}{i % 7}" for i in range(n))


# ------------------------------------------------------------ reviewer checks

class Reply:
    def __init__(self, payload):
        self.payload, self.calls = payload, []

    def __call__(self, prompt, **kwargs):
        self.calls.append({"prompt": prompt, **kwargs})
        return self.payload if isinstance(self.payload, str) \
            else json.dumps(self.payload)


def _review(tmp_path, monkeypatch, payload, book=""):
    _config(tmp_path, book)
    reply = Reply(payload)
    monkeypatch.setattr(reviewer, "generate_with_wait", reply)
    return reviewer.run_reviewer(1, "Ch1", "A draft."), reply


def test_all_three_checks_are_on_by_default(tmp_path, monkeypatch):
    _, reply = _review(tmp_path, monkeypatch, {"verdict": "pass", "issues": []})
    prompt = reply.calls[0]["prompt"]
    assert "STORY FACTS BEFORE THIS CHAPTER" in prompt
    assert "CHAPTER OUTLINE" in prompt and "Aria finds the hidden letter" in prompt
    assert "LORE BRIEF (plot beats to hit)" in prompt
    assert "AUTHOR CONSTRAINTS" in prompt and "Close third person" in prompt
    assert "outline_gap" in prompt and "constraint" in prompt
    assert "dead_resurrection" in prompt


def test_outline_only_review_leaves_out_the_other_checks(tmp_path,
                                                         monkeypatch):
    (verdict, issues), reply = _review(
        tmp_path, monkeypatch,
        {"verdict": "revise", "issues": [
            {"type": "outline_gap", "description": "the letter is never found",
             "fix": "add a short discovery"},
            {"type": "timeline", "description": "a continuity complaint",
             "fix": ""}]},
        book="  review_checks: [outline]\n")
    prompt = reply.calls[0]["prompt"]
    assert "STORY FACTS" not in prompt and "AUTHOR CONSTRAINTS" not in prompt
    assert "CHAPTER OUTLINE" in prompt
    # the continuity-type issue belongs to a disabled check and is dropped
    assert verdict == "revise"
    assert [i["type"] for i in issues] == ["outline_gap"]


def test_constraint_issue_is_reported(tmp_path, monkeypatch):
    (verdict, issues), _ = _review(
        tmp_path, monkeypatch,
        {"verdict": "revise", "issues": [
            {"type": "constraint", "description": "head-hops into Tom's POV",
             "fix": "keep to Aria"}]},
        book="  review_checks: [constraints]\n")
    assert verdict == "revise" and issues[0]["type"] == "constraint"


def test_unknown_check_names_fall_back_to_all(tmp_path, monkeypatch):
    _config(tmp_path, "  review_checks: [vibes]\n")
    assert reviewer.enabled_checks() == reviewer.ALL_CHECKS
    _config(tmp_path, "  review_checks: constraints\n")     # a bare string
    assert reviewer.enabled_checks() == ("constraints",)


def test_verdict_follows_the_issues_not_the_models_label(tmp_path,
                                                         monkeypatch):
    (verdict, issues), _ = _review(tmp_path, monkeypatch, {
        "verdict": "pass", "issues": [
            {"type": "setting", "description": "wrong room", "fix": ""}]})
    assert verdict == "revise" and len(issues) == 1
    (verdict, issues), _ = _review(tmp_path, monkeypatch,
                                   {"verdict": "revise", "issues": []})
    assert verdict == "pass" and issues == []


def test_review_file_records_what_was_checked(tmp_path, monkeypatch):
    _review(tmp_path, monkeypatch, {"verdict": "pass", "issues": []},
            book="  review_checks: [continuity, outline]\n")
    text = (tmp_path / "out" / "interim" / "review_chapter_01.md").read_text()
    assert "PASS" in text and "checked: continuity, outline" in text


def test_garbage_review_never_blocks_the_pipeline(tmp_path, monkeypatch):
    (verdict, issues), _ = _review(tmp_path, monkeypatch, "not json at all")
    assert (verdict, issues) == ("pass", [])


def test_endpoint_outage_is_not_swallowed(tmp_path, monkeypatch):
    _config(tmp_path)

    def down(*a, **k):
        raise EndpointUnavailable("down")

    monkeypatch.setattr(reviewer, "generate_with_wait", down)
    with pytest.raises(EndpointUnavailable):
        reviewer.run_reviewer(1, "Ch1", "A draft.")


# --------------------------------------------------- repetition (deterministic)

PHRASE = "a shiver ran down her spine as the door creaked open slowly"
EARLIER = [f"{PHRASE} in chapter one. " + _words(40, "one"),
           f"Again {PHRASE} and the lamp went out. " + _words(40, "two")]


def test_phrase_reused_from_earlier_chapters_is_found():
    found = phrase_repeats(f"Later, {PHRASE}. Then tea.", EARLIER, ["Aria"])
    assert len(found) == 1
    phrase, count = found[0]
    assert "shiver ran down her spine" in phrase and count == 2


def test_a_phrase_used_only_once_before_is_not_flagged_yet():
    assert phrase_repeats(f"{PHRASE}.", EARLIER[:1]) == []
    assert phrase_repeats(f"{PHRASE}.", EARLIER[:1], min_prior=1) != []


def test_function_word_phrases_and_names_are_ignored():
    filler = "and then it was all of the way up in the air"
    assert phrase_repeats(filler, [filler, filler, filler]) == []
    said = "Aria said to Tom that Aria said to Tom"
    assert phrase_repeats(said, [said] * 3, names=["Aria", "Tom"]) == []


def test_repeated_phrase_finding_shape():
    finding = repeated_phrase_finding(f"{PHRASE}.", EARLIER, ["Aria"])
    assert finding["check"] == "repeated_phrase"
    assert "reword" in finding["detail"] and "used 2x before" in finding["detail"]
    assert repeated_phrase_finding("A wholly fresh sentence about rivers.",
                                   EARLIER) is None


def test_book_wide_repetition_report():
    texts = {1: EARLIER[0], 2: EARLIER[1], 3: f"Last: {PHRASE}. Done."}
    ((phrase, count, chapters),) = repeated_phrases_in_book(texts, ["Aria"])
    assert "shiver ran down her spine" in phrase
    assert count == 3 and chapters == [1, 2, 3]
    assert repeated_phrases_in_book({1: EARLIER[0], 2: "Nothing shared."}) == []


def test_lint_book_reports_repeated_phrases():
    texts = {1: EARLIER[0], 2: EARLIER[1], 3: f"Last: {PHRASE}."}
    findings = lint_book("\n".join(texts.values()), BIBLE, [], (), texts)
    assert any(f["check"] == "repeated_phrase" and "chapters 1, 2, 3"
               in f["detail"] for f in findings)
    assert not any(f["check"] == "repeated_phrase"
                   for f in lint_book("x", BIBLE, [], (), None))


# --------------------------------------------- editor: repetition and diffs

def _editor_run(tmp_path, monkeypatch, book="", revise=None, only=3):
    _config(tmp_path, book)
    drafts = {1: EARLIER[0] + " " + _words(60, "beta"),
              2: EARLIER[1] + " " + _words(60, "gamma"),
              3: f"Chapter three: {PHRASE}. " + _words(110, "delta")}
    update_context("drafts", drafts)
    prompts = []

    def prose(prompt, **kwargs):
        prompts.append(prompt)
        if "You are revising" in prompt:
            return revise if revise is not None else _words(120, "fresh")
        start = prompt.index("## Chapter")
        return prompt[start:prompt.index("\n\nReturn ONLY")]

    monkeypatch.setattr(editor, "generate_prose", prose)
    editor.run_editor(only=only)
    return prompts


def test_editor_asks_for_a_reused_phrase_to_be_reworded(tmp_path, monkeypatch):
    prompts = _editor_run(tmp_path, monkeypatch)
    revise = next(p for p in prompts if "You are revising" in p)
    assert "already used 2+ times in earlier chapters" in revise
    assert "shiver ran down her spine" in revise
    assert "For repeated_phrase findings, reword" in revise
    assert PHRASE not in context["final"][-1]


def test_repetition_lint_can_be_switched_off(tmp_path, monkeypatch):
    prompts = _editor_run(tmp_path, monkeypatch,
                          book="  repetition_lint: false\n")
    assert not any("You are revising" in p for p in prompts)


def test_earlier_chapters_use_their_edited_text(tmp_path, monkeypatch):
    update = editor._earlier_bodies(
        3, {1: "draft one", 2: "draft two"},
        ["## Chapter 1: A\n\nedited one"])
    assert update == ["edited one", "draft two"]
    assert editor._earlier_bodies(1, {1: "x"}, []) == []


def test_editor_saves_a_diff_with_its_decisions(tmp_path, monkeypatch):
    _editor_run(tmp_path, monkeypatch)
    diff = (tmp_path / "out" / "interim" / "diff_chapter_03.md").read_text()
    assert diff.startswith("# Edit diff - Chapter 3: Ch3")
    assert "reviewer" not in diff                       # reviewer is off here
    assert "lint on the draft: 1 finding(s)" in diff and "repeated_phrase" in diff
    assert "round 1: revision accepted" in diff
    assert "polish accepted" in diff
    assert "```diff" in diff and "-Chapter three" in diff and "+fresh0" in diff


def test_diff_report_for_an_unchanged_chapter():
    text = "One sentence. Two sentences."
    report = editor.diff_report(1, "T", text, text, ["polish accepted"])
    assert "(no changes)" in report and "sentences unchanged: 100%" in report


def test_rejected_revision_is_recorded_in_the_diff(tmp_path, monkeypatch):
    shrunk = _words(20, "tiny")                     # < 100 words: discarded
    _editor_run(tmp_path, monkeypatch, revise=shrunk)
    diff = (tmp_path / "out" / "interim" / "diff_chapter_03.md").read_text()
    assert "revision discarded (too short)" in diff
