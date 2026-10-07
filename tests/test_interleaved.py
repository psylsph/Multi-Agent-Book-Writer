"""book.review_as_you_go: write, review and polish one chapter at a time."""

import json

import main as pipeline
from agents import editor, writer
from shared.context import context, reset_context, update_context
from shared.llm_client import load_config
from shared.story_state import minimal_state

BIBLE = {"title": "T", "genre": "g", "tone": "t", "premise": "p", "world": "w",
         "constraints": [], "characters": [{"name": "Aria", "role": "",
                                            "description": ""}]}
CHAPTERS = [{"number": n, "title": f"Ch{n}", "summary": f"s{n}", "part": ""}
            for n in (1, 2, 3)]


def _config(tmp_path, book="", editor_on=True):
    path = tmp_path / "config.yaml"
    path.write_text(
        f'output:\n  directory: "{tmp_path / "out"}"\n'
        f"book:\n  words_per_chapter: 100\n{book}"
        f"agents:\n  reviewer:\n    enabled: false\n"
        f"  editor:\n    enabled: {str(editor_on).lower()}\n")
    load_config(path)
    reset_context()
    update_context("bible", BIBLE)
    update_context("chapters", CHAPTERS)
    update_context("title", "T")


def _words(n, prefix="alpha"):
    return " ".join(f"{prefix}{i % 7}" for i in range(n))


# ------------------------------------------------------------------ the switch

def test_review_as_you_go_defaults_to_off(tmp_path):
    _config(tmp_path)
    assert pipeline.review_as_you_go() is False


def test_review_as_you_go_on(tmp_path):
    _config(tmp_path, "  review_as_you_go: true\n")
    assert pipeline.review_as_you_go() is True


def test_review_as_you_go_needs_the_editor(tmp_path, capsys):
    _config(tmp_path, "  review_as_you_go: true\n", editor_on=False)
    assert pipeline.review_as_you_go() is False
    assert "needs the editor" in capsys.readouterr().out


# ------------------------------------------------------------ loop order (stubs)

def _stub_loop(monkeypatch, drafted=(1, 2, 3)):
    events = []

    def write(only=None):
        events.append(("write", only))
        if only in drafted:
            context["drafts"][only] = "draft"

    def edit(only=None):
        events.append(("edit", only))
        context.setdefault("completed_chapters", set()).add(only)

    monkeypatch.setattr(pipeline, "run_writer", write)
    monkeypatch.setattr(pipeline, "run_editor", edit)
    monkeypatch.setattr(pipeline, "refresh_state",
                        lambda n: events.append(("refresh", n)))
    monkeypatch.setattr(pipeline, "finalize_book",
                        lambda final: events.append(("finalize", None)))
    return events


def test_each_chapter_is_written_edited_and_refreshed_before_the_next(
        tmp_path, monkeypatch):
    _config(tmp_path)
    events = _stub_loop(monkeypatch)
    pipeline.run_interleaved()
    assert events == [
        ("write", 1), ("edit", 1), ("refresh", 1),
        ("write", 2), ("edit", 2), ("refresh", 2),
        ("write", 3), ("edit", 3), ("refresh", 3),
        ("finalize", None)]


def test_loop_stops_at_a_chapter_that_cannot_be_drafted(tmp_path, monkeypatch):
    _config(tmp_path)
    events = _stub_loop(monkeypatch, drafted=(1, 3))      # chapter 2 fails
    pipeline.run_interleaved()
    # chapter 2 was attempted, failed, and nothing ran after it
    assert events == [("write", 1), ("edit", 1), ("refresh", 1), ("write", 2)]


def test_already_edited_chapters_are_not_edited_or_refreshed_again(
        tmp_path, monkeypatch):
    _config(tmp_path)
    update_context("completed_chapters", {1})
    events = _stub_loop(monkeypatch)
    pipeline.run_interleaved()
    assert ("edit", 1) not in events and ("refresh", 1) not in events
    assert ("write", 1) in events                  # the writer skips it itself
    assert ("edit", 2) in events


# ----------------------------------------------- per-chapter modes (real code)

def test_writer_only_writes_the_requested_chapter(tmp_path, monkeypatch):
    _config(tmp_path)
    monkeypatch.setattr(writer, "generate_prose", lambda p, **k: _words(120))
    monkeypatch.setattr(writer, "generate_with_wait", lambda p, **k: json.dumps(
        {"summary": "S", "events": []}))
    writer.run_writer(only=2)
    assert sorted(context["drafts"]) == [2]


def test_editor_only_edits_the_requested_chapter(tmp_path, monkeypatch):
    _config(tmp_path)
    update_context("drafts", {1: _words(120), 2: _words(120)})
    update_context("chronology", {})

    def prose(prompt, **kwargs):
        start = prompt.index("## Chapter")
        return prompt[start:prompt.index("\n\nReturn ONLY")]

    monkeypatch.setattr(editor, "generate_prose", prose)
    editor.run_editor(only=2)
    assert context["completed_chapters"] == {2}
    assert [e.split(":")[0] for e in context["final"]] == ["## Chapter 2"]
    assert (tmp_path / "out" / "draft.md").exists()      # partial book on disk


def test_refresh_state_uses_the_edited_text(tmp_path, monkeypatch):
    _config(tmp_path)
    seen = {}

    def extract(prompt, **kwargs):
        seen["prompt"] = prompt
        return json.dumps({"summary": "NEW", "time": "dusk", "location": "inn",
                           "present": ["Aria"], "events": []})

    monkeypatch.setattr(writer, "generate_with_wait", extract)
    update_context("drafts", {1: "ORIGINAL DRAFT"})
    update_context("summaries", {1: "OLD"})
    update_context("chronology", {1: minimal_state(1, "Ch1", "OLD")})
    update_context("final", ["## Chapter 1: Ch1\n\nEDITED TEXT"])
    writer.refresh_state(1)
    assert "EDITED TEXT" in seen["prompt"] and "ORIGINAL" not in seen["prompt"]
    assert context["summaries"][1] == "NEW"
    assert context["chronology"][1]["location"] == "inn"


def test_refresh_state_keeps_the_old_state_when_extraction_fails(
        tmp_path, monkeypatch):
    _config(tmp_path)
    monkeypatch.setattr(writer, "generate_with_wait", lambda p, **k: "garbage")
    good = {**minimal_state(1, "Ch1", "GOOD"), "location": "kept"}
    update_context("drafts", {1: "draft"})
    update_context("summaries", {1: "GOOD"})
    update_context("chronology", {1: good})
    update_context("final", ["## Chapter 1: Ch1\n\nEDITED"])
    writer.refresh_state(1)
    assert context["summaries"][1] == "GOOD"
    assert context["chronology"][1]["location"] == "kept"


# --------------------------------------------- the whole loop, real agent code

def test_next_chapter_is_written_from_the_edited_chapter(tmp_path, monkeypatch):
    """Chapter 2's prompt must carry facts and an ending taken from chapter 1
    AFTER editing, not from its first draft."""
    _config(tmp_path, "  review_as_you_go: true\n")
    writer_prompts = []

    def write(prompt, **kwargs):
        writer_prompts.append(prompt)
        return _words(120)

    def edit(prompt, **kwargs):
        start = prompt.index("## Chapter")
        text = prompt[start:prompt.index("\n\nReturn ONLY")]
        return text + " EDITEDMARK"        # what the polish pass "did"

    def extract(prompt, **kwargs):
        edited = "EDITEDMARK" in prompt
        return json.dumps({"summary": f"EDITED={edited}", "time": "day",
                           "location": "town", "present": ["Aria"],
                           "events": []})

    monkeypatch.setattr(writer, "generate_prose", write)
    monkeypatch.setattr(writer, "generate_with_wait", extract)
    monkeypatch.setattr(editor, "generate_prose", edit)
    pipeline.run_interleaved()

    assert len(writer_prompts) == 3
    second = writer_prompts[1]
    assert "EDITED=True" in second          # facts re-extracted after editing
    assert "EDITEDMARK" in second           # ending taken from the edited text
    assert sorted(context["completed_chapters"]) == [1, 2, 3]
    book = (tmp_path / "out" / "draft.md").read_text()
    assert book.count("EDITEDMARK") == 3
    assert [c for c in range(1, 4) if f"## Chapter {c}:" in book] == [1, 2, 3]
