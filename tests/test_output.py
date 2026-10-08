"""Unit tests for interim output helpers (no LLM, no real output dir)."""

from pathlib import Path

from shared.llm_client import load_config
from shared.output import (chapter_filename, chapters_dir, clear_interim,
                           format_bible_markdown, interim_enabled,
                           save_chapter, save_interim)


def _use_config(tmp_path, interim=True, directory=None):
    directory = directory or (tmp_path / "out")
    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        f"output:\n  directory: \"{directory}\"\n  interim: {str(interim).lower()}\n"
    )
    load_config(cfg)
    return Path(directory)


def test_save_interim_writes_file(tmp_path):
    out = _use_config(tmp_path)
    path = save_interim("outline.md", "# Outline\n\n1. Chapter One")
    assert path is not None
    assert path == out / "interim" / "outline.md"
    assert path.read_text(encoding="utf-8").startswith("# Outline")


def test_save_interim_disabled(tmp_path):
    _use_config(tmp_path, interim=False)
    assert interim_enabled() is False
    assert save_interim("outline.md", "text") is None
    assert not (tmp_path / "out" / "interim").exists()


def test_save_interim_empty_text_is_noop(tmp_path):
    _use_config(tmp_path)
    assert save_interim("outline.md", "") is None


def test_clear_interim_removes_old_run_files(tmp_path):
    _use_config(tmp_path)
    save_interim("draft_chapter_01.md", "old draft")
    clear_interim()
    assert not (tmp_path / "out" / "interim").exists()


def test_clear_interim_respects_disabled_flag(tmp_path):
    out = _use_config(tmp_path, interim=False)
    interim = out / "interim"
    interim.mkdir(parents=True)
    (interim / "stale.md").write_text("stale")
    clear_interim()  # disabled -> leaves the directory alone
    assert (interim / "stale.md").exists()


def test_save_chapter_writes_durable_file(tmp_path):
    out = _use_config(tmp_path)
    path = save_chapter(3, "The Long Road", "It was a dark night.")
    assert path is not None
    assert path == out / "chapters" / "chapter_03.md"
    text = path.read_text(encoding="utf-8")
    assert text.startswith("# Chapter 3: The Long Road\n\n")
    assert "It was a dark night." in text


def test_save_chapter_ignores_interim_flag(tmp_path):
    _use_config(tmp_path, interim=False)
    path = save_chapter(1, "Title", "Prose.")
    assert path is not None
    assert path.exists()


def test_save_chapter_survives_clear_interim(tmp_path):
    out = _use_config(tmp_path)
    save_chapter(2, "Title", "Prose.")
    save_interim("draft_chapter_02.md", "Prose.")
    clear_interim()
    assert (out / "chapters" / "chapter_02.md").exists()
    assert not (out / "interim").exists()


def test_save_chapter_empty_text_is_noop(tmp_path):
    _use_config(tmp_path)
    assert save_chapter(1, "Title", "") is None


def test_chapters_dir_creates_directory(tmp_path):
    out = _use_config(tmp_path)
    assert chapters_dir() == out / "chapters"
    assert (out / "chapters").is_dir()


def test_chapter_filename_zero_pads():
    assert chapter_filename("draft", 3) == "draft_chapter_03.md"
    assert chapter_filename("edited", 12) == "edited_chapter_12.md"


def test_format_bible_markdown_sections():
    bible = {
        "title": "My Book",
        "premise": "A story.",
        "genre": "mystery",
        "tone": "spare",
        "characters": [{"name": "Aria", "role": "protagonist",
                        "description": "a keeper"}],
        "world": "An island.",
        "constraints": ["All characters are adults."],
        "notes": "Very explicit, no euphemisms.",
    }
    text = format_bible_markdown(bible)
    assert "# My Book - Story Bible" in text
    assert "- **Aria** (protagonist): a keeper" in text
    assert "- All characters are adults." in text
    assert "Very explicit, no euphemisms." in text


# ----------------------------------------------------- archive, never delete

def _populate(out):
    (out / "interim").mkdir(parents=True)
    (out / "interim" / "bible.json").write_text("{}")
    (out / "chapters").mkdir()
    (out / "chapters" / "chapter_01.md").write_text("# Chapter 1\n\nText")
    (out / "draft.md").write_text("the old book")
    (out / "story_bible.md").write_text("bible")


def test_archive_moves_the_previous_run_instead_of_deleting(tmp_path):
    from shared.output import archive_previous_run
    out = _use_config(tmp_path)
    _populate(out)
    dest = archive_previous_run()
    assert dest.parent == out / "archive"
    assert (dest / "interim" / "bible.json").exists()
    assert (dest / "chapters" / "chapter_01.md").read_text().endswith("Text")
    assert (dest / "draft.md").read_text() == "the old book"
    assert (dest / "story_bible.md").exists()
    # the output dir is clean for the new run
    for gone in ("interim", "chapters", "draft.md", "story_bible.md"):
        assert not (out / gone).exists()


def test_archive_is_a_noop_when_there_is_nothing_to_archive(tmp_path):
    from shared.output import archive_previous_run
    out = _use_config(tmp_path)
    assert archive_previous_run() is None
    (out / "chapters").mkdir(parents=True)               # empty dir
    assert archive_previous_run() is None
    assert not (out / "archive").exists()


def test_two_archives_in_the_same_second_do_not_collide(tmp_path, monkeypatch):
    from shared import output
    out = _use_config(tmp_path)
    monkeypatch.setattr(output.time, "strftime", lambda fmt: "20260101-000000")
    _populate(out)
    first = output.archive_previous_run()
    _populate(out)
    second = output.archive_previous_run()
    assert first != second and first.exists() and second.exists()


def test_archive_keeps_the_book_when_overwrite_is_off(tmp_path):
    from shared.output import archive_previous_run
    directory = tmp_path / "out"
    cfg = tmp_path / "config.yaml"
    cfg.write_text(f'output:\n  directory: "{directory}"\n  overwrite: false\n')
    from shared.llm_client import load_config
    load_config(cfg)
    _populate(directory)
    dest = archive_previous_run()
    assert (directory / "draft.md").exists()        # -N naming handles it
    assert not (dest / "draft.md").exists()


def test_a_fresh_pipeline_run_archives_before_it_starts(tmp_path, monkeypatch):
    import main as pipeline
    out = _use_config(tmp_path)
    _populate(out)
    monkeypatch.setattr(pipeline, "run_seed_review", lambda *a, **k: {"chapters": None, "words_per_chapter": 100,
                                    "clarifications": [], "stop": False})
    monkeypatch.setattr(pipeline, "run_architect", lambda *a: None)
    monkeypatch.setattr(pipeline, "run_planner", lambda **k: [])   # stops early
    assert pipeline.run_pipeline("A seed.", resuming=False) == 1
    (archive,) = list((out / "archive").iterdir())
    assert (archive / "chapters" / "chapter_01.md").exists()
    assert not (out / "chapters" / "chapter_01.md").exists()
