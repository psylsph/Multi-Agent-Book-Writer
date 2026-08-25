"""Unit tests for interim output helpers (no LLM, no real output dir)."""

from pathlib import Path

from shared.llm_client import load_config
from shared.output import (chapter_filename, clear_interim, format_bible_markdown,
                           interim_enabled, save_interim)


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
