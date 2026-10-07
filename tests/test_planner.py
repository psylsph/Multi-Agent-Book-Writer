"""Planner tests: LLM-suggested chapter count and outline assembly (no LLM)."""

import json

import pytest

from agents import planner
from shared.context import reset_context, update_context
from shared.llm_client import load_config

SEED_NO_OUTLINE = "# A Story\n\nA premise with no outline section.\n"


@pytest.fixture
def cfg(tmp_path):
    def _load(book=""):
        path = tmp_path / "config.yaml"
        path.write_text(
            f'output:\n  directory: "{tmp_path / "out"}"\nbook:\n{book}')
        load_config(path)
    _load()
    reset_context()
    return _load


def _stub_llm(monkeypatch, *replies):
    """Return queued replies in order; record the prompts."""
    prompts, queue = [], list(replies)

    def fake(prompt, **kwargs):
        prompts.append(prompt)
        return queue.pop(0)

    monkeypatch.setattr(planner, "generate_with_wait", fake)
    return prompts


def _outline(n):
    return json.dumps([{"title": f"T{i}", "summary": f"S{i}"}
                       for i in range(1, n + 1)])


def test_suggest_chapter_count_uses_llm_answer(cfg, monkeypatch):
    prompts = _stub_llm(monkeypatch, '{"chapters": 8, "reason": "ok"}')
    assert planner.suggest_chapter_count(SEED_NO_OUTLINE, {}) == 8
    assert "A premise with no outline" in prompts[0]


@pytest.mark.parametrize("answer,expected", [(1, 3), (99, 30)])
def test_suggest_chapter_count_is_clamped(cfg, monkeypatch, answer, expected):
    _stub_llm(monkeypatch, json.dumps({"chapters": answer}))
    assert planner.suggest_chapter_count(SEED_NO_OUTLINE, {}) == expected


def test_suggest_chapter_count_falls_back_on_garbage(cfg, monkeypatch):
    cfg("  num_chapters: 6\n")
    _stub_llm(monkeypatch, "no idea, sorry")
    assert planner.suggest_chapter_count(SEED_NO_OUTLINE, {}) == 6


def test_run_planner_uses_suggestion_when_no_outline(cfg, monkeypatch):
    update_context("seed", SEED_NO_OUTLINE)
    _stub_llm(monkeypatch, '{"chapters": 4, "reason": "x"}', _outline(4))
    chapters = planner.run_planner()
    assert [c["number"] for c in chapters] == [1, 2, 3, 4]


def test_cli_chapter_count_skips_suggestion(cfg, monkeypatch):
    update_context("seed", SEED_NO_OUTLINE)
    prompts = _stub_llm(monkeypatch, _outline(2))  # only the outline call
    assert len(planner.run_planner(num_chapters=2)) == 2
    assert len(prompts) == 1


def test_auto_chapters_off_uses_config_count(cfg, monkeypatch):
    cfg("  auto_chapters: false\n  num_chapters: 3\n")
    update_context("seed", SEED_NO_OUTLINE)
    _stub_llm(monkeypatch, _outline(3))
    assert len(planner.run_planner()) == 3


def test_resume_reuses_saved_outline(cfg, monkeypatch):
    saved = [{"number": 1, "title": "Old", "summary": "", "part": ""}]
    update_context("chapters", saved)
    update_context("seed", SEED_NO_OUTLINE)
    prompts = _stub_llm(monkeypatch)  # any LLM call would pop an empty queue
    assert planner.run_planner() == saved
    assert prompts == []


def test_expansion_keeps_author_chapters_verbatim(cfg, monkeypatch):
    update_context("seed", "# Outline\nChapter 1: A - first\nChapter 2: B - second\n")
    # model rewrites the fixed chapters and adds one more
    reply = json.dumps([{"title": "Changed", "summary": "x"},
                        {"title": "Changed2", "summary": "x"},
                        {"title": "C", "summary": "third"}])
    _stub_llm(monkeypatch, reply)
    chapters = planner.run_planner(num_chapters=3)
    assert [c["title"] for c in chapters] == ["A", "B", "C"]
    assert [c["number"] for c in chapters] == [1, 2, 3]
