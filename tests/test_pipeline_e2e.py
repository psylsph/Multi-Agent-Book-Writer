"""End-to-end: main.main() against an in-process fake LLM (no network)."""

import json
import sys

import pytest

import main as pipeline
from shared import llm_client, runlog
from tests.fake_llm import SEED, FakeLLM, marker


@pytest.fixture(autouse=True)
def _clean_global_state():
    llm_client.reset_stats()
    yield
    runlog.stop_log()               # never leave sys.stdout wrapped


def write_config(tmp_path, book="", llm="", agents="", output="", web=""):
    path = tmp_path / "config.yaml"
    path.write_text(
        f"book:\n  words_per_chapter: 100\n  revision_rounds: 1\n{book}"
        'llm:\n  base_url: "http://fake:1"\n  model: fake\n  retries: 0\n'
        f"  endpoint_wait: 0\n  timeout: 5\n{llm}"
        f"agents:\n  researcher:\n    enabled: true\n{agents}"
        f'output:\n  directory: "{tmp_path / "out"}"\n  filename: book.md\n'
        f"{output}{web}")
    return path


def run(monkeypatch, config, *args, seed=SEED):
    """Run main() with argv; returns the exit code."""
    argv = ["main.py", "--config", str(config), *args]
    if "--demo" not in args and "--seed" not in args and "--prompt" not in args:
        argv += ["--prompt", seed]
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(SystemExit) as exc:
        pipeline.main()
    code = exc.value.code
    return code if isinstance(code, int) else (0 if code is None else 1)


def out(tmp_path):
    return tmp_path / "out"


# ------------------------------------------------------------------ fresh run

def test_a_fresh_run_writes_the_whole_book(tmp_path, monkeypatch, capsys):
    fake = FakeLLM().install(monkeypatch)
    assert run(monkeypatch, write_config(tmp_path)) == 0
    book = (out(tmp_path) / "book.md").read_text()
    assert book.startswith("# The Marsh Light")
    assert "## Chapter 1: The Letter" in book and "## Chapter 2: The Ferry" in book
    assert marker(1) in book and marker(2) in book
    assert sorted(p.name for p in (out(tmp_path) / "chapters").iterdir()) == \
        ["chapter_01.md", "chapter_02.md"]
    state = {p.name for p in (out(tmp_path) / "state").iterdir()}
    assert {"bible.json", "outline.json", "research.json", "drafts.json",
            "summaries.json", "chronology.json", "final.json"} <= state
    interim = {p.name for p in (out(tmp_path) / "interim").iterdir()}
    assert {"lore_chapter_01.md",
            "draft_chapter_02.md", "edited_chapter_02.md",
            "review_chapter_01.md", "diff_chapter_01.md",
            "run_stats.md", "run_stats.json"} <= interim
    # the seed carries an outline, so nothing was asked of the planner
    assert fake.kinds == {"seed_review": 1, "architect": 1, "researcher": 2,
                          "writer": 2, "extractor": 2, "reviewer": 2,
                          "polisher": 2}
    assert "PIPELINE COMPLETE" in capsys.readouterr().out


def test_the_run_is_logged_to_a_file(tmp_path, monkeypatch):
    FakeLLM().install(monkeypatch)
    run(monkeypatch, write_config(tmp_path))
    (log,) = list((out(tmp_path) / "logs").glob("run-*.log"))
    text = log.read_text()
    assert "PIPELINE COMPLETE" in text and "Book written" in text
    assert sys.stdout is not None and not isinstance(
        sys.stdout, runlog._Tee)                    # streams were restored


def test_logging_can_be_switched_off(tmp_path, monkeypatch):
    FakeLLM().install(monkeypatch)
    run(monkeypatch, write_config(tmp_path, output="  log: false\n"))
    assert not (out(tmp_path) / "logs").exists()


def test_unknown_config_keys_are_warned_about(tmp_path, monkeypatch, capsys):
    FakeLLM().install(monkeypatch)
    run(monkeypatch, write_config(tmp_path, book="  revision_round: 3\n"))
    text = capsys.readouterr().out
    assert "[CONFIG] unknown key 'book.revision_round'" in text
    assert "did you mean 'revision_rounds'?" in text


def test_chapter_count_is_suggested_when_the_seed_has_no_outline(
        tmp_path, monkeypatch):
    fake = FakeLLM().install(monkeypatch)
    cfg = write_config(tmp_path, book="  min_chapters: 1\n  seed_review: off\n")
    assert run(monkeypatch, cfg, seed="# Story\n\nA premise only.") == 0
    assert fake.kinds["suggest"] == 1 and fake.kinds["planner"] == 1
    assert fake.kinds["writer"] == 2
    assert "seed_review" not in fake.kinds


def test_the_seed_review_sizes_a_book_without_an_outline(tmp_path,
                                                         monkeypatch):
    fake = FakeLLM().install(monkeypatch)
    cfg = write_config(tmp_path, book="  min_chapters: 1\n")
    assert run(monkeypatch, cfg, seed="# Story\n\nA premise only.") == 0
    assert fake.kinds["seed_review"] == 1 and "suggest" not in fake.kinds
    assert fake.kinds["writer"] == 2             # its recommendation: 2
    plan = json.loads((out(tmp_path) / "state" / "plan.json").read_text())
    assert plan["chapters"] == 2 and plan["words_per_chapter"] == 100


def test_plan_only_stops_after_the_outline_and_a_rerun_continues(
        tmp_path, monkeypatch, capsys):
    fake = FakeLLM().install(monkeypatch)
    cfg = write_config(tmp_path)
    assert run(monkeypatch, cfg, "--plan-only") == 0
    assert "writer" not in fake.kinds
    assert (out(tmp_path) / "interim" / "outline.md").exists()
    assert "Book written" not in capsys.readouterr().out
    assert run(monkeypatch, cfg) == 0
    assert fake.kinds["seed_review"] == 1        # not asked again on resume
    assert fake.kinds["architect"] == 1 and fake.kinds["writer"] == 2


def test_the_demo_seed_runs(tmp_path, monkeypatch):
    fake = FakeLLM().install(monkeypatch)
    assert run(monkeypatch, write_config(tmp_path), "--demo") == 0
    written = list((out(tmp_path) / "chapters").glob("chapter_*.md"))
    assert len(written) == fake.kinds["writer"] >= 2


def test_a_missing_model_stops_the_run_before_any_work(tmp_path, monkeypatch):
    fake = FakeLLM(models=["something-else"]).install(monkeypatch)
    assert run(monkeypatch, write_config(tmp_path)) != 0
    assert fake.chat_calls == 0


# ------------------------------------------------------------ review as you go

def test_review_as_you_go_alternates_writing_and_editing(tmp_path,
                                                         monkeypatch):
    fake = FakeLLM().install(monkeypatch)
    cfg = write_config(tmp_path, book="  review_as_you_go: true\n")
    assert run(monkeypatch, cfg) == 0
    order = [c["kind"] for c in fake.calls
             if c["kind"] in ("writer", "polisher")]
    assert order == ["writer", "polisher", "writer", "polisher"]
    assert fake.kinds["extractor"] == 4          # 2 drafts + 2 refreshes
    assert "## Chapter 2: The Ferry" in (out(tmp_path) / "book.md").read_text()


# ------------------------------------------------------------ resume / safety

def test_an_interrupted_run_resumes_without_redoing_finished_work(
        tmp_path, monkeypatch, capsys):
    cfg = write_config(tmp_path)
    # architect, 2 briefs, chapter 1 + its facts, then the server drops
    first = FakeLLM(fail_after=5).install(monkeypatch)
    assert run(monkeypatch, cfg) == 1
    assert "endpoint went down" in capsys.readouterr().out
    assert first.kinds["writer"] == 1 and (
        out(tmp_path) / "chapters" / "chapter_01.md").exists()

    second = FakeLLM().install(monkeypatch)          # the server is back
    assert run(monkeypatch, cfg) == 0
    assert "architect" not in second.kinds and "researcher" not in second.kinds
    assert second.kinds["writer"] == 1               # only chapter 2
    assert "## Chapter 2: The Ferry" in (out(tmp_path) / "book.md").read_text()


def test_a_different_seed_is_refused_not_silently_reused(tmp_path,
                                                         monkeypatch, capsys):
    cfg = write_config(tmp_path)
    FakeLLM().install(monkeypatch)
    assert run(monkeypatch, cfg) == 0
    fake = FakeLLM().install(monkeypatch)
    with pytest.raises(SystemExit):
        monkeypatch.setattr(sys, "argv", ["main.py", "--config", str(cfg),
                                          "--prompt", "A different book."])
        pipeline.main()
    assert fake.chat_calls == 0
    assert (out(tmp_path) / "book.md").read_text().startswith("# The Marsh")


def test_no_resume_archives_the_old_run_then_starts_over(tmp_path,
                                                         monkeypatch):
    cfg = write_config(tmp_path)
    FakeLLM().install(monkeypatch)
    run(monkeypatch, cfg)
    fake = FakeLLM().install(monkeypatch)
    assert run(monkeypatch, cfg, "--no-resume") == 0
    (archive,) = list((out(tmp_path) / "archive").iterdir())
    assert (archive / "chapters" / "chapter_01.md").exists()
    assert fake.kinds["architect"] == 1 and fake.kinds["writer"] == 2


def test_a_chapter_the_model_refuses_ends_the_run_incomplete(
        tmp_path, monkeypatch, capsys):
    fake = FakeLLM(refuse_chapter=2).install(monkeypatch)
    assert run(monkeypatch, write_config(tmp_path)) == 1
    text = capsys.readouterr().out
    assert "PIPELINE INCOMPLETE" in text and "Chapters not finished: 2" in text
    assert fake.kinds["writer"] == 3                 # ch 1, then ch 2 twice
    assert not (out(tmp_path) / "chapters" / "chapter_02.md").exists()


# --------------------------------------------------------------- observability

def test_run_statistics_are_printed_and_saved(tmp_path, monkeypatch, capsys):
    fake = FakeLLM().install(monkeypatch)
    run(monkeypatch, write_config(tmp_path))
    data = json.loads((out(tmp_path) / "interim" / "run_stats.json").read_text())
    assert data["total"]["calls"] == len(fake.calls)
    assert data["total"]["prompt_tokens"] > 0
    assert set(data["agents"]) >= {"writer", "reviewer", "editor", "extractor"}
    assert [p["phase"] for p in data["phases"]] == [
        "seed review", "architect", "planner", "researcher", "writer",
        "editor"]
    shown = capsys.readouterr().out
    assert "# Run statistics" in shown and "| writer | 2 |" in shown
    assert "unreported" not in (out(tmp_path) / "interim" /
                                "run_stats.md").read_text()


def test_servers_that_hide_token_usage_are_reported_honestly(tmp_path,
                                                             monkeypatch):
    FakeLLM(usage=False).install(monkeypatch)
    run(monkeypatch, write_config(tmp_path))
    text = (out(tmp_path) / "interim" / "run_stats.md").read_text()
    assert "did not report token usage" in text
    data = json.loads((out(tmp_path) / "interim" / "run_stats.json").read_text())
    assert data["total"]["prompt_tokens"] == 0     # nothing is invented


def test_streaming_run_completes_and_still_counts_tokens(tmp_path,
                                                         monkeypatch):
    fake = FakeLLM(stream=True).install(monkeypatch)
    cfg = write_config(tmp_path, llm="  stream: true\n")
    assert run(monkeypatch, cfg) == 0
    assert all(c["payload"]["stream"] is True for c in fake.calls)
    assert marker(2) in (out(tmp_path) / "book.md").read_text()
    data = json.loads((out(tmp_path) / "interim" / "run_stats.json").read_text())
    assert data["total"]["completion_tokens"] > 0


def test_context_window_guard_warns_once_per_agent(tmp_path, monkeypatch,
                                                   capsys):
    FakeLLM().install(monkeypatch)
    run(monkeypatch, write_config(tmp_path, llm="  context_window: 50\n"))
    text = capsys.readouterr().out
    assert text.count("Warning: the writer prompt is about") == 1
    assert "close to the 50-token context window" in text


def test_json_mode_is_requested_only_for_object_replies(tmp_path, monkeypatch):
    fake = FakeLLM().install(monkeypatch)
    run(monkeypatch, write_config(tmp_path, llm="  json_mode: true\n"),
        seed="# Story\n\nA premise only.")
    by_kind = {}
    for c in fake.calls:
        by_kind.setdefault(c["kind"], set()).add(
            "response_format" in c["payload"])
    assert by_kind["architect"] == by_kind["extractor"] == {True}
    assert by_kind["reviewer"] == by_kind["seed_review"] == {True}
    assert by_kind["writer"] == by_kind["planner"] == {False}  # arrays/prose


def test_a_server_without_json_mode_is_handled_gracefully(tmp_path,
                                                          monkeypatch, capsys):
    fake = FakeLLM(reject_json_mode=True).install(monkeypatch)
    assert run(monkeypatch, write_config(tmp_path,
                                         llm="  json_mode: true\n")) == 0
    assert capsys.readouterr().out.count("rejected response_format") == 1
    assert not any("response_format" in c["payload"] for c in fake.calls)
