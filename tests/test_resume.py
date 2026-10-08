"""Tests for resume state load/save."""


from shared.llm_client import load_config
from shared.output import save_interim, save_interim_json
from shared.resume import (StateWriteError, has_resume, load_plan,
                           load_state, save_state)
import pytest


def _use_config(tmp_path):
    directory = tmp_path / "out"
    cfg = tmp_path / "config.yaml"
    cfg.write_text(f"output:\n  directory: \"{directory}\"\n  interim: true\n")
    load_config(cfg)
    return directory


def test_load_state_empty_when_no_interim(tmp_path):
    _use_config(tmp_path)
    state = load_state()
    assert state["bible"] is None
    assert state["chapters"] is None
    assert state["drafts"] == {}
    assert state["final"] == {}
    assert not has_resume()


def test_a_run_saved_by_an_older_version_still_loads(tmp_path):
    """Older versions kept the run's state in interim/."""
    _use_config(tmp_path)
    bible = {"title": "Test", "characters": [{"name": "Aria"}], "seed": "x"}
    chapters = [{"number": 1, "title": "One", "summary": "begin"},
                {"number": 2, "title": "Two", "summary": "end"}]
    summaries = {1: "intro", 2: "outro"}
    chronology = {1: {"present": ["Aria"], "events": []},
                  2: {"present": ["Aria"], "events": []}}
    save_interim_json("bible.json", bible)
    save_interim_json("outline.json", chapters)
    save_interim_json("summaries.json", summaries)
    save_interim_json("chronology.json", chronology)
    save_interim("draft_chapter_01.md", "## Chapter 1: One\n\nfirst draft")
    save_interim("lore_chapter_01.md", "# Lore Brief - Chapter 1: One\n\nbrief")
    save_interim("edited_chapter_02.md", "## Chapter 2: Two\n\nfinal form")
    save_interim("edited_chapter_01.md", "## Chapter 1: One\n\nfinal one")

    assert has_resume()
    state = load_state()
    assert state["bible"]["title"] == "Test"
    assert [c["number"] for c in state["chapters"]] == [1, 2]
    assert state["drafts"][1] == "first draft"            # heading stripped
    assert state["research"][1] == "brief"                 # heading stripped
    assert state["final"] == {1: "final one", 2: "final form"}
    assert state["summaries"] == summaries
    assert state["chronology"] == chronology
    # copied to the new layout once, so later saves and resumes use state/
    assert (tmp_path / "out" / "state" / "bible.json").exists()
    assert load_state()["final"] == {1: "final one", 2: "final form"}


def test_strip_heading_leaves_headingless_text_alone():
    from shared.output import strip_heading
    assert strip_heading("First paragraph.\n\nSecond.") == \
        "First paragraph.\n\nSecond."
    assert strip_heading("## Chapter 1: One\n\nBody.") == "Body."


def test_strip_heading_keeps_a_paragraph_right_under_the_heading():
    from shared.output import strip_heading
    assert strip_heading("## Chapter 1: One\nFirst.\n\nSecond.") == \
        "First.\n\nSecond."
    assert strip_heading("## Chapter 1: One") == ""


def test_interim_writes_are_atomic_and_leave_no_temp_files(tmp_path):
    out = _use_config(tmp_path)
    save_interim("draft_chapter_01.md", "one")
    save_interim("draft_chapter_01.md", "two")
    files = sorted(p.name for p in (out / "interim").iterdir())
    assert files == ["draft_chapter_01.md"]
    assert (out / "interim" / "draft_chapter_01.md").read_text() == "two"


def test_state_round_trip(tmp_path):
    out = _use_config(tmp_path)
    save_state("bible", {"title": "T", "seed": "S"})
    save_state("plan", {"chapters": 2, "words_per_chapter": 900})
    save_state("outline", [{"number": 1, "title": "One", "summary": ""}])
    save_state("drafts", {1: "draft one", 2: "draft two"})
    save_state("research", {1: "brief"})
    save_state("summaries", {1: "sum"})
    save_state("chronology", {1: {"present": [], "events": []}})
    save_state("final", {1: "edited one"})
    save_state("unreviewed", [1])
    assert (out / "state" / "drafts.json").exists()
    assert has_resume()
    state = load_state()
    assert state["bible"]["title"] == "T"
    assert state["chapters"][0]["title"] == "One"
    assert state["drafts"] == {1: "draft one", 2: "draft two"}   # int keys
    assert state["research"] == {1: "brief"}
    assert state["summaries"] == {1: "sum"}
    assert state["final"] == {1: "edited one"}
    assert state["unreviewed"] == {1}
    assert load_plan()["words_per_chapter"] == 900


def test_state_is_saved_even_with_interim_output_off(tmp_path):
    directory = tmp_path / "out"
    cfg = tmp_path / "config.yaml"
    cfg.write_text(f"output:\n  directory: \"{directory}\"\n  interim: false\n")
    load_config(cfg)
    save_state("bible", {"title": "T"})
    save_interim("story_bible.md", "human copy")
    assert has_resume()
    assert not (directory / "interim").exists()


def test_a_failed_state_write_raises(tmp_path):
    out = _use_config(tmp_path)
    out.mkdir(parents=True)
    (out / "state").write_text("a file where the directory should be")
    with pytest.raises(StateWriteError, match="drafts"):
        save_state("drafts", {1: "x"})


def test_unknown_state_keys_are_a_bug():
    with pytest.raises(ValueError):
        save_state("draftz", {})


def test_the_new_layout_wins_over_an_old_one(tmp_path):
    _use_config(tmp_path)
    save_interim_json("bible.json", {"title": "OLD"})
    save_state("bible", {"title": "NEW"})
    assert load_state()["bible"]["title"] == "NEW"



def test_concurrent_writes_of_one_file_do_not_collide(tmp_path, monkeypatch):
    """Both writers finish their temp file before either swaps it in: with
    a shared temp name the second swap would find its file already gone."""
    import os
    import threading
    from shared import output
    out = _use_config(tmp_path)
    (out / "interim").mkdir(parents=True)
    both_written = threading.Barrier(2, timeout=5)
    real_replace = os.replace

    def replace(src, dst):
        both_written.wait()
        real_replace(src, dst)

    monkeypatch.setattr(output.os, "replace", replace)
    errors = []

    def write(i):
        try:
            output.atomic_write_text(out / "interim" / "same.md", f"writer {i}")
        except Exception as e:
            errors.append(e)

    threads = [threading.Thread(target=write, args=(i,)) for i in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors
    assert [p.name for p in (out / "interim").iterdir()] == ["same.md"]
