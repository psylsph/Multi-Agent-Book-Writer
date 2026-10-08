"""Tests for resume state load/save."""


from shared.llm_client import load_config
from shared.output import save_interim, save_interim_json
from shared.resume import has_resume, load_state


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


def test_save_and_load_round_trip(tmp_path):
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
