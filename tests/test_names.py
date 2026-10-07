"""Character-name handling: seed parsing, canonical names, name lint."""

import json

import pytest

from agents import architect, writer
from shared.consistency import check_chronology, lint_chapter
from shared.context import context, reset_context, update_context
from shared.llm_client import load_config
from shared.llm_utils import extract_seed_characters, same_character
from shared.story_state import canonical_name, merge_states, normalize_state


# -------------------------------------------- fix 4: seed parsing and dupes

def test_bracketed_qualifier_is_not_part_of_the_name():
    seed = """## Characters

- **Carole (from No.12)** - supporting. The village's watcher.
- **Lisa** (the third) - protagonist. A therapist.
"""
    chars = extract_seed_characters(seed)
    assert [c["name"] for c in chars] == ["Carole", "Lisa"]
    assert chars[0]["role"] == "supporting"                  # role still found
    assert chars[0]["description"].endswith("(from No.12)")
    assert chars[1]["description"].endswith("(the third)")


@pytest.mark.parametrize("a,b,same", [
    ("Carole", "Carole (from No.12)", True),
    ("Lisa", "lisa hale", True),
    ("Lisa Hale", "Lisa", True),
    ("Tom", "Tomas", False),
    ("Amy", "Matt", False),
    ("", "Amy", False),
])
def test_same_character(a, b, same):
    assert same_character(a, b) is same


def test_llm_extras_do_not_duplicate_seed_characters(tmp_path, monkeypatch):
    path = tmp_path / "config.yaml"
    path.write_text(f'output:\n  directory: "{tmp_path / "out"}"\n')
    load_config(path)
    reset_context()
    seed = "# Story\n\n## Characters\n\n- **Carole (from No.12)** - supporting. Nosy.\n"
    llm_bible = {"title": "S", "characters": [
        {"name": "Carole", "role": "", "description": "dupe"},
        {"name": "Carole Jones", "role": "", "description": "dupe 2"},
        {"name": "Henry", "role": "", "description": "a genuine extra"}]}
    monkeypatch.setattr(architect, "generate_with_wait",
                        lambda p, **k: json.dumps(llm_bible))
    bible = architect.run_architect(seed)
    assert [c["name"] for c in bible["characters"]] == ["Carole", "Henry"]


# ------------------------------------------ fix 5: canonical names in state

CANON = ["Elizabeth Hale", "Tom", "Lisa"]


@pytest.mark.parametrize("name,expected", [
    ("Elizabeth Hale", "Elizabeth Hale"),
    ("elizabeth hale", "Elizabeth Hale"),
    ("Elizabeth", "Elizabeth Hale"),            # first name
    ("Hale", "Elizabeth Hale"),                 # surname
    ("Tom", "Tom"),
    ("Tom Baker", "Tom"),                       # bible name plus extra words
    ("Liz", "Elizabeth Hale"),                  # known nickname group
    ("Henry", "Henry"),                         # a new character stays new
])
def test_canonical_name(name, expected):
    assert canonical_name(name, CANON) == expected


def test_ambiguous_names_are_left_alone():
    assert canonical_name("Anna", ["Anna Reid", "Anna Cole"]) == "Anna"


def test_state_tracks_deaths_under_the_bibles_spelling():
    """The probe that failed before: a death recorded as 'Elizabeth' and a
    later appearance under the full name must be the SAME person."""
    chron = {
        1: normalize_state(1, "A", {"present": ["Elizabeth", "Tom"],
                                    "events": [{"type": "death",
                                                "who": ["Elizabeth"]}]}, CANON),
        2: normalize_state(2, "B", {"present": ["Elizabeth Hale"],
                                    "events": []}, CANON)}
    cum = merge_states(chron, upto=3)
    assert cum["dead"] == {"Elizabeth Hale": 1}
    assert "Elizabeth Hale" not in cum["alive"]
    # and the dead-character check now fires, by first name too
    findings = check_chronology(3, "Elizabeth walked in and sat down.", cum)
    assert [f["check"] for f in findings] == ["dead_character"]


def test_present_names_are_deduplicated_after_snapping():
    state = normalize_state(1, "A", {"present": ["Elizabeth", "Elizabeth Hale",
                                                 "Tom"]}, CANON)
    assert state["present"] == ["Elizabeth Hale", "Tom"]


def test_first_meeting_check_works_for_multi_word_names():
    cum = {"dead": {}, "met_pairs": {"elizabeth hale+tom": 1}}
    text = "Elizabeth smiled. Tom said, \"Pleased to meet you.\""
    assert [f["check"] for f in check_chronology(2, text, cum)] == \
        ["already_met"]


def test_extractor_prompt_lists_the_bibles_names(tmp_path, monkeypatch):
    path = tmp_path / "config.yaml"
    path.write_text(f'output:\n  directory: "{tmp_path / "out"}"\n')
    load_config(path)
    reset_context()
    update_context("bible", {"characters": [{"name": "Carole (from No.12)"},
                                            {"name": "Elizabeth Hale"}]})
    seen = {}

    def fake(prompt, **kwargs):
        seen.setdefault("prompt", prompt)      # the extraction prompt, not
        return json.dumps({"summary": "s", "present": ["Elizabeth"],  # later
                           "events": [{"type": "death", "who": ["Elizabeth"]}]})

    monkeypatch.setattr(writer, "generate_with_wait", fake)
    _, state = writer._summarize_and_extract(
        1, "One", "Elizabeth Hale was there when she died.")
    assert "Carole, Elizabeth Hale" in seen["prompt"]       # qualifier dropped
    assert state["present"] == ["Elizabeth Hale"]           # snapped
    assert context["bible"]["characters"][0]["name"].startswith("Carole")


# ------------------------------------------------ fix 6: name lint false hits

BIBLE = {"characters": [{"name": "Lisa"}, {"name": "Stuart"},
                        {"name": "Kirsty"}, {"name": "Mark"}]}


def _flagged(text, ignore=()):
    return [f["detail"] for f in lint_chapter(1, text, BIBLE, [], ignore)
            if f["check"] == "name_mismatch"]


def test_plurals_and_suffixed_forms_are_not_misspellings():
    text = "Stuarts mother met Marks brother. Stuartson waved."
    assert _flagged(text) == []


def test_real_misspellings_are_still_flagged():
    assert any("Kirsty" in d for d in _flagged("Kirstie smiled."))
    assert any("Stuart" in d for d in _flagged("Stuat nodded."))
    assert any("Lisa" in d for d in _flagged("Lissa laughed."))


def test_name_lint_ignore_list():
    assert _flagged("Lissa laughed.", ignore=["lissa"]) == []
    assert any("Kirsty" in d for d in _flagged("Kirstie smiled.",
                                               ignore=["lissa"]))


# -------------------------------------------- nicknames and declared aliases

def test_aliases_are_read_from_the_seed():
    from shared.llm_utils import find_aliases
    seed = """## Characters

- **Elizabeth (Liz)** - protagonist. A surgeon.
- **Robert** (aka Bobby) - supporting. Her brother.
- **Margaret** - supporting. Known as Peggy to everyone, a baker.
- **Carole (from No.12)** - supporting. Nosy.
- **Tom** - supporting. Quiet.
  Sometimes called Tommo by the lads.
"""
    chars = {c["name"]: c for c in extract_seed_characters(seed)}
    assert chars["Elizabeth"]["aliases"] == ["Liz"]
    assert chars["Elizabeth"]["description"] == "protagonist. A surgeon."
    assert chars["Robert"]["aliases"] == ["Bobby"]
    assert chars["Margaret"]["aliases"] == ["Peggy"]
    assert chars["Carole"]["aliases"] == []              # not an alias
    assert chars["Carole"]["description"].endswith("(from No.12)")
    assert chars["Tom"]["aliases"] == ["Tommo"]          # wrapped line
    assert find_aliases("", "plain description") == ([], False)


def test_declared_alias_wins_over_everything():
    aliases = {"Elizabeth Hale": ["Bunny"]}
    assert canonical_name("Bunny", ["Elizabeth Hale", "Tom"], aliases) == \
        "Elizabeth Hale"


@pytest.mark.parametrize("name,canon,expected", [
    ("Bob", ["Robert Hale", "Tom"], "Robert Hale"),
    ("Stu", ["Stuart", "Lisa"], "Stuart"),
    ("Carol", ["Carole"], "Carole"),
    ("Kate", ["Catherine"], "Catherine"),
    # a nickname that fits two bible characters stays unresolved
    ("Alex", ["Alexander", "Alexandra"], "Alex"),
    # a nickname for nobody in the bible stays a new character
    ("Bob", ["Lisa", "Tom"], "Bob"),
])
def test_nickname_table(name, canon, expected):
    assert canonical_name(name, canon) == expected


def test_alias_tracked_through_state_and_dead_check():
    aliases = {"Margaret": ["Peggy"]}
    chron = {1: normalize_state(1, "A", {
        "present": ["Peggy"],
        "events": [{"type": "death", "who": ["Peggy"]}]},
        ["Margaret"], aliases)}
    cum = merge_states(chron, upto=2)
    assert cum["dead"] == {"Margaret": 1}
    text = "Peggy laughed and put the kettle on."
    assert check_chronology(2, text, cum) == []                # unaware
    hits = check_chronology(2, text, cum, aliases)             # alias-aware
    assert [f["check"] for f in hits] == ["dead_character"]


def test_bible_shows_aliases_to_the_writer():
    from shared.llm_utils import render_bible
    text = render_bible({"characters": [
        {"name": "Elizabeth", "role": "protagonist", "description": "d",
         "aliases": ["Liz"]}]})
    assert "Elizabeth (also called Liz) (protagonist): d" in text


def test_extractor_prompt_lists_aliases(tmp_path, monkeypatch):
    path = tmp_path / "config.yaml"
    path.write_text(f'output:\n  directory: "{tmp_path / "out"}"\n')
    load_config(path)
    reset_context()
    update_context("bible", {"characters": [
        {"name": "Elizabeth", "aliases": ["Bunny"]}, {"name": "Tom"}]})
    seen = {}

    def fake(prompt, **kwargs):
        seen["prompt"] = prompt
        return json.dumps({"summary": "s", "present": ["Bunny"], "events": []})

    monkeypatch.setattr(writer, "generate_with_wait", fake)
    _, state = writer._summarize_and_extract(1, "One", "Bunny waved at Tom.")
    assert "Elizabeth (also called Bunny), Tom" in seen["prompt"]
    assert state["present"] == ["Elizabeth"]
