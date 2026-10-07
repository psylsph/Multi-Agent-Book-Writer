"""Unit tests for story-state tracking and chronology continuity checks."""

from shared.consistency import check_chronology
from shared.story_state import (merge_states, normalize_state,
                                render_story_facts)


def _state(n, title="T", **kw):
    data = {"summary": f"chapter {n}", "time": "", "location": "",
            "present": [], "events": []}
    data.update(kw)
    return normalize_state(n, title, data)


def test_normalize_state_validates_events():
    state = normalize_state(1, "T", {
        "summary": "s", "time": "dawn", "location": "village",
        "present": ["Aria", "Pip"],
        "events": [
            {"type": "first_meeting", "who": ["Aria", "Pip"]},
            {"type": "nonsense_type", "who": ["X"]},   # dropped
            {"type": "death", "who": [], "detail": ""},  # dropped (no who)
            {"type": "death", "who": ["Maren"], "detail": "drowned"},
        ],
    })
    assert state["time"] == "dawn"
    assert [e["type"] for e in state["events"]] == ["first_meeting", "death"]


def test_merge_states_tracks_meets_deaths_relationships():
    chronology = {
        1: _state(1, present=["Stuart", "Kirsty"],
                  events=[{"type": "relationship_change",
                           "who": ["Stuart", "Kirsty"],
                           "detail": "married 25 years"}]),
        2: _state(2, present=["Stuart", "Lisa"],
                  events=[{"type": "first_meeting",
                           "who": ["Stuart", "Lisa"]}]),
        5: _state(5, present=["Stuart", "Kirsty", "Lisa"],
                  events=[{"type": "relationship_change",
                           "who": ["Stuart", "Kirsty", "Lisa"],
                           "detail": "slept together for the first time"}]),
        7: _state(7, present=["Tom"],
                  events=[{"type": "death", "who": ["Tom"],
                           "detail": "heart attack"}]),
    }
    cum = merge_states(chronology)
    assert cum["dead"] == {"Tom": 7}
    assert "kirsty+lisa" in cum["met_pairs"]      # from the Ch5 threesome
    assert "lisa+stuart" in cum["met_pairs"]      # from Ch2
    rel = cum["relationships"]["kirsty+stuart"]
    assert rel[1] == 5  # latest change wins
    assert "Tom" not in cum["alive"]


def test_merge_states_upto_excludes_current_chapter():
    chronology = {
        1: _state(1, present=["A"], events=[]),
        2: _state(2, present=["B"],
                  events=[{"type": "death", "who": ["B"]}]),
    }
    cum_before_2 = merge_states(chronology, upto=2)  # only Ch1
    assert cum_before_2["dead"] == {}
    cum_all = merge_states(chronology)
    assert cum_all["dead"] == {"B": 2}


def test_render_story_facts_lists_relationships():
    chronology = {
        2: _state(2, time="Thursday", location="clinic",
                  present=["Stuart", "Lisa"],
                  events=[{"type": "first_meeting",
                           "who": ["Stuart", "Lisa"]}]),
        9: _state(9, time="fete night", location="Lisa's",
                  present=["Stuart", "Kirsty", "Lisa"],
                  events=[{"type": "relationship_change",
                           "who": ["Stuart", "Kirsty", "Lisa"],
                           "detail": "first threesome"}]),
    }
    text = render_story_facts(merge_states(chronology))
    assert "Have already met" in text
    assert "Lisa & Stuart" in text
    assert "first threesome" in text
    assert "Relationships (state must not regress)" in text


# ------------------------------------------------------ chronology lint

DEAD_RESURRECTION = '''Kirsty walked to the green. "Lovely morning," said Tom,
waving from the harbour wall. Kirsty laughed.'''
DEAD_MENTION_OK = '''Kirsty walked past the harbour wall and thought of Tom,
who had died there last winter. "I miss him," she said.'''


def test_dead_character_acting_alive_is_flagged():
    chronology = {3: _state(3, present=["Tom"],
                            events=[{"type": "death", "who": ["Tom"]}])}
    findings = check_chronology(4, DEAD_RESURRECTION,
                                merge_states(chronology, upto=4))
    assert any(f["check"] == "dead_character" and "Tom" in f["detail"]
               for f in findings)


def test_dead_character_memory_is_not_flagged():
    chronology = {3: _state(3, present=["Tom"],
                            events=[{"type": "death", "who": ["Tom"]}])}
    findings = check_chronology(4, DEAD_MENTION_OK,
                                merge_states(chronology, upto=4))
    assert not any(f["check"] == "dead_character" for f in findings)


def test_death_chapter_itself_is_not_flagged():
    chronology = {3: _state(3, present=["Tom"],
                            events=[{"type": "death", "who": ["Tom"]}])}
    # chapter 3 contains the death itself; Tom still acts earlier in it
    findings = check_chronology(3, DEAD_RESURRECTION,
                                merge_states(chronology, upto=3))
    assert not any(f["check"] == "dead_character" for f in findings)


def test_strangers_language_between_already_met_pair():
    chronology = {2: _state(2, present=["Stuart", "Lisa"],
                            events=[{"type": "first_meeting",
                                     "who": ["Stuart", "Lisa"]}])}
    text = ('Lisa turned to Stuart. "Pleased to meet you," she said, '
            'extending a hand.')
    findings = check_chronology(5, text, merge_states(chronology, upto=5))
    assert any(f["check"] == "already_met" and "Lisa" in f["detail"]
               for f in findings)


def test_strangers_language_with_only_one_of_pair_present_not_flagged():
    chronology = {2: _state(2, present=["Stuart", "Lisa"],
                            events=[{"type": "first_meeting",
                                     "who": ["Stuart", "Lisa"]}])}
    # Sophie (not Lisa) meets Stuart for real - that's fine
    text = ('Sophie turned to Stuart. "Pleased to meet you," she said.')
    findings = check_chronology(5, text, merge_states(chronology, upto=5))
    assert not any(f["check"] == "already_met" for f in findings)


# ------------------------------------------------------ windowed recap

def test_story_so_far_windows_older_chapters():
    from shared.story_state import render_story_so_far
    summaries = {n: f"Event {n} happens. Then more detail about {n}."
                 for n in range(1, 7)}
    text = render_story_so_far(summaries, before=6, window=2)
    assert "Chapter 1: Event 1 happens." in text
    assert "more detail about 1" not in text        # old: first sentence only
    assert "more detail about 4" in text            # in the window: full
    assert "more detail about 3" not in text
    assert "Chapter 6" not in text                  # never the current chapter


def test_story_so_far_empty_for_first_chapter():
    from shared.story_state import render_story_so_far
    assert render_story_so_far({1: "x"}, before=1) == ""


# ------------------------------------------- loosely typed model output

import pytest  # noqa: E402

from shared.story_state import as_names  # noqa: E402


@pytest.mark.parametrize("value,expected", [
    (["Tom", "Liz"], ["Tom", "Liz"]),
    ("Tom Baker", ["Tom Baker"]),                      # a bare string
    ("Tom and Liz", ["Tom", "Liz"]),
    ("Tom, Liz; Marcus & Anna + Sam", ["Tom", "Liz", "Marcus", "Anna", "Sam"]),
    ("Sandra Rowland", ["Sandra Rowland"]),            # 'and' inside a word
    ([{"name": "Tom"}, {"role": "no name"}, " Liz ", "", None, ["x"]],
     ["Tom", "Liz"]),
    ({"name": "Tom"}, ["Tom"]),
    (None, []), (42, []), (True, []), ("", []),
])
def test_as_names(value, expected):
    assert as_names(value) == expected


def test_a_string_for_who_does_not_become_single_letter_characters():
    """The bug the extractor test found: "who": "Tom Baker" was iterated per
    character, creating 'T', 'o', 'm', ... as characters in the story state."""
    state = normalize_state(1, "A", {
        "present": "Elizabeth Hale, Tom Baker",
        "events": [{"type": "injury", "who": "Tom Baker"},
                   {"type": "death", "who": "Marcus Reed and Liz"}]},
        ["Elizabeth Hale", "Tom Baker", "Marcus Reed"],
        {"Elizabeth Hale": ["Liz"]})
    assert state["present"] == ["Elizabeth Hale", "Tom Baker"]
    assert state["events"][0]["who"] == ["Tom Baker"]
    assert state["events"][1]["who"] == ["Marcus Reed", "Elizabeth Hale"]
    cumulative = merge_states({1: state})
    assert all(len(name) > 1 for name in cumulative["dead"])
    assert cumulative["dead"] == {"Marcus Reed": 1, "Elizabeth Hale": 1}


def test_odd_container_types_are_ignored_not_crashed_on():
    state = normalize_state(1, "A", {"present": 5, "events": "none"})
    assert state["present"] == [] and state["events"] == []
    state = normalize_state(1, "A", {"events": {"type": "death"}})
    assert state["events"] == []
    state = normalize_state(1, "A", {"events": [{"type": "death", "who": 7},
                                                "junk", None]})
    assert state["events"] == []        # a number is not a person: dropped
