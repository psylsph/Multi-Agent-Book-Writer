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
