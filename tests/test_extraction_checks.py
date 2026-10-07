"""Cross-checks on the extractor's output: grounding names, verifying deaths."""

import json

import pytest

from agents import writer
from shared import extraction_checks as xc
from shared.context import context, reset_context, update_context
from shared.llm_client import load_config
from shared.story_state import merge_states, normalize_state

# ------------------------------------------------------------ pure helpers


def test_name_variants():
    v = xc.name_variants("Elizabeth Hale (the surveyor)", ["Bunny"])
    assert {"elizabeth hale", "elizabeth", "hale", "bunny", "liz",
            "lizzie", "beth"} <= v
    assert "the surveyor" not in v and "surveyor" not in v
    assert "will" not in xc.name_variants("William")        # an ordinary word
    assert "bill" not in xc.name_variants("William")
    assert xc.name_variants("Al") == {"al"}                  # short: full only


@pytest.mark.parametrize("text,hit", [
    ("Tom waved.", True), ("tom waved", True), ("the tomato", False),
    ("customs of the marsh", False), ("Tom's boat", True),
    ("Liz laughed", True)])
def test_mentions_is_whole_word_and_case_insensitive(text, hit):
    variants = xc.name_variants("Tom Baker") | xc.name_variants("Elizabeth")
    assert xc.mentions(text, variants) is hit


def _state(present, events):
    return {"number": 1, "title": "T", "summary": "s", "time": "", "location": "",
            "present": present,
            "events": [{"type": t, "who": w, "detail": ""} for t, w in events]}


def test_names_the_text_never_mentions_are_dropped():
    variants = {"Elizabeth Hale": xc.name_variants("Elizabeth Hale"),
                "Tom Baker": xc.name_variants("Tom Baker"),
                "Marcus Reed": xc.name_variants("Marcus Reed")}
    text = "Liz and Tom crossed the marsh. Nobody else was there."
    state = _state(["Elizabeth Hale", "Tom Baker", "Marcus Reed", "Zed Quinn"],
                   [("first_meeting", ["Elizabeth Hale", "Marcus Reed"]),
                    ("death", ["Marcus Reed"]),
                    ("injury", ["Tom Baker"])])
    out, dropped = xc.ground_state(state, text, variants)
    assert out["present"] == ["Elizabeth Hale", "Tom Baker"]   # nickname 'Liz'
    # the meeting lost Marcus and is no longer a meeting; his death has nobody
    assert [(e["type"], e["who"]) for e in out["events"]] == [
        ("injury", ["Tom Baker"])]
    assert sorted(dropped) == ["Marcus Reed", "Zed Quinn"]
    assert state["present"][-1] == "Zed Quinn"                 # input untouched


def test_death_hints_find_death_language_next_to_a_name():
    variants = {"Marcus Reed": xc.name_variants("Marcus Reed"),
                "Tom Baker": xc.name_variants("Tom Baker")}
    filler = "The tide came in and the lamps burned low. " * 60   # ~2.5k chars
    text = (f"Marcus drowned in the storm. {filler}"
            f"Tom said the harbour was dead quiet that night.")
    hints = xc.death_hints(text, variants)
    assert set(hints) == {"Marcus Reed", "Tom Baker"}
    assert "Marcus drowned" in hints["Marcus Reed"][0]
    assert "dead quiet" in hints["Tom Baker"][0]
    assert "dead quiet" not in hints["Marcus Reed"][0]         # too far away
    assert "Marcus" not in hints["Tom Baker"][0]


def test_a_death_written_with_a_pronoun_still_names_the_character():
    """'He's gone' sits in a paragraph that names only Elizabeth; Marcus is
    named a few sentences earlier. Both must be offered to the model."""
    variants = {"Marcus Reed": xc.name_variants("Marcus Reed"),
                "Elizabeth Hale": xc.name_variants("Elizabeth Hale")}
    text = ("Marcus Reed had not woken since the storm. The doctor sat with "
            "him all night and said very little. At dawn she covered his face."
            "\n\n\"He's gone,\" she told Elizabeth quietly.")
    assert set(xc.death_hints(text, variants)) == {"Marcus Reed",
                                                   "Elizabeth Hale"}


def test_death_hints_need_both_the_word_and_the_name():
    variants = {"Marcus Reed": xc.name_variants("Marcus Reed")}
    assert xc.death_hints("Marcus waved from the jetty.", variants) == {}
    assert xc.death_hints("A gull died on the jetty.", variants) == {}
    assert xc.death_hints("", variants) == {}


def test_overlapping_hints_merge_and_are_capped():
    variants = {"Marcus Reed": xc.name_variants("Marcus Reed")}
    close = "Marcus died. They buried Marcus. The funeral for Marcus was short."
    assert len(xc.death_hints(close, variants)["Marcus Reed"]) == 1
    far = ("Marcus died. " + "x " * 600) * 5
    assert len(xc.death_hints(far, variants, limit=3)["Marcus Reed"]) == 3


# --------------------------------------------- the writer's use of them

CANON = ["Elizabeth Hale", "Tom Baker", "Marcus Reed"]
QUESTION = "Question: which of the candidates die"
ALIVE_QUESTION = "Question: which of the candidates are still alive"


def _setup(tmp_path, monkeypatch, extraction, answer, book="", alive=None):
    """Stub the model: `extraction` for the main call, `answer` (dict or raw
    string) for the focused death question and `alive` for the cross-question
    (default: nobody is alive, so a confirmed death is never contradicted).
    Returns the call log."""
    path = tmp_path / "config.yaml"
    path.write_text(f'output:\n  directory: "{tmp_path / "out"}"\n'
                    + (f"book:\n{book}" if book else ""))
    load_config(path)
    reset_context()
    update_context("bible", {"characters": [
        {"name": "Elizabeth Hale", "aliases": ["Liz"]},
        {"name": "Tom Baker"}, {"name": "Marcus Reed"}]})
    calls = []

    def fake(prompt, **kwargs):
        calls.append(prompt)
        if ALIVE_QUESTION in prompt:
            reply = {"alive": []} if alive is None else alive
            return reply if isinstance(reply, str) else json.dumps(reply)
        if QUESTION in prompt:
            return answer if isinstance(answer, str) else json.dumps(answer)
        return json.dumps(extraction)

    monkeypatch.setattr(writer, "generate_with_wait", fake)
    return calls


def _extract(**extraction):
    base = {"summary": "S.", "time": "", "location": "", "present": [],
            "events": []}
    base.update(extraction)
    return base


def _questions(calls):
    return [c for c in calls if QUESTION in c]


DEATH_TEXT = ("Tom came back at first light with a grey face. \"It's Marcus,\" "
              "he said. \"He drowned in the storm. He's dead, Liz.\" Liz sat down.")
DIES = {"type": "death", "who": ["Marcus Reed"]}


def _run(text, number=4):
    return writer._summarize_and_extract(number, "Drowned", text)[1]


def test_a_claimed_death_the_text_supports_is_kept(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch, _extract(
        present=["Tom Baker", "Liz"], events=[DIES]),
        {"dead": ["Marcus Reed"]})
    state = _run(DEATH_TEXT)
    assert [(e["type"], e["who"]) for e in state["events"]] == [
        ("death", ["Marcus Reed"])]
    assert state["present"] == ["Tom Baker", "Elizabeth Hale"]


def test_everyone_near_the_death_costs_one_call_not_one_each(tmp_path,
                                                             monkeypatch):
    """Tom, Liz and Marcus all sit next to death words in this text; the
    model is still asked ONCE, about all three."""
    calls = _setup(tmp_path, monkeypatch, _extract(events=[DIES]),
                   {"dead": ["Marcus Reed"]})
    _run(DEATH_TEXT)
    (question,) = _questions(calls)
    listed = question.split("Candidates: ")[1].split("\n")[0].split(", ")
    assert sorted(listed) == ["Elizabeth Hale", "Marcus Reed", "Tom Baker"]
    assert "drowned" in question


def test_a_claimed_death_the_text_does_not_support_is_dropped(
        tmp_path, monkeypatch, capsys):
    _setup(tmp_path, monkeypatch, _extract(
        present=["Tom Baker", "Liz"],
        events=[{"type": "death", "who": ["Tom Baker"]}]), {"dead": []})
    state = _run("Tom went through the ice and nearly drowned. Liz hauled him "
                 "out. He coughed up water and lived.")
    assert state["events"] == []
    assert "does not confirm that Tom Baker dies" in capsys.readouterr().out


def test_a_death_the_extractor_missed_is_found(tmp_path, monkeypatch, capsys):
    _setup(tmp_path, monkeypatch, _extract(present=["Tom Baker", "Liz"]),
           {"dead": ["Marcus Reed"]})
    state = _run(DEATH_TEXT)
    assert [(e["type"], e["who"]) for e in state["events"]] == [
        ("death", ["Marcus Reed"])]
    assert "found a death the extractor missed: Marcus Reed" in \
        capsys.readouterr().out


def test_a_first_name_answer_is_matched_to_the_candidate(tmp_path,
                                                         monkeypatch):
    _setup(tmp_path, monkeypatch, _extract(), {"dead": ["marcus"]})
    assert [e["who"] for e in _run(DEATH_TEXT)["events"]] == [["Marcus Reed"]]


def test_death_language_where_nobody_dies_adds_nothing(tmp_path, monkeypatch):
    text = ("Tom said the marsh was dead quiet. Liz laughed and said the "
            "funeral parlour joke was in poor taste.")
    calls = _setup(tmp_path, monkeypatch, _extract(present=["Tom Baker"]),
                   {"dead": []})
    assert _run(text)["events"] == []
    assert len(_questions(calls)) == 1                     # it did ask, once


@pytest.mark.parametrize("reply", ["no idea", '{"dead": "Marcus Reed"}',
                                   "[]", '{"dies": true}'])
def test_an_unreadable_answer_leaves_the_extractors_decision(
        tmp_path, monkeypatch, reply):
    _setup(tmp_path, monkeypatch, _extract(events=[DIES]), reply)
    assert [e["who"] for e in _run(DEATH_TEXT)["events"]] == [["Marcus Reed"]]
    # ...and a missed death is never invented from a garbled answer
    _setup(tmp_path, monkeypatch, _extract(present=["Tom Baker"]), reply)
    assert _run(DEATH_TEXT)["events"] == []


def test_a_claimed_death_with_no_death_language_gets_the_whole_chapter(
        tmp_path, monkeypatch):
    text = "Marcus did not wake again. Tom closed the shutters and sat down."
    calls = _setup(tmp_path, monkeypatch, _extract(events=[DIES]),
                   {"dead": ["Marcus Reed"]})
    state = _run(text)
    assert [e["who"] for e in state["events"]] == [["Marcus Reed"]]
    assert text in _questions(calls)[0]


def test_a_character_who_died_earlier_is_not_killed_again(
        tmp_path, monkeypatch):
    calls = _setup(tmp_path, monkeypatch, _extract(
        present=["Tom Baker"], events=[DIES]), {"dead": []})
    update_context("chronology", {
        2: normalize_state(2, "A", {"events": [DIES]}, CANON)})
    funeral = "They buried Marcus at the chapel. Tom carried the coffin."
    state = writer._summarize_and_extract(5, "Funeral", funeral)[1]
    assert state["events"] == []                       # not re-killed
    for question in _questions(calls):                 # Marcus is not asked about
        assert "Marcus Reed" not in question.split("Candidates:")[1].split("\n")[0]


def test_the_first_death_chapter_stands(tmp_path):
    chron = {n: normalize_state(n, "T", {"events": [DIES]}, CANON)
             for n in (2, 5)}
    assert merge_states(chron)["dead"] == {"Marcus Reed": 2}


def test_invented_names_and_the_events_they_broke_are_dropped(
        tmp_path, monkeypatch, capsys):
    _setup(tmp_path, monkeypatch, _extract(
        present=["Tom Baker", "Marcus Reed", "Zed Quinn"],
        events=[{"type": "first_meeting", "who": ["Tom Baker", "Zed Quinn"]},
                {"type": "injury", "who": ["Tom Baker"]}]), {})
    state = _run("Tom crossed the marsh alone and cut his hand on the rope.")
    assert state["present"] == ["Tom Baker"]
    assert [e["type"] for e in state["events"]] == ["injury"]  # meeting needs 2
    message = capsys.readouterr().out
    assert "never appear in the text" in message
    assert "Marcus Reed" in message and "Zed Quinn" in message


def test_a_quiet_chapter_costs_no_extra_calls(tmp_path, monkeypatch):
    calls = _setup(tmp_path, monkeypatch, _extract(
        present=["Tom Baker", "Liz"]), {})
    _run("Tom and Liz mended the winch rope all morning and argued about knots.")
    assert len(calls) == 1


def test_the_checks_can_be_switched_off(tmp_path, monkeypatch):
    calls = _setup(tmp_path, monkeypatch, _extract(
        present=["Tom Baker", "Zed Quinn"], events=[DIES]),
        {"dead": []}, book="  extraction_checks: false\n")
    state = _run(DEATH_TEXT)
    assert "Zed Quinn" in state["present"]               # nothing was checked
    assert state["events"][0]["who"] == ["Marcus Reed"]
    assert len(calls) == 1


def test_an_outage_during_the_check_still_aborts(tmp_path, monkeypatch):
    from shared.llm_client import EndpointUnavailable
    _setup(tmp_path, monkeypatch, _extract(events=[DIES]), {})

    def flaky(prompt, **kwargs):
        if QUESTION in prompt:
            raise EndpointUnavailable("down")
        return json.dumps(_extract(events=[DIES]))

    monkeypatch.setattr(writer, "generate_with_wait", flaky)
    with pytest.raises(EndpointUnavailable):
        _run(DEATH_TEXT)
    assert context["bible"]


@pytest.mark.parametrize("etype,who,kept", [
    ("first_meeting", ["Tom Baker", "Ghost"], False),
    ("relationship_change", ["Tom Baker", "Ghost"], False),
    ("injury", ["Tom Baker", "Ghost"], True),
    ("death", ["Tom Baker", "Ghost"], True),
    ("secret_revealed", ["Ghost", "Tom Baker"], True)])
def test_pair_events_need_two_grounded_people(etype, who, kept):
    variants = {"Tom Baker": xc.name_variants("Tom Baker")}
    out, _ = xc.ground_state(_state(["Tom Baker"], [(etype, who)]),
                             "Tom sat alone.", variants)
    assert bool(out["events"]) is kept


@pytest.mark.parametrize("text", [
    "Dr Cole laid two fingers to his throat for a long time. \"He's gone,\" "
    "she said. Marcus lay still.",
    "Marcus breathed his last at dawn.",
    "There was no pulse. Marcus did not wake.",
    "They laid Marcus to rest above the chapel.",
    "Marcus was not breathing when they reached him.",
    "Marcus passed on in the night."])
def test_euphemisms_for_death_reach_the_confirmation_step(text):
    variants = {"Marcus Reed": xc.name_variants("Marcus Reed")}
    assert "Marcus Reed" in xc.death_hints(text, variants)


def test_ordinary_gone_is_not_a_death_hint():
    variants = {"Tom Baker": xc.name_variants("Tom Baker")}
    assert xc.death_hints("Tom had gone to fetch the post. The tea was "
                          "gone too.", variants) == {}


# ------------------- a death is only ADDED when two questions agree

def test_a_missed_death_is_added_only_when_the_alive_question_agrees(
        tmp_path, monkeypatch, capsys):
    calls = _setup(tmp_path, monkeypatch, _extract(present=["Tom Baker"]),
                   {"dead": ["Marcus Reed"]},
                   alive={"alive": ["Tom Baker", "Elizabeth Hale"]})
    state = _run(DEATH_TEXT)
    assert [e["who"] for e in state["events"]] == [["Marcus Reed"]]
    (alive_question,) = [c for c in calls if ALIVE_QUESTION in c]
    assert "Candidates: Marcus Reed" in alive_question   # asks only about him
    assert "Tom Baker" not in alive_question.split("Candidates:")[1].split(
        "\n")[0]


def test_a_character_reported_both_dead_and_alive_is_not_killed(
        tmp_path, monkeypatch, capsys):
    """The false deaths the error-location run exposed: one 'yes' for a living
    character standing near death language."""
    _setup(tmp_path, monkeypatch, _extract(present=["Tom Baker"]),
           {"dead": ["Marcus Reed"]}, alive={"alive": ["Marcus Reed"]})
    assert _run(DEATH_TEXT)["events"] == []
    assert "possible death of Marcus Reed not recorded" in \
        capsys.readouterr().out


@pytest.mark.parametrize("alive", ["no idea", '{"alive": "Marcus"}', "[]"])
def test_an_unreadable_alive_answer_blocks_an_added_death(tmp_path,
                                                          monkeypatch, alive):
    _setup(tmp_path, monkeypatch, _extract(present=["Tom Baker"]),
           {"dead": ["Marcus Reed"]}, alive=alive)
    assert _run(DEATH_TEXT)["events"] == []


def test_a_claimed_death_needs_no_second_agreement(tmp_path, monkeypatch):
    """Removing a claimed death is safe on one 'no'; keeping the extractor's
    own claim needs only one confirming 'yes'. The alive question is for
    ADDED deaths only."""
    calls = _setup(tmp_path, monkeypatch, _extract(events=[DIES]),
                   {"dead": ["Marcus Reed"]})
    assert [e["who"] for e in _run(DEATH_TEXT)["events"]] == [["Marcus Reed"]]
    assert not any(ALIVE_QUESTION in c for c in calls)


# ------------------------------------------------ titles and dead characters

@pytest.mark.parametrize("name,expected", [
    ("Dr Cole", "Anna Cole"), ("Dr. Cole", "Anna Cole"),
    ("Doctor Cole", "Anna Cole"), ("Dr Anna Cole", "Anna Cole"),
    ("Miss Hale", "Elizabeth Hale"), ("Mr Baker", "Tom Baker"),
    ("Constable Reed", "Marcus Reed"),
    ("Dr Stranger", "Dr Stranger"),          # no such character
])
def test_titles_do_not_stop_a_name_matching(name, expected):
    from shared.story_state import canonical_name
    canon = ["Elizabeth Hale", "Tom Baker", "Marcus Reed", "Anna Cole"]
    assert canonical_name(name, canon) == expected


def test_strip_title():
    from shared.story_state import strip_title
    assert strip_title("Dr. Cole") == "Cole"
    assert strip_title("Captain Sir Arthur Pym") == "Arthur Pym"
    assert strip_title("Doctor") == "Doctor"             # the whole name
    assert strip_title("Cole") == "Cole"


def test_a_character_who_died_earlier_is_not_listed_as_present(
        tmp_path, monkeypatch, capsys):
    _setup(tmp_path, monkeypatch, _extract(
        present=["Tom Baker", "Marcus Reed", "Liz"]), {})
    update_context("chronology", {
        2: normalize_state(2, "A", {"events": [DIES]}, CANON)})
    state = writer._summarize_and_extract(
        5, "Funeral", "They buried Marcus Reed. Tom and Liz stood by the grave.")[1]
    assert state["present"] == ["Tom Baker", "Elizabeth Hale"]
