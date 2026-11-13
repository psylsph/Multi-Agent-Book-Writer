"""Cast, place and spelling checks (shared/scene_checks.py)."""

from shared.consistency import style_findings, style_rules
from shared.scene_checks import (cast_finding, chapter_cast, place_finding,
                                 seed_places, spelling_finding, spelling_hits)

CHARACTERS = [
    {"name": "Stuart"}, {"name": "Kirsty"},
    {"name": "Angela", "aliases": ["Angela & Pete", "The Instigators"]},
    {"name": "Pete", "aliases": ["Angela & Pete", "The Instigators"]},
    {"name": "James", "aliases": ["James & Chloe", "The Newcomers"]},
    {"name": "Chloe", "aliases": ["James & Chloe", "The Newcomers"]},
]
CH3 = ("Couples: Angela & Pete (Guests) visiting Stuart & Kirsty (Hosts). "
       "Plot: The mood shifts. Angela introduces a \"game\" with stakes.")
# the measured failure: the other couple at a dinner the outline gave to two
INTRUDED = ('Angela set the box down. "Not like the frozen stuff," Kirsty '
            'said. James cleared his throat. "We haven\'t!" Chloe protested. '
            "James nodded slowly. Chloe reached for the wine. James looked "
            "away. Chloe smiled. Pete smiled.")


def test_chapter_cast_reads_the_outline_field():
    assert chapter_cast(CH3, CHARACTERS) == ["Stuart", "Kirsty", "Angela", "Pete"]
    # stops at the next field: "Angela" in the plot is not the cast
    assert chapter_cast("Couples: Stuart & Kirsty. Plot: Angela calls.",
                        CHARACTERS) == ["Stuart", "Kirsty"]
    assert chapter_cast("Characters: James, Chloe", CHARACTERS) == ["James", "Chloe"]


def test_chapter_cast_is_unknown_without_a_named_cast():
    assert chapter_cast("Couples: All three couples. Plot: x", CHARACTERS) is None
    assert chapter_cast("Plot: Stuart and Kirsty argue.", CHARACTERS) is None
    assert chapter_cast("", CHARACTERS) is None


def test_cast_finding_flags_characters_acting_outside_the_cast():
    finding = cast_finding(INTRUDED, chapter_cast(CH3, CHARACTERS), CHARACTERS)
    assert finding["check"] == "cast"
    assert finding["names"] == ["Chloe", "James"]
    assert '"James cleared"' in finding["detail"]
    assert "Stuart, Kirsty, Angela, Pete" in finding["detail"]


def test_cast_finding_allows_mentions_memories_and_talk():
    # the other run: Angela discussed and remembered, never acting 3 times
    text = ('"You decided the night Angela kissed me," Kirsty said. Stuart '
            "thought of Angela's laugh, of Pete's hand on the table, of what "
            'Angela had said. "Angela would love this," he said. James '
            "smiled. Chloe laughed.")
    cast = ["Stuart", "Kirsty", "James", "Chloe"]
    assert cast_finding(text, cast, CHARACTERS) is None


def test_group_aliases_do_not_put_a_partner_in_the_room():
    # with "Angela & Pete" as an alias of Angela, Pete acting counted as her;
    # with "The Instigators", every sentence opening "The" did
    text = "Pete laughed. Pete poured. Pete nodded. The glass turned. The room held."
    finding = cast_finding(text, ["Stuart", "Kirsty", "Pete"], CHARACTERS)
    assert finding is None


def test_cast_finding_needs_a_cast():
    assert cast_finding(INTRUDED, None, CHARACTERS) is None


def test_seed_places_and_place_finding():
    seed = "Chestnut Close, a quiet cul-de-sac in rural Hampshire."
    places = seed_places(seed, "")
    assert places == ["Hampshire"]
    finding = place_finding("It was a heavy, Somerset rain. Hampshire slept.", places)
    assert finding["check"] == "place"
    assert "Somerset" in finding["detail"] and "use Hampshire" in finding["detail"]
    assert place_finding("The Hampshire rain fell.", places) is None


def test_place_finding_leaves_dialogue_and_unset_settings_alone():
    places = ["Hampshire"]
    assert place_finding('"My sister lives in Dorset," Kirsty said.', places) is None
    assert place_finding("A heavy, Somerset rain.", []) is None


def test_spelling_hits_british():
    text = ("The organized neighbor walked toward the gray sidewalk; "
            "she realized her favorite color. Colour and towards are fine.")
    found = dict(spelling_hits(text, "british"))
    assert found == {"organized": "organised", "neighbor": "neighbour",
                     "toward": "towards", "gray": "grey",
                     "sidewalk": "pavement", "realized": "realised",
                     "favorite": "favourite", "color": "colour"}


def test_spelling_keeps_case_and_skips_words_with_british_meanings():
    assert spelling_hits("Gray skies. Pants on the line. Check the tire.",
                         "british") == [("Gray", "Grey")]


def test_spelling_is_opt_in():
    assert spelling_hits("the neighbor", None) == []
    assert spelling_hits("the neighbor", "american") == []
    assert spelling_finding("the colour", "british") is None


def test_spelling_runs_as_a_style_rule():
    style = {"spelling": "british"}
    findings = style_findings("Her neighbor's color.", style)
    assert [f["check"] for f in findings] == ["spelling"]
    assert '"neighbor" -> "neighbour"' in findings[0]["detail"]
    assert any("British spelling" in r for r in style_rules(style))
    assert not any("British" in r for r in style_rules({}))
