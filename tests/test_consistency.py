"""Unit tests for the deterministic consistency linter."""

from shared.consistency import (extract_banned_words, extract_quotas,
                                lint_chapter, style_findings, style_metrics,
                                word_count, word_count_finding)

USER_CONSTRAINTS = [
    'British English. Never the words "unhurried", "unrushed", or "exactly".',
    "Em-dash rare — max 8 per scene, never in dialogue (only for a sentence "
    "cut off mid-word).",
    'Sentence-opening "And" max 5 per scene.',
    "Adverbs max 8 per 1000 words.",
    '"the thing" max 6; "which was" max 4; "that was" max 8.',
    '"collarbone" max once per volume.',
    'Never "black lace".',
    "All characters are adults.",
]


def test_banned_words():
    banned = extract_banned_words(USER_CONSTRAINTS)
    for word in ("unhurried", "unrushed", "exactly", "black lace"):
        assert word in banned, word
    assert "British" not in banned


def test_quotas_extraction():
    quotas = {(q["target"], q["phrase"]): q for q in
              extract_quotas(USER_CONSTRAINTS)}
    assert quotas[("em_dash", "\u2014")]["max"] == 8
    assert quotas[("sentence_opening", "And")]["max"] == 5
    assert quotas[("adverbs", "adverbs")]["scope"] == "per_1000_words"
    assert quotas[("phrase", "the thing")]["max"] == 6
    assert quotas[("phrase", "collarbone")]["scope"] == "book"
    assert quotas[("phrase", "collarbone")]["max"] == 1


def test_lint_chapter_catches_banned_word_and_quota():
    bible = {"characters": [{"name": "Aria"}]}
    text = ("She moved unhurried and slowly softly gently. "
            "And he waited. And she sighed. And it rained. "
            "And they left. And more. And more again. "
            "The thing was the thing indeed.")
    findings = lint_chapter(1, text, bible, USER_CONSTRAINTS)
    checks = [f["check"] for f in findings]
    assert "banned_word" in checks            # 'unhurried'
    assert "quota" in checks                  # 'And' x6 > 5


def test_lint_chapter_clean_text_has_no_findings():
    bible = {"characters": [{"name": "Aria"}]}
    text = ("Aria climbed the stair in the dark. The lamp had failed again. "
            "She struck a match and kept going, slowly enough to be safe.")
    findings = lint_chapter(1, text, bible, USER_CONSTRAINTS)
    assert findings == []


def test_lint_chapter_name_near_miss():
    bible = {"characters": [{"name": "Kirsty"}]}
    findings = lint_chapter(1, "Kirstie smiled and waved twice.",
                            bible, [])
    assert any(f["check"] == "name_mismatch" and "Kirsty" in f["detail"]
               for f in findings)


def test_lint_chapter_name_check_no_determiner_false_positives():
    bible = {"characters": [{"name": "The Keeper"}]}  # 'The' as a name word
    text = "These were the days. Those nights mattered. There it was."
    findings = lint_chapter(1, text, bible, [])
    assert not any(f["check"] == "name_mismatch" for f in findings)


def test_book_scope_quota_not_flagged_per_chapter():
    bible = {"characters": []}
    text = "Her collarbone showed. Her collarbone again."
    findings = lint_chapter(1, text, bible, USER_CONSTRAINTS)
    # collarbone is a volume-level quota: not flagged per chapter
    assert not any("collarbone" in f["detail"] for f in findings)


# ------------------------------------------------------ word count (wc -w)

def test_word_count_matches_wc_semantics():
    text = "One  two\tthree\n\nfour   five"  # runs of whitespace collapse
    assert word_count(text) == 5
    # wc -w equivalence on a paragraph with mixed whitespace
    para = "The  quick brown\tfox.\nJumps over  the lazy dog."
    assert word_count(para) == 9


def test_word_count_finding_flags_short_chapter():
    short = "word " * 1200                      # 1200 words
    finding = word_count_finding(short, target_words=2000, tolerance=0.8)
    assert finding is not None
    assert finding["check"] == "word_count"
    assert "1200 words" in finding["detail"]
    assert "1600" in finding["detail"]           # minimum
    assert "SUBSTANCE" in finding["detail"]      # anti-padding instruction


def test_word_count_finding_passes_at_minimum():
    ok = "word " * 1600                          # exactly at the minimum
    assert word_count_finding(ok, 2000, 0.8) is None


def test_word_count_finding_disabled_without_target():
    assert word_count_finding("a few words", None, 0.8) is None


# ------------------------------------------- banned words: whole words only

def test_banned_word_matches_whole_words_only():
    bible = {"characters": []}
    findings = lint_chapter(1, "It hurt his heart. Part of him knew.", bible,
                            ['Never use "art".'])
    assert findings == []
    findings = lint_chapter(1, "The art was lovely.", bible,
                            ['Never use "art".'])
    assert [f["check"] for f in findings] == ["banned_word"]


def test_bare_words_only_banned_after_the_word():
    assert extract_banned_words(["Never mention orange things."]) == []
    assert extract_banned_words(["Never use passive voice."]) == []
    assert extract_banned_words(["Never use the word unhurried."]) == \
        ["unhurried"]


# ----------------------------------------------------- dead-character check

def _dead_hits(text, name="Anna"):
    from shared.consistency import check_chronology
    cum = {"dead": {name: 2}, "met_pairs": {}}
    return [f for f in check_chronology(5, text, cum)
            if f["check"] == "dead_character"]


def test_dead_character_acting_alive_is_flagged_across_a_gap():
    assert _dead_hits("Anna said hello.")
    assert _dead_hits("Anna slowly turned around.")        # gap before verb
    assert _dead_hits("Anna, pale and cold, walked in.")
    assert _dead_hits('"Fine," said Anna.')
    assert _dead_hits("Anna's voice carried down the hall.")


def test_dead_character_false_positives_avoided():
    assert not _dead_hits("She remembered Anna; he walked away.")
    assert not _dead_hits("Anna's mother walked in.")        # possessive
    assert not _dead_hits("The funeral was for Anna. He left early.")
    # name match is case-sensitive: the verb 'rose' is not the dead 'Rose'
    assert not _dead_hits("The sun rose and she walked on.", name="Rose")
    assert _dead_hits("Rose stood at the door.", name="Rose")


def test_word_count_finding_has_structured_fields():
    finding = word_count_finding("word " * 10, 100, 0.8)
    assert finding["count"] == 10 and finding["minimum"] == 80


# ------------------------------------------------------------- style lint

STYLE = {
    "banned_phrases": ["coil of", "reclamation"],
    "max_phrase_hits": 1,
    "max_hedges_per_1000": 4,
    "min_sentence_cv": 0.30,
}


def test_an_empty_style_block_adds_nothing():
    """The default is `style: {}`; it must not change any existing book."""
    text = "The coil of heat. The coil of need. Somehow it seemed to matter."
    assert style_findings(text, {}) == []
    assert style_findings(text, None) == []


def test_banned_phrase_over_the_limit_is_flagged():
    text = ("A coil of rope. Another coil of smoke. A third coil of hair. "
            + "The room was quiet. " * 12)
    findings = style_findings(text, STYLE)
    flagged = [f for f in findings if f["check"] == "banned_phrase"]
    assert flagged and "coil of" in flagged[0]["detail"]


def test_a_phrase_within_the_limit_is_not_flagged():
    text = ("A single coil of rope. " + "The room was quiet and empty. " * 12)
    findings = style_findings(text, STYLE)
    assert not [f for f in findings if f["check"] == "banned_phrase"]


def test_hedging_above_the_configured_rate_is_flagged():
    text = ("It seemed to be late. She somehow knew. A sense of dread came. "
            "He almost as if waited. " + "The clock ticked past noon. " * 20)
    findings = style_findings(text, {"max_hedges_per_1000": 2})
    assert any(f["check"] == "hedging" for f in findings)


def test_flat_sentences_are_flagged_and_varied_ones_are_not():
    flat = ("He walked to the door. She opened the door. He walked inside. "
            "She closed the door. He sat down. She sat down too.")
    varied = ("He walked to the door. She opened it. Outside, the rain had "
              "stopped, and the whole street smelled of wet stone and diesel. "
              "He stepped out. She did not follow. A bus passed. The street "
              "was empty again, and he stood in it until he was cold.")
    assert any(f["check"] == "flat_prose"
               for f in style_findings(flat, {"min_sentence_cv": 0.30}))
    assert not [f for f in style_findings(varied, {"min_sentence_cv": 0.30})
                if f["check"] == "flat_prose"]


def test_pov_slip_is_flagged_only_when_clear():
    first_person = ("I opened the door. I saw the body. My hands were wet. "
                    "I knew the man. I had met him before. I took the letter. "
                    "I could not stay. I left.")
    assert any(f["check"] == "pov"
               for f in style_findings(first_person, {"pov": "third"}))
    assert not [f for f in style_findings(first_person, {"pov": "first"})
                if f["check"] == "pov"]


def test_lint_chapter_passes_style_through():
    bible = {"characters": [{"name": "Aria"}]}
    text = ("A coil of rope. Another coil of smoke. A third coil of hair. "
            "And another coil of wire. " + "The room was quiet. " * 12)
    plain = lint_chapter(1, text, bible, [])
    styled = lint_chapter(1, text, bible, [], style=STYLE)
    assert not [f for f in plain if f["check"] == "banned_phrase"]
    assert any(f["check"] == "banned_phrase" for f in styled)


def test_style_metrics_match_the_eval_tool():
    """The pipeline and tools/style_eval.py must score the same text the same
    way, or a threshold picked on the battery means nothing in the editor."""
    from tools.style_eval import (hedge_per_1k, sentence_cv, stock_per_1k)
    text = ("A coil of smoke. Somehow it seemed to matter. The room was "
            "quiet. It was very quiet. Nobody spoke. Nobody moved at all.")
    assert style_metrics(text)["sentence_cv"] == sentence_cv(text)
    assert style_metrics(text)["hedges_per_1000"] == hedge_per_1k(text)
    assert stock_per_1k(text) >= 1.0


# ------------------------------------------- style checks: false positives

def test_dialogue_does_not_decide_the_point_of_view():
    """Regression: a third-person, past-tense narration with first-person,
    present-tense dialogue read as first person / present."""
    from shared.consistency import pov_tense
    text = ('"I am tired and I need to go, you know I do," she said. '
            '"I think you are right. I see it," he answered. ') * 6
    pov, _, tense, _ = pov_tense(text)
    assert pov == "third"
    assert tense == "past"


def test_words_ending_in_ed_that_are_not_verbs_are_not_past():
    from shared.consistency import pov_tense
    text = ("I need the speed. I feed the seed. Indeed I need a hundred. "
            "I am here. I see it. I know it. ") * 5
    assert pov_tense(text)[2] == "present"


def test_a_sentence_ends_after_a_closing_quote():
    from shared.consistency import _sentences
    assert _sentences('"Go." She went. “Now?” He nodded.') == [
        '"Go."', "She went.", "“Now?”", "He nodded."]


def test_banned_phrases_match_whole_words():
    from shared.consistency import style_metrics
    metrics = style_metrics("A sigh. The sight of it. Another sigh.",
                            {"banned_phrases": ["sigh"]})
    assert metrics["phrase_hits"] == {"sigh": 2}


def test_a_bad_style_number_does_not_crash_the_lint():
    from shared.consistency import style_findings
    text = "She went home. " * 20
    assert style_findings(text, {"max_hedges_per_1000": "eight",
                                 "min_pov_share": None, "pov": "first"}) \
        is not None


def test_introduced_names():
    from shared.consistency import introduced_names
    before = "she told marcus about the tower in london."
    assert introduced_names(before, "She told Marcus about the tower in "
                            "London.") == []
    assert introduced_names(before, "She told Darius about the tower in "
                            "London.") == ["Darius"]


def test_style_rules_lists_what_the_lint_checks():
    from shared.consistency import style_rules
    rules = style_rules(
        {"banned_phrases": ["coil of"], "max_phrase_hits": 2,
         "max_hedges_per_1000": 8, "pov": "first", "tense": "present"},
        ['Never the words "unhurried" or "exactly".',
         "Em-dash rare - max 8 per scene"])
    text = "\n".join(rules)
    assert '"unhurried"' in text and '"exactly"' in text
    assert "At most 8 em dashes per chapter." in text
    assert '"coil of"' in text and "at most 2 each" in text
    assert "first person" in text and "present tense" in text
    assert "hedges" in text.lower()


def test_style_rules_are_empty_without_rules():
    from shared.consistency import style_rules
    assert style_rules({}) == []
    assert style_rules({"enabled": False, "pov": "first"}) == []
    assert style_rules(None, ["Keep it light."]) == []


def test_fused_words_are_split_back_to_the_source_text():
    """Regression: a polish copied "against her" as "againsther"."""
    from shared.consistency import repair_fused_words
    before = "He whispered against her ear. Rain fell."
    text, fixes = repair_fused_words(before, "He whispered againsther ear. Rain fell.")
    assert text == before and fixes == [("againsther", "against her")]


def test_only_fusions_of_words_the_source_had_side_by_side_are_repaired():
    from shared.consistency import repair_fused_words
    before = "She took the train home."
    # 'nothing' is new but 'no thing' was never in the source: left alone
    assert repair_fused_words(before, "Nothing. She took the train home.")[1] == []
    # a word already in the source is never touched
    assert repair_fused_words("a rainbow and a rain bow",
                              "a rainbow and a rainbow")[1] == []


def test_garbled_words_are_new_mixed_case_debris():
    from shared.consistency import garbled_words
    before = "An unremarkable week with McGregor and his iPhone."
    after = "An unremarkableRE week with McGregor, his iPhone and a MacBook."
    assert garbled_words(before, after) == ["unremarkableRE"]


# ------------------------------------------------------- narrative person

_THIRD = "He walked to the door and she followed him into the rain. " * 20
_FIRST = "I walked to the door and my hands shook in the rain again. " * 20


def test_chapter_person_ignores_dialogue_and_italic_thoughts():
    from shared.consistency import chapter_person
    speech = ('"I know," he said. "My mind is made up." *I can\'t,* he '
              'thought. ')
    assert chapter_person(_THIRD + speech * 5) == "third"
    assert chapter_person(_FIRST) == "first"
    assert chapter_person(_THIRD * 5 + "I went. " * 6) == "mixed"
    assert chapter_person("Too short. I went.") == ""


def test_a_part_way_slip_is_caught_where_the_pronoun_share_is_not():
    """Regression: a chapter that turned to "I" part-way still read 69% third
    person by pronoun share (narrators talk about "she" a lot), so
    book.style.pov: third let it through."""
    from shared.consistency import pov_slip_finding
    text = _THIRD + "\n\n" + _THIRD + "\n\n" + _FIRST
    found = [f for f in style_findings(text, {"pov": "third"})
             if f["check"] == "pov"]
    assert found and "paragraph 3 of 3" in found[0]["detail"]
    assert pov_slip_finding(_THIRD, "third") is None
    assert pov_slip_finding(_FIRST, "first") is None
    assert pov_slip_finding(_THIRD, "first")["check"] == "pov"


def test_book_pov_is_third_person_unless_config_or_seed_says_otherwise():
    from shared.consistency import book_pov, stated_pov
    assert stated_pov("Third-person limited, past tense.") == "third"
    assert stated_pov("Single POV (Stuart).") is None
    assert stated_pov("first person POV, then third-person narration") is None
    assert stated_pov("POV: first person, past tense.") == "first"
    assert stated_pov("Kirsty was the first person to arrive.") is None
    assert book_pov({}, ["Single POV (Stuart)."]) == ("third", "the default")
    assert book_pov({}, []) == ("third", "the default")
    assert book_pov({}, ["A first-person narrator."]) == ("first", "the seed")
    assert book_pov({"pov": "first"}, ["third person"]) == \
        ("first", "book.style.pov")
    assert book_pov({"pov": "second"}, []) == (None, "")


def test_word_count_finding_flags_a_chapter_over_the_maximum():
    from shared.consistency import word_count_finding
    finding = word_count_finding("word " * 160, 100, 0.8, max_ratio=1.5)
    assert finding["check"] == "too_long"
    assert (finding["count"], finding["maximum"], finding["target"]) == (160, 150, 100)
    assert word_count_finding("word " * 150, 100, 0.8, max_ratio=1.5) is None
    assert word_count_finding("word " * 999, 100, 0.8, max_ratio=0) is None


def test_max_length_is_never_under_what_the_writer_is_asked_for():
    from shared.consistency import max_length
    assert max_length(1600, 1.5) == 2400
    assert max_length(1600, 1.0) == 1920          # 1.2x: the writer's own ceiling
    assert max_length(1600, 0) == 0


def test_dropped_names_matches_any_part_of_a_name():
    from shared.consistency import dropped_names
    names = ["Mara Vance", "Tom", "Elias Crane"]
    before = "Mara spoke. Tom nodded. Crane watched."
    assert dropped_names(before, "Vance spoke. Crane watched.", names) == ["Tom"]
    assert dropped_names(before, before, names) == []
