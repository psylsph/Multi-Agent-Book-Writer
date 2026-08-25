"""Unit tests for the deterministic consistency linter."""

from shared.consistency import (extract_banned_words, extract_quotas,
                                lint_chapter)

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


def test_book_scope_quota_not_flagged_per_chapter():
    bible = {"characters": []}
    text = "Her collarbone showed. Her collarbone again."
    findings = lint_chapter(1, text, bible, USER_CONSTRAINTS)
    # collarbone is a volume-level quota: not flagged per chapter
    assert not any("collarbone" in f["detail"] for f in findings)
