"""Unit tests for LLM output parsing/cleanup and seed outline extraction."""

import pytest

from shared.llm_utils import (clean_llm_text, extract_json, extract_seed_characters,
                              extract_seed_extras, render_bible, split_sections)
from shared.context import reset_context, get_context
from agents.planner import extract_seed_outline, _fallback_line_parse


# ---------------------------------------------------------------- extract_json

def test_extract_json_plain_object():
    data = extract_json('{"title": "Foo", "chapters": 3}')
    assert data == {"title": "Foo", "chapters": 3}


def test_extract_json_fenced_array():
    raw = 'Here is the outline:\n```json\n[{"number": 1, "title": "A"}]\n```\nDone.'
    data = extract_json(raw, expect="array")
    assert data == [{"number": 1, "title": "A"}]


def test_extract_json_trailing_comma():
    data = extract_json('{"a": 1, "b": 2,}')
    assert data == {"a": 1, "b": 2}


def test_extract_json_braces_inside_strings():
    raw = '{"summary": "the chapter ends with } and { symbols"}'
    assert extract_json(raw)["summary"].startswith("the chapter")


def test_extract_json_skips_brackets_in_commentary_before_the_json():
    raw = 'Use {name} as the key. Here it is:\n{"title": "Foo"}'
    assert extract_json(raw) == {"title": "Foo"}
    assert extract_json('Items [see below] then [1, 2]', expect="array") == [1, 2]


def test_extract_json_never_returns_a_member_of_a_malformed_object():
    raw = '{"summary": "x" "events": [{"type": "death"}]}'
    with pytest.raises(ValueError, match="could not parse"):
        extract_json(raw)


def test_extract_json_unclosed_object_is_unbalanced():
    with pytest.raises(ValueError, match="unbalanced"):
        extract_json('{"title": "Foo", "events": [')


def test_extract_json_none_found():
    with pytest.raises(ValueError):
        extract_json("no json here at all", expect="object")


# --------------------------------------------------------------- clean_llm_text

def test_clean_llm_text_strips_preamble():
    raw = "Here is the edited chapter:\n\nIt was the season of storms."
    assert clean_llm_text(raw) == "It was the season of storms."


def test_clean_llm_text_strips_duplicate_heading():
    raw = "Chapter 3: The Ledgers\n\nAria climbed the stair."
    assert clean_llm_text(raw) == "Aria climbed the stair."


def test_clean_llm_text_keeps_normal_prose():
    prose = "This chapter was different from the others, and Aria knew it."
    assert clean_llm_text(prose) == prose


def test_clean_llm_text_strips_fences():
    raw = "```\nsome content\n```"
    assert clean_llm_text(raw) == "some content"


# -------------------------------------------------------------- render_bible

def test_render_bible_includes_characters_and_constraints():
    bible = {
        "title": "T", "genre": "mystery", "tone": "spare",
        "premise": "A returns.",
        "characters": [{"name": "Aria", "role": "protagonist",
                        "description": "keeper"}],
        "world": "An island.",
        "constraints": ["Aria left at seventeen"],
    }
    text = render_bible(bible)
    assert "Aria (protagonist): keeper" in text
    assert "Aria left at seventeen" in text
    assert "An island." in text


def test_render_bible_empty():
    assert "no story bible" in render_bible({})


# -------------------------------------------------------- extract_seed_outline

SEED_WITH_SECTION = """# My Book

## Premise
A story.

## Outline
- Chapter 1: Homecoming - Aria returns to the island.
- Chapter 2: What the Flame Remembers - her first memory in the flame.
- Chapter 3: The Ledgers

## Tone
Sparse.
"""

SEED_LOOSE_LINES = """A premise paragraph.

Chapter 1: Arrival - she lands at dawn.
Some prose that is not an outline at all.
Chapter 2: The Wreck
"""

SEED_NO_OUTLINE = """# My Book

## Characters
- Aria - a keeper.
"""

# Parts, wrapped summaries, and a fold-in beat (mirrors a structured seed)
SEED_WITH_PARTS = """# My Book

## Outline

### Part One - The Setup

- Chapter 1: The Fall (POV: hero, heat 2) - He rides the woods loop
  behind the house and comes off the bike on a rooty descent.
- Chapter 2: The Treatment (POV: therapist, heat 4) - First meeting at
  the clinic. The session is charged but nothing crosses.

### Part Two - The Build

- Chapter 3: The Party (heat 6) - A charged evening with the group.
- The picnic beat (folded into the build) - A comedic beat that sets up
  a later payoff; not its own chapter.

Heat curve: 2-4-6.

## Explicitness
Keep it tasteful.
"""


def test_outline_from_dedicated_section():
    outline = extract_seed_outline(SEED_WITH_SECTION)
    assert [c["number"] for c in outline] == [1, 2, 3]
    assert outline[0]["title"] == "Homecoming"
    assert outline[0]["summary"].startswith("Aria returns")
    assert outline[2]["title"] == "The Ledgers"
    assert outline[2]["summary"] == ""


def test_outline_loose_scan_requires_two_chapter_lines():
    outline = extract_seed_outline(SEED_LOOSE_LINES)
    assert len(outline) == 2
    assert outline[1]["title"] == "The Wreck"


def test_outline_none_when_absent():
    assert extract_seed_outline(SEED_NO_OUTLINE) == []
    assert extract_seed_outline("") == []


def test_outline_bare_bullets_only_count_inside_section():
    # bare "- Title - summary" lines outside an Outline section are ignored
    seed = "## Characters\n- Aria - a keeper\n- Pip - a gull-whisperer\n"
    assert extract_seed_outline(seed) == []


def test_outline_with_parts_wrapped_lines_and_beats():
    outline = extract_seed_outline(SEED_WITH_PARTS)

    # all three real chapters, in order, with unique numbers
    assert [c["number"] for c in outline] == [1, 2, 3]
    assert outline[0]["title"] == "The Fall (POV: hero, heat 2)"
    assert outline[2]["title"] == "The Party (heat 6)"

    # wrapped continuation lines join the summary
    assert "rooty descent" in outline[0]["summary"]
    assert "nothing crosses" in outline[1]["summary"]

    # part labels are captured
    assert outline[0]["part"] == "Part One - The Setup"
    assert outline[2]["part"] == "Part Two - The Build"

    # the non-chapter beat folds into the previous chapter's summary,
    # and trailing non-chapter lines (heat curve) do the same
    assert "later payoff" in outline[2]["summary"]
    assert "Heat curve" in outline[2]["summary"]

    # the Explicitness section after the outline is NOT swallowed
    assert all("tasteful" not in c["summary"] for c in outline)


def test_outline_bare_bullet_section_without_chapter_numbers():
    seed = """## Outline
- The Fall - he comes off the bike
- The Treatment - the clinic session
"""
    outline = extract_seed_outline(seed)
    assert [c["number"] for c in outline] == [1, 2]
    assert outline[0]["title"] == "The Fall"


def test_outline_duplicate_numbers_sanitized():
    seed = ("## Outline\n"
            "Chapter 1: One\n"
            "Chapter 1: One-again - inserted duplicate\n"
            "Chapter 2: Two\n")
    outline = extract_seed_outline(seed)
    assert [c["number"] for c in outline] == [1, 2, 3]


def test_fallback_line_parse():
    raw = """Sure, here's a plan:

Chapter 1: The Arrival - Aria lands on Hesswick.
Chapter 2: The Beacon

Some closing commentary."""
    chapters = _fallback_line_parse(raw)
    assert len(chapters) == 2
    assert chapters[0]["title"] == "The Arrival"
    assert chapters[1]["summary"] == ""


# ------------------------------------------------------ seed section parsing

def test_split_sections_keeps_sub_headers_in_body():
    seed = "# Book\nPreamble text.\n\n## Outline\nstuff\n\n### Part One\nmore\n"
    sections = split_sections(seed)
    assert len(sections) == 2
    assert sections[0][0] == "Book"
    assert sections[0][1] == "Preamble text."
    assert "### Part One" in sections[1][1]


def test_extract_seed_extras_constraints_and_notes():
    seed = """# My Book
Dry humour, British English.

## Premise
A story.

## Explicitness
Very explicit, no euphemisms.

## Constraints
- Never the word "unhurried".
- All characters are adults.
"""
    constraints, notes = extract_seed_extras(seed)
    assert constraints == ['Never the word "unhurried".',
                           "All characters are adults."]
    assert "Very explicit, no euphemisms." in notes
    assert "Dry humour, British English." in notes
    assert "A story." not in notes  # known sections are not notes


def test_extract_seed_characters_bold_bullets():
    seed = """## Characters

### The trio

- **Stuart** - protagonist. 52, works in IT, quiet, dry.
- **Lisa (the third)** - 35, sports therapist, direct.

### The wider circle

- **Amy** - owns the cabins.
"""
    chars = extract_seed_characters(seed)
    # a bracketed qualifier is part of the description, not of the name
    assert [c["name"] for c in chars] == ["Stuart", "Lisa", "Amy"]
    assert chars[1]["description"].endswith("(the third)")
    assert chars[0]["role"] == "protagonist"
    assert chars[0]["description"].startswith("protagonist.")
    assert chars[1]["role"] == ""
    assert "sports therapist" in chars[1]["description"]


def test_extract_seed_extras_joins_wrapped_constraints():
    seed = """# My Book

## Constraints
- Em-dash rare — max 8 per scene, never in dialogue (only for a sentence
  cut off mid-word).
- All characters are adults.
"""
    constraints, _ = extract_seed_extras(seed)
    assert constraints == [
        "Em-dash rare — max 8 per scene, never in dialogue "
        "(only for a sentence cut off mid-word).",
        "All characters are adults.",
    ]


def test_extract_seed_characters_joins_wrapped_descriptions():
    seed = """## Characters

### The trio

- **Stuart** - protagonist. 52, works in IT, fit,
  mountain-bikes the woods. Quiet, dry.
- **Lisa (the third)** - 35, sports therapist,
  direct, at ease in her body.
"""
    chars = extract_seed_characters(seed)
    assert len(chars) == 2
    assert chars[0]["description"].endswith("Quiet, dry.")
    assert "mountain-bikes the woods" in chars[0]["description"]
    assert chars[1]["description"].endswith("in her body.")


# ------------------------------------------------------------------- context

def test_reset_context_mutates_in_place():
    import shared.context as ctx
    ctx.context["title"] = "Changed"
    reset_context()
    assert ctx.context["title"] == ""
    assert get_context("title") == ""
    assert get_context()["chapters"] == []


# A seed that marks its sections with bold lines instead of headings, after a
# real "## Constraints" heading (the shape of a seed that broke both parsers).
BOLD_SEED = """# The Close
**The Setting:**
A cul-de-sac.
## Constraints
- **POV** - Single POV
* **Duration** - A few weeks
**Plot Roadmap:**
**Chapter 1: The Rant**
* **Couples:** Ann & Bo.
* **Plot:** Boredom sets in.
**Chapter 2: The Swap**
* **Plot:** A dull dinner.
**Themes:**
* Masks.
"""


def test_a_bold_line_ends_the_constraints():
    """Regression: the section ran on to the end of the seed, so every
    roadmap bullet became a 'constraint' - in the bible, and in the check that
    then rejected every seed expansion for 'rewording' them."""
    from shared.llm_utils import extract_seed_extras
    constraints, _ = extract_seed_extras(BOLD_SEED)
    assert constraints == ["**POV** - Single POV", "**Duration** - A few weeks"]


def test_a_star_bullet_keeps_its_bold_text():
    from shared.llm_utils import extract_seed_extras
    constraints, _ = extract_seed_extras(
        "## Constraints\n* **Tense** - past\n*plain star bullet\n")
    assert constraints == ["**Tense** - past", "plain star bullet"]


def test_chapters_without_an_outline_heading_keep_their_notes():
    """Regression: 'Chapter N' lines found without an Outline heading lost the
    lines under them, so every chapter's summary was empty - and the seed's
    outline overrides the planner's, so the writer got none."""
    from agents.planner import extract_seed_outline
    chapters = extract_seed_outline(BOLD_SEED)
    assert [c["title"] for c in chapters] == ["The Rant", "The Swap"]
    assert chapters[0]["summary"] == "Couples: Ann & Bo. Plot: Boredom sets in."
    assert chapters[1]["summary"] == "Plot: A dull dinner."   # Themes not taken


def test_chapters_in_different_sections_are_not_merged():
    from agents.planner import extract_seed_outline
    seed = ("## Intro\nChapter 1: Start - she lands.\nIntro prose.\n"
            "## Notes\nChapter 2: End - she leaves.\nOther prose.\n")
    chapters = extract_seed_outline(seed)
    assert [c["summary"] for c in chapters] == ["she lands.", "she leaves."]


def test_stated_chapter_words_reads_ranges_and_single_numbers():
    """Regression: "1,500–1,700 words per chapter" fitted no field of the
    seed reviewer's JSON, so the configured 2,000 replaced it."""
    from shared.llm_utils import stated_chapter_words
    assert stated_chapter_words(
        "* **Word Count & Depth:** 1,500–1,700 words per chapter.") == \
        [(1500, 1700)]
    assert stated_chapter_words("about 2000 words a chapter") == [(2000, 2000)]
    assert stated_chapter_words("Chapters of around 1.8k words") == \
        [(1800, 1800)]
    assert stated_chapter_words("3,000 to 2,500 words each chapter") == \
        [(2500, 3000)]
    assert stated_chapter_words(
        "1,500-1,700 words per chapter. 2,000 words per chapter.") == \
        [(1500, 1700), (2000, 2000)]
    assert stated_chapter_words("a 50,000 word novel in 12 chapters") == []
    assert stated_chapter_words("5 words per chapter") == []


def test_strip_chapter_length_takes_out_only_the_length():
    from shared.llm_utils import strip_chapter_length as strip
    line = ("* **Word Count & Depth:** 1,500–1,700 words per chapter. 2,000 "
            "words per chapter. No summaries. Meticulous sensory detail.\n")
    text, removed = strip(line)
    assert text == ("* **Word Count & Depth:** No summaries. Meticulous "
                    "sensory detail.\n")
    assert removed == ["1,500–1,700 words per chapter.",
                       "2,000 words per chapter."]
    # an outline entry's bracketed length; other brackets stay
    assert strip("- Chapter 1: The Rant - the club (about 2,000 words).")[0] \
        == "- Chapter 1: The Rant - the club."
    assert strip("- Chapter 2: The Swap (Stuart POV, heat 3).")[1] == []
    # a line that was only the length goes; its neighbours stay
    assert strip("Intro.\n* **Length:** 2,000 words per chapter.\nEnd.\n")[0] \
        == "Intro.\nEnd.\n"
    assert strip("Write in third person. Chapters of about 1,800 words. "
                 "Keep it tense.")[0] == "Write in third person. Keep it tense."
    for keep in ("**Target Length:** approximately 50,000 words",
                 "Kirsty was the first to arrive; she wrote 300 words of notes.",
                 "Chapter 3 is set in 1750."):
        assert strip(keep) == (keep, [])
