"""Chapters with planted errors, for tools/reviewer_eval.py.

An invented world and four earlier chapters of recorded story facts:

  ch1 Tuesday dusk     Elizabeth meets Tom (first meeting)
  ch2 Wednesday noon   Elizabeth meets Marcus; a letter naming HER heir (a
                       secret revealed to Elizabeth only)
  ch3 Wednesday night  Tom's fingers are crushed; Elizabeth and Tom grow close
  ch4 Thursday dawn    Marcus is found drowned

The chapter under review is chapter 5 (Friday): Elizabeth and Tom repair the
winch and, with Dr Cole, plan Marcus's funeral. Every case is that same
chapter with ONE change, so a model that flags a clean chapter, or misses the
planted error, is easy to see. The cases mix clean chapters, including tempting
ones (a dream about the dead man; a warm scene that IS consistent with the
established relationship), with one planted error of each kind the reviewer
claims to check.
"""

from shared.story_state import normalize_state

NUMBER, TITLE = 5, "The Winch"

CANON = ["Elizabeth Hale", "Tom Baker", "Marcus Reed", "Anna Cole"]

BIBLE = {
    "title": "The Marsh Light", "genre": "gothic mystery",
    "tone": "dry, quiet, close third person",
    "premise": "A surveyor inherits a drowned chapel on a salt marsh.",
    "world": "Saltmere, a tidal marsh village reached by ferry.",
    "characters": [
        {"name": "Elizabeth Hale", "role": "protagonist", "aliases": ["Liz"],
         "description": "A surveyor."},
        {"name": "Tom Baker", "role": "supporting", "aliases": [],
         "description": "The ferryman."},
        {"name": "Marcus Reed", "role": "supporting", "aliases": [],
         "description": "The harbourmaster."},
        {"name": "Anna Cole", "role": "supporting", "aliases": [],
         "description": "The village doctor."},
    ],
    "constraints": [
        "Close third person, past tense, from Elizabeth's point of view only: "
        "never enter another character's head.",
    ],
    "notes": "",
}

CHAPTERS = [
    {"number": 1, "title": "The Landing", "summary": "Elizabeth meets Tom."},
    {"number": 2, "title": "The Chapel", "summary": "Marcus gives her the letter."},
    {"number": 3, "title": "Storm", "summary": "Tom is hurt; they grow close."},
    {"number": 4, "title": "The Drowned Man", "summary": "Marcus is found dead."},
    {"number": NUMBER, "title": TITLE,
     "summary": "Elizabeth and Tom repair the winch and, with Dr Cole, plan "
                "Marcus's funeral."},
]

BRIEF = ("- Plot beats: Elizabeth and Tom repair the winch; Dr Cole arrives; "
         "the funeral is planned for the chapel.")

_STATES = [
    (1, "The Landing", {
        "summary": "Elizabeth Hale arrives at Saltmere and meets ferryman "
                   "Tom Baker.",
        "time": "Tuesday dusk", "location": "Saltmere landing",
        "present": ["Elizabeth Hale", "Tom Baker"],
        "events": [{"type": "first_meeting",
                    "who": ["Elizabeth Hale", "Tom Baker"]}]}),
    (2, "The Chapel", {
        "summary": "Marcus Reed gives Elizabeth a letter naming her heir to "
                   "the chapel and the land.",
        "time": "Wednesday noon", "location": "the drowned chapel",
        "present": ["Elizabeth Hale", "Marcus Reed"],
        "events": [{"type": "first_meeting",
                    "who": ["Elizabeth Hale", "Marcus Reed"]},
                   {"type": "secret_revealed", "who": ["Elizabeth Hale"],
                    "detail": "the letter names her heir to the chapel"}]}),
    (3, "Storm", {
        "summary": "Tom's fingers are crushed in the winch; he and Elizabeth "
                   "grow close.",
        "time": "Wednesday night", "location": "the boathouse",
        "present": ["Elizabeth Hale", "Tom Baker"],
        "events": [{"type": "injury", "who": ["Tom Baker"],
                    "detail": "two crushed fingers"},
                   {"type": "relationship_change",
                    "who": ["Elizabeth Hale", "Tom Baker"],
                    "detail": "grew close"}]}),
    (4, "The Drowned Man", {
        "summary": "Marcus Reed is found drowned at the chapel steps.",
        "time": "Thursday dawn", "location": "the landing",
        "present": ["Elizabeth Hale", "Tom Baker"],
        "events": [{"type": "death", "who": ["Marcus Reed"],
                    "detail": "drowned"}]}),
]


def chronology():
    """The recorded story state of chapters 1-4."""
    return {n: normalize_state(n, title, data, CANON)
            for n, title, data in _STATES}


# ---------------------------------------------------------------- the prose

OPEN = ("Friday dawn found Elizabeth and Tom back at the winch. The new rope "
        "lay coiled between them, still stiff from the shop, and Tom held the "
        "end steady with his good hand while Elizabeth worked the knot. His "
        "bandaged fingers stayed tucked against his chest, and whenever the "
        "rope bit he winced and said nothing. They had not spoken of the "
        "storm since the night it broke.")

FUNERAL = ("By mid-morning Dr Anna Cole came down the jetty with her bag over "
           "her shoulder. She sat on an upturned crate and asked whether "
           "anything had been settled about Marcus. Nothing had. Tom "
           "suggested the chapel, because Marcus had loved it. Elizabeth said "
           "the chapel was hers now, and she would not have the harbourmaster "
           "buried anywhere else. Anna wrote it in her notebook: Sunday, at "
           "noon, if the tide allowed.")

CLOSE = ("When the winch was rigged they tested it twice, and it held both "
         "times. Elizabeth looked out over the flat bright marsh and thought, "
         "not for the first time, how strange it was to have inherited a place "
         "that already felt like home.")


def chapter(opening=OPEN, funeral=FUNERAL, close=CLOSE):
    return "\n\n".join([opening, funeral, close])


# (id, planted type or None, draft). The planted type is what a good reviewer
# should name; None means the chapter is clean and should pass.
CASES = [
    # --- clean chapters (must pass), the last three deliberately tempting
    ("clean-a", None, chapter()),
    ("clean-b", None, chapter(
        OPEN + " She had dreamed of Marcus in the night, standing in the "
               "pulpit with the letter in his hand, and had woken before he "
               "could speak.",
        FUNERAL.replace("Nothing had.", "Nothing had. Tom said Marcus would "
                                        "have hated the fuss."))),
    ("clean-c", None, chapter(
        OPEN.replace("They had not spoken of the storm since the night it "
                     "broke.",
                     "Tom's good hand found hers on the rope, and neither of "
                     "them moved it. \"Morning, Liz,\" he said quietly."))),
    ("clean-d", None, chapter(
        OPEN.replace("Friday dawn found", "A day after the fishermen had "
                     "found Marcus, Friday dawn found"))),

    # --- planted errors, one of each kind
    ("dead", "dead_resurrection", chapter(
        OPEN, "Marcus Reed stood at the end of the jetty with his ledger "
              "under one arm. \"Mind that knot,\" he called, and Elizabeth "
              "nodded without looking up.\n\n" + FUNERAL)),
    ("regress", "relationship_regression", chapter(
        "Friday dawn found Elizabeth at the winch with a stranger's hand held "
        "out to her. \"Tom Baker,\" he said. \"Pleased to meet you.\" She had "
        "never seen him before, and they shook hands for the first time. "
        "Together they lifted the new rope, still stiff from the shop, and "
        "Tom held the end steady with his good hand while Elizabeth worked "
        "the knot. His bandaged fingers stayed tucked against his chest.")),
    ("leap", "relationship_leap", chapter(OPEN.replace(
        "Friday dawn found Elizabeth and Tom back at the winch.",
        "Friday dawn found Elizabeth and Tom back at the winch. Tom, her "
        "husband of ten years, kissed the top of her head the way he had "
        "every morning of their marriage."))),
    ("knowledge", "knowledge", chapter(OPEN, FUNERAL.replace(
        "Tom suggested the chapel, because Marcus had loved it. Elizabeth "
        "said the chapel was hers now, and she would not have the "
        "harbourmaster buried anywhere else.",
        "\"The letter names you heir to the chapel and all the land around "
        "it,\" Tom said, \"so of course you'll want him buried there.\" "
        "Elizabeth nodded."))),
    ("heal", "timeline", chapter(
        "Friday dawn found Elizabeth and Tom back at the winch. The new rope "
        "lay coiled between them, still stiff from the shop, and Tom hauled "
        "on it with both hands, the fingers that had been crushed on "
        "Wednesday night working as easily as if nothing had ever happened "
        "to them. They had not spoken of the storm since the night it "
        "broke.")),
    ("weekday", "timeline", chapter(OPEN.replace(
        "Friday dawn found Elizabeth and Tom back at the winch.",
        "Tuesday dusk found Elizabeth and Tom back at the winch, the very "
        "evening she had first stepped off the coach at Saltmere."))),
    ("london", "setting", chapter(
        "Friday dawn found Elizabeth at her desk in the London survey "
        "office, three hundred miles from Saltmere, while Tom held the end "
        "of the new rope steady beside her and she worked the knot. His "
        "bandaged fingers stayed tucked against his chest, and whenever the "
        "rope bit he winced and said nothing.")),
    ("outline", "outline_gap", (
        "Friday dawn found Elizabeth and Tom walking the sea wall. The tide "
        "was far out, and the mud shone like pewter in the early light. "
        "Tom's bandaged fingers stayed tucked against his chest, and he said "
        "very little. They watched a heron stalk the shallows for a long "
        "time.\n\nElizabeth thought of the letter, and of the chapel, and of "
        "how little she knew about the woman who had written it. By noon the "
        "tide had begun to turn, and they walked back to the landing "
        "without having decided anything at all.")),
    ("pov", "constraint", chapter(OPEN.replace(
        "They had not spoken of the storm since the night it broke.",
        "Tom thought she was the bravest woman he had ever met, and wondered "
        "whether he ought to tell her. They had not spoken of the storm "
        "since the night it broke."))),
    ("tense", "constraint", chapter(
        OPEN, "By mid-morning Dr Anna Cole comes down the jetty with her bag "
              "over her shoulder. She sits on an upturned crate and asks "
              "whether anything has been settled about Marcus. Nothing has. "
              "Tom suggests the chapel, because Marcus loved it. Elizabeth "
              "says the chapel is hers now, and she will not have the "
              "harbourmaster buried anywhere else. Anna writes it in her "
              "notebook: Sunday, at noon, if the tide allows.")),
]

PLANTED_TYPES = sorted({t for _, t, _ in CASES if t})
