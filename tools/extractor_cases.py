"""Synthetic chapters with known ground truth for tools/extractor_eval.py.

Invented content only (nothing from any real book). Each case is a short
chapter plus what a correct extraction must contain. The cases include the
traps that matter for continuity:

  - a character who is only MENTIONED must not count as present;
  - a nickname ("Liz") must map to the bible character;
  - a chapter where nothing happens must yield no events;
  - an already-acquainted pair must not produce a first_meeting;
  - a death in someone's BACKSTORY, a near-death, and an attack that nobody
    dies in must not produce a death (a false death kills a living character
    in every later chapter);
  - a death shown only implicitly must still be found;
  - a funeral after the death is already on record is not a new death.

Optional case keys:
  present_optional  names that may or may not be listed (ambiguous: neither
                    rewarded nor penalised)
  prior_deaths      characters recorded as dead by an EARLIER chapter
"""

BIBLE_CHARACTERS = [
    {"name": "Elizabeth Hale", "role": "protagonist", "aliases": ["Liz"],
     "description": "A surveyor."},
    {"name": "Tom Baker", "role": "supporting", "aliases": [],
     "description": "The ferryman."},
    {"name": "Marcus Reed", "role": "supporting", "aliases": [],
     "description": "The harbourmaster."},
    {"name": "Anna Cole", "role": "supporting", "aliases": [],
     "description": "The village doctor."},
]

# who: bible names. type: first_meeting | relationship_change | death |
# injury | secret_revealed
CASES = [
    {
        "number": 1, "title": "The Landing",
        "text": (
            "The ferry landing at Saltmere was little more than a plank jetty "
            "and a lantern on a post. Elizabeth Hale stepped off the late "
            "coach with a single case and a roll of survey maps, and looked "
            "for someone to carry her across the marsh. A man was coiling "
            "rope at the end of the jetty.\n\n"
            "\"You'll be the surveyor,\" he said, not looking up. \"I'm Tom "
            "Baker. I take the post across, and apparently you.\"\n\n"
            "She had never seen him before in her life. They shook hands for "
            "the first time on the damp boards. Tom told her that the "
            "harbourmaster, Marcus Reed, had asked about her twice that week "
            "and would expect her at the chapel by noon tomorrow. Elizabeth "
            "said she would be there. She climbed into the flat boat, the "
            "lantern swung behind them, and Tom poled out into the dark "
            "water without another word."),
        "present": ["Elizabeth Hale", "Tom Baker"],       # Marcus: mentioned only
        "events": [("first_meeting", ["Elizabeth Hale", "Tom Baker"])],
    },
    {
        "number": 2, "title": "The Chapel",
        "text": (
            "The chapel stood knee-deep in the marsh, its door swollen "
            "shut. Tom had left her at the steps and gone back to the "
            "landing, so Elizabeth pushed through alone. Inside, a broad "
            "man in an oilskin was waiting beside the pulpit.\n\n"
            "\"Marcus Reed,\" he said, holding out a hand. \"We have not "
            "met, Miss Hale, though I knew your mother.\"\n\n"
            "She took his hand, startled. Marcus lifted a loose board from "
            "the pulpit and drew out a sealed letter. He told her, "
            "quietly, that her mother had written it thirty years ago, "
            "that no one in Saltmere knew he had found it, and that it "
            "named Elizabeth as the heir to the chapel and the land around "
            "it. Elizabeth read the first line and could not speak."),
        "present": ["Elizabeth Hale", "Marcus Reed"],      # Tom: mentioned only
        "events": [("first_meeting", ["Elizabeth Hale", "Marcus Reed"]),
                   ("secret_revealed", ["Marcus Reed", "Elizabeth Hale"])],
    },
    {
        "number": 3, "title": "Storm",
        "text": (
            "The storm broke at midnight. Liz was helping Tom lash the "
            "ferry down when the winch rope snapped and whipped round his "
            "left hand, crushing two fingers against the post.\n\n"
            "\"Sit,\" she said, and pushed him onto an upturned crate. She "
            "bound the hand with strips torn from her shirt while the "
            "rain hammered the roof of the boathouse. Tom swore, then "
            "laughed, then stopped laughing and looked at her.\n\n"
            "\"You're not what I expected, Liz,\" he said.\n\n"
            "They sat up until the wind dropped. Somewhere before dawn "
            "his good hand found hers, and neither of them moved it. "
            "Whatever had been guarded between them since the landing "
            "had gone, and something warmer had taken its place."),
        "present": ["Elizabeth Hale", "Tom Baker"],
        "events": [("injury", ["Tom Baker"]),
                   ("relationship_change", ["Elizabeth Hale", "Tom Baker"])],
    },
    {
        "number": 4, "title": "The Drowned Man",
        "text": (
            "Tom came back from the landing at first light with his "
            "bandaged hand held against his chest and his face grey. "
            "Liz put down the survey chain.\n\n"
            "\"It's Marcus,\" he said. \"The fishermen found him at the "
            "foot of the chapel steps. He drowned in the night, in the "
            "storm. He's dead, Liz.\"\n\n"
            "She sat down on the boat's thwart. She had known the "
            "harbourmaster for three days, and he had been the only "
            "person in Saltmere who had spoken of her mother. Tom put "
            "his good hand on her shoulder. Neither of them said "
            "anything while the tide came in."),
        "present": ["Elizabeth Hale", "Tom Baker"],        # Marcus: dead, absent
        "events": [("death", ["Marcus Reed"])],
    },
    {
        "number": 5, "title": "Ropes",
        "text": (
            "They spent the morning replacing the winch rope. Tom held "
            "the coil with his good hand and criticised her knots; "
            "Elizabeth ignored him and tied a better one. At noon they "
            "ate bread and cheese on the jetty and argued about whether "
            "the tide table was wrong or the clock. The marsh was flat "
            "and bright and nothing at all happened. By evening the "
            "ferry was rigged and the lantern was lit."),
        "present": ["Elizabeth Hale", "Tom Baker"],
        "events": [],                                       # nothing to extract
    },
    {
        "number": 6, "title": "Pale Water",                  # backstory death
        "text": (
            "Elizabeth walked the sea wall alone at dusk. Her father, "
            "Edmund, had drowned at this very spot when she was nine years "
            "old, and she had not stood here since. She told the gulls the "
            "whole story anyway: how the letter had come to her, how the "
            "chapel had passed to her, how Tom had mended the ferry.\n\n"
            "Out on the water a lamp swung into view. It was Tom, poling "
            "home late, and he lifted a hand. Liz lifted hers, and when the "
            "lamp had gone round the point she walked back along the wall "
            "to the landing."),
        "present": ["Elizabeth Hale", "Tom Baker"],
        "events": [],                      # Edmund died long ago: no death
    },
    {
        "number": 7, "title": "Thin Ice",                    # near-death
        "text": (
            "The marsh had frozen overnight, and Tom, who never believed in "
            "ice, walked out onto it to check the winch post. It held for "
            "ten steps. On the eleventh it opened under him and he dropped "
            "to the chest in black water. Liz went flat on the ice and "
            "crawled, got the boat hook under his arms, and hauled until her "
            "shoulders burned.\n\n"
            "He came out coughing and swearing in two languages. \"I nearly "
            "drowned,\" he said through his teeth. \"I nearly bloody "
            "drowned.\" She wrapped her coat around him and rubbed his arms "
            "until the colour came back. By the time the sun cleared the "
            "reeds he was sitting up with a mug of tea, shaking, a little "
            "ashamed, and entirely alive."),
        "present": ["Elizabeth Hale", "Tom Baker"],
        "events": [],                      # nobody dies, nobody is hurt
    },
    {
        "number": 8, "title": "Dawn Watch",                  # implicit death
        "text": (
            "Marcus Reed had not woken since the storm. Dr Anna Cole had sat "
            "with him through the night, and Elizabeth had sat with her, and "
            "between them they had said very little. At first light the "
            "doctor laid two fingers against his throat and held them there "
            "for a long time. Then she drew the blanket up over his face.\n\n"
            "\"He's gone,\" she said quietly. Elizabeth looked at the "
            "window, where the marsh was turning from grey to gold, and "
            "found that she could not make herself look at the bed."),
        "present": ["Elizabeth Hale", "Anna Cole"],
        "present_optional": ["Marcus Reed"],   # the patient: arguably present
        "events": [("death", ["Marcus Reed"])],
    },
    {
        "number": 9, "title": "The Funeral",                 # already dead
        "prior_deaths": ["Marcus Reed"],
        "text": (
            "They buried Marcus Reed on the high ground above the chapel, "
            "where the wind kept the grass flat. Tom carried the coffin with "
            "one good hand and one bad, refusing help twice. Dr Cole read the "
            "psalm because no one else could, and Elizabeth stood at the "
            "back with the letter folded small in her fist.\n\n"
            "Afterwards the three of them walked down to the landing without "
            "speaking, and Tom lit the lantern even though it was still "
            "light."),
        "present": ["Elizabeth Hale", "Tom Baker", "Anna Cole"],
        "events": [],                      # a funeral is not a new death
    },
    {
        "number": 10, "title": "The Knife",                  # hurt, not dead
        "text": (
            "The stranger came out of the reeds without a sound. Tom saw the "
            "blade a moment too late and turned into it, and it went into "
            "his shoulder instead of Liz's side. She screamed. The man ran, "
            "and was lost in the mist before either of them could follow.\n\n"
            "Tom sat down heavily in the mud, one hand pressed to the wound, "
            "blood running between his fingers. \"It's not deep,\" he said, "
            "and then, less certainly, \"I think it's not deep.\" Liz tore "
            "her scarf in two and packed the cut while he swore. Dr Cole "
            "arrived at a run twenty minutes later, looked once, and said "
            "he would live, but that he had been very lucky."),
        "present": ["Elizabeth Hale", "Tom Baker", "Anna Cole"],
        "events": [("injury", ["Tom Baker"])],      # injured, NOT dead
    },
]
