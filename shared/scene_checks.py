"""
Deterministic scene checks: who is in a chapter, where the book is set, and
which spelling it uses.

Measured on one seed, two models, chapters 1-5 (November 2026): one model
put a couple at a dinner the outline gave to two others ("James cleared his
throat" in a chapter whose cast was Angela & Pete visiting Stuart & Kirsty),
called a Hampshire village's rain "a heavy, Somerset rain", and wrote
"neighbor" and "organized" in a book the seed says is set in the UK. The
reviewer passed all three. Each is a question code can answer:

- cast: the seed outline says who is in each chapter ("Couples: ...",
  "Characters: ..."); a bible character outside that list who *acts* in the
  scene (speaks, nods, walks in) three or more times is there. Being
  mentioned, remembered or discussed is fine: the other run talked about
  Angela in the chapters she was absent from and never had her act more than
  twice.
- place: the counties the seed (and the bible's setting) name are where the
  book is; narration that names another UK county is flagged. Dialogue is
  left alone ("my sister in Dorset" is a character's business).
- spelling: book.style.spelling: british flags American spellings and words
  ("color", "neighbor", "realize", "sidewalk"). Opt-in: -ize is also correct
  Oxford spelling, and an American book wants the opposite.

Pure functions; the editor runs them as lint (agents/editor.py).
"""

import re

from shared.consistency import _NOT_NAMES, narration

# ------------------------------------------------------------------ cast

# "* **Couples:** Stuart & Kirsty (Guests) visiting James & Chloe (Hosts)."
# The planner folds these lines into the chapter summary as "Couples: ...".
_CAST_LABEL_RE = re.compile(
    r"\b(?:couples?|characters?|cast|present|pov|who|with|featuring)\s*:\s*",
    re.IGNORECASE)
_NEXT_LABEL_RE = re.compile(r"\s(?:\*\*)?[A-Z][A-Za-z ]{1,20}:")

# Something a person does in the room. Wider than the dead-character list
# (consistency._ALIVE_VERBS): a guest who is there nods and clears a throat.
_PRESENT_VERBS = (
    "said|says|asked|replied|answered|shouted|whispered|murmured|muttered|"
    "laughed|smiled|grinned|frowned|sighed|nodded|shrugged|protested|added|"
    "interjected|called|cleared|leaned|leant|reached|glanced|looked|watched|"
    "turned|stood|sat|walked|arrived|entered|poured|raised|lifted|took|put|"
    "held|grabbed|touched|kissed|winked|blushed|flushed|wore|offered|handed"
)
CAST_MIN_ACTIONS = 3     # the absent characters of the measured run: 0-2


_GROUP_ALIAS_RE = re.compile(r"&|\band\b|^the\s", re.IGNORECASE)


def _name_parts(character):
    """The words a character is called by: name parts, and aliases that name
    one person. Group aliases ("Angela & Pete", "The Instigators") are left
    out: with them, Pete nodding put Angela in the room, and every "The ..."
    sentence matched."""
    aliases = [a for a in character.get("aliases") or []
               if not _GROUP_ALIAS_RE.search(str(a).strip())]
    names = [character.get("name") or ""] + aliases
    return [p for n in names for p in re.findall(r"[A-Z][A-Za-z'-]+", str(n))
            if len(p) > 2 and p not in _NOT_NAMES]


def chapter_cast(summary, characters):
    """The bible characters a chapter's outline puts in it, from a
    "Couples:" / "Characters:" / "Cast:" field in its summary. None when the
    summary has no such field, or it names nobody ("All three couples"):
    then nothing is known and nothing is checked."""
    m = _CAST_LABEL_RE.search(summary or "")
    if not m:
        return None
    rest = summary[m.end():]
    end = _NEXT_LABEL_RE.search(rest)
    field = rest[:end.start()] if end else rest
    named = [c["name"] for c in characters
             if c.get("name") and any(re.search(rf"\b{re.escape(p)}\b", field)
                                      for p in _name_parts(c))]
    return named or None


def _present_re(part):
    n = re.escape(part)
    return re.compile(
        rf"\b{n}\b(?![\'\u2019]s)[^.!?;\n\"\u201c\u201d]{{0,30}}?"
        rf"\b(?i:{_PRESENT_VERBS})\b"
        rf"|(?i:\b(?:said|asked|whispered|replied|murmured|added)\s+){n}\b")


def cast_finding(text, cast, characters, min_actions=CAST_MIN_ACTIONS):
    """A "cast" finding when a character the outline leaves out of this
    chapter acts in its narration `min_actions` times or more."""
    if not cast:
        return None
    nar = narration(text)
    intruders = {}
    for c in characters:
        name = c.get("name")
        if not name or name in cast:
            continue
        hits = [m.group(0).strip() for p in _name_parts(c)
                for m in _present_re(p).finditer(nar)]
        if len(hits) >= min_actions:
            intruders[name] = hits
    if not intruders:
        return None
    seen = "; ".join(f"{n}: " + ", ".join(f'"{h}"' for h in hits[:3])
                     for n, hits in intruders.items())
    return {
        "check": "cast",
        "names": sorted(intruders),
        "detail": f"{', '.join(intruders)} {'is' if len(intruders) == 1 else 'are'}"
                  " in this scene (" + seen + "), but the outline puts only "
                  + ", ".join(cast) + " in this chapter. Take them out of the "
                  "scene: they may be mentioned, remembered or talked about, "
                  "but not be present, speak or act.",
    }


# ----------------------------------------------------------------- place

# Counties and their common short forms. Names that are also everyday words
# or first names (Kent, Angus, Down, Moray, Durham's city) are left out or
# kept in a form that cannot be anything else.
UK_COUNTIES = (
    "Bedfordshire", "Berkshire", "Buckinghamshire", "Cambridgeshire",
    "Cheshire", "Cornwall", "Cumbria", "Derbyshire", "Devon", "Dorset",
    "County Durham", "Essex", "Gloucestershire", "Hampshire", "Herefordshire",
    "Hertfordshire", "Isle of Wight", "Lancashire", "Leicestershire",
    "Lincolnshire", "Merseyside", "Middlesex", "Norfolk", "Northamptonshire",
    "Northumberland", "Nottinghamshire", "Oxfordshire", "Rutland",
    "Shropshire", "Somerset", "Staffordshire", "Suffolk", "Surrey", "Sussex",
    "Warwickshire", "Wiltshire", "Worcestershire", "Yorkshire",
    "Anglesey", "Ceredigion", "Carmarthenshire", "Pembrokeshire", "Powys",
    "Gwynedd", "Monmouthshire", "Glamorgan", "Denbighshire", "Flintshire",
    "Aberdeenshire", "Argyll", "Ayrshire", "Dumfriesshire", "Fife",
    "Lanarkshire", "Perthshire", "Midlothian", "Stirlingshire",
    "Inverness-shire", "Sutherland", "Caithness", "Roxburghshire",
    "County Antrim", "County Armagh", "County Down", "County Fermanagh",
    "County Londonderry", "County Tyrone",
)
_COUNTY_RE = re.compile(r"\b(" + "|".join(re.escape(c) for c in UK_COUNTIES)
                        + r")\b")


def seed_places(*texts):
    """The UK counties the seed and the bible's setting name: where the book
    is. Empty when they name none (then no place is checked)."""
    found = []
    for text in texts:
        found += _COUNTY_RE.findall(str(text or ""))
    return list(dict.fromkeys(found))


def place_finding(text, places):
    """A "place" finding for a UK county the narration names that the seed
    does not. `places`: seed_places(...)."""
    if not places:
        return None
    others = {}
    for m in _COUNTY_RE.finditer(narration(text)):
        if m.group(1) not in places:
            start = max(0, m.start() - 40)
            others.setdefault(m.group(1), narration(text)[start:m.end() + 20]
                              .replace("\n", " ").strip())
    if not others:
        return None
    return {
        "check": "place",
        "detail": "the book is set in " + " / ".join(places) + ", but the "
                  "narration names " + "; ".join(
                      f'{c} ("...{q}...")' for c, q in others.items())
                  + ". Where that means the book's own setting, use "
                  + places[0] + "; a place the characters really travel to "
                  "may stay.",
    }


# -------------------------------------------------------------- spelling

# American -> British. Words with a different everyday meaning in British
# English (pants, fall, check, tire, curb, program, gas) are left out: the
# lint must not "fix" a correct word.
_US_TO_UK = {
    "color": "colour", "colors": "colours", "colored": "coloured",
    "colorful": "colourful", "colorless": "colourless", "honor": "honour",
    "honored": "honoured", "honorable": "honourable", "favor": "favour",
    "favors": "favours", "favorite": "favourite", "favorites": "favourites",
    "neighbor": "neighbour", "neighbors": "neighbours",
    "neighborhood": "neighbourhood", "neighborly": "neighbourly",
    "labor": "labour", "flavor": "flavour", "flavors": "flavours",
    "humor": "humour", "rumor": "rumour", "rumors": "rumours",
    "harbor": "harbour", "behavior": "behaviour", "behaviors": "behaviours",
    "odor": "odour", "armor": "armour", "parlor": "parlour",
    "endeavor": "endeavour", "vapor": "vapour", "clamor": "clamour",
    "glamor": "glamour", "savor": "savour", "savored": "savoured",
    "fervor": "fervour", "rigor": "rigour", "valor": "valour",
    "center": "centre", "centers": "centres", "centered": "centred",
    "theater": "theatre", "fiber": "fibre", "somber": "sombre",
    "meager": "meagre", "luster": "lustre", "specter": "spectre",
    "gray": "grey", "grays": "greys", "graying": "greying",
    "mom": "mum", "moms": "mums", "mommy": "mummy",
    "sidewalk": "pavement", "sidewalks": "pavements",
    "apartment": "flat", "gotten": "got", "toward": "towards",
    "faucet": "tap", "diaper": "nappy", "cookie": "biscuit",
    "cookies": "biscuits", "candy": "sweets", "vacation": "holiday",
    "drapes": "curtains", "pajamas": "pyjamas", "skeptical": "sceptical",
    "aluminum": "aluminium", "mustache": "moustache", "jewelry": "jewellery",
    "traveled": "travelled", "traveling": "travelling",
    "traveler": "traveller", "canceled": "cancelled", "modeled": "modelled",
    "fueled": "fuelled", "leveled": "levelled", "marveled": "marvelled",
    "defense": "defence", "offense": "offence", "pretense": "pretence",
    "plow": "plough", "airplane": "aeroplane", "math": "maths",
    "elevator": "lift", "flashlight": "torch", "garbage": "rubbish",
    "mailman": "postman", "cellphone": "mobile", "stroller": "pushchair",
    "parking lot": "car park", "gas station": "petrol station",
    "cell phone": "mobile phone", "trash can": "bin",
}
# -ize verbs: British style here is -ise (Oxford -ize is not this option)
_IZE_STEMS = (
    "real", "organ", "recogn", "apolog", "critic", "emphas", "memor",
    "priorit", "summar", "sympath", "character", "final", "special",
    "stabil", "visual", "util", "minim", "maxim", "agon", "scrutin",
    "fantas", "mesmer", "idol", "tantal", "hypnot", "energ", "patron",
    "symbol", "normal", "harmon", "author", "civil", "jeopard", "monopol",
    "sanit", "categor", "vapor", "trivial",
)
_IZE_RE = re.compile(r"\b((?:" + "|".join(_IZE_STEMS) + r")i)z(e|es|ed|ing|ation|ations)\b",
                     re.IGNORECASE)
_US_RE = re.compile(r"\b(" + "|".join(
    re.escape(w) for w in sorted(_US_TO_UK, key=len, reverse=True)) + r")\b",
    re.IGNORECASE)
SPELLINGS = ("british",)


def _match_case(word, like):
    if like.isupper() and len(like) > 1:
        return word.upper()
    return word[0].upper() + word[1:] if like[:1].isupper() else word


def spelling_hits(text, spelling):
    """[(found, suggested)] in order of first appearance, one per word."""
    if str(spelling or "").strip().lower() not in SPELLINGS:
        return []
    hits = {}
    for m in _US_RE.finditer(text):
        found = m.group(1)
        hits.setdefault(found.lower(), (found, _match_case(
            _US_TO_UK[found.lower()], found)))
    for m in _IZE_RE.finditer(text):
        found = m.group(0)
        hits.setdefault(found.lower(), (found, f"{m.group(1)}s{m.group(2)}"))
    return list(hits.values())


def spelling_finding(text, spelling):
    """A "spelling" finding listing the American forms in a British book."""
    hits = spelling_hits(text, spelling)
    if not hits:
        return None
    return {
        "check": "spelling",
        "detail": "British spelling and vocabulary: "
                  + ", ".join(f'"{f}" -> "{s}"' for f, s in hits[:12])
                  + (f" (+{len(hits) - 12} more)" if len(hits) > 12 else ""),
    }


def spelling_rule(spelling):
    """The prompt line for the writer, reviser and polish pass."""
    if str(spelling or "").strip().lower() == "british":
        return ("Use British spelling and vocabulary: colour, neighbour, "
                "realise, grey, towards, pavement, flat, mum, holiday.")
    return ""
