"""Story-state tracking: the structured chronology used for continuity.

After each chapter is written, the writer agent extracts structured facts
(when/where, who is present, first meetings, relationship changes, deaths,
injuries, secrets revealed). This module merges those per-chapter states
into a cumulative story state, renders it for prompts, and exposes it to
the deterministic checks. Pure functions - unit-testable without a model.
"""

import re

EVENT_TYPES = ("first_meeting", "relationship_change", "death", "injury",
               "departure", "secret_revealed")


# Names that denote the same person (nicknames and spelling variants). Only
# used to match an extracted name to a bible character, never to invent one.
NAME_GROUPS = [
    {"elizabeth", "liz", "lizzie", "lizzy", "beth", "betty", "eliza", "libby"},
    {"katherine", "catherine", "kathryn", "katharine", "kate", "katie", "kath",
     "kathy", "kat", "cathy", "kitty"},
    {"margaret", "maggie", "meg", "peggy", "marge", "greta"},
    {"robert", "rob", "bob", "bobby", "robbie", "bert"},
    {"william", "will", "bill", "billy", "liam"},
    {"richard", "rich", "rick", "ricky", "dick"},
    {"james", "jim", "jimmy", "jamie"},
    {"john", "jack", "johnny", "jon"},
    {"thomas", "tom", "tommy"},
    {"michael", "mike", "mick", "mikey"},
    {"christopher", "chris"},
    {"jennifer", "jen", "jenny"},
    {"jessica", "jess", "jessie"},
    {"alexander", "alex", "xander"},
    {"alexandra", "alex", "lexi", "sandra"},
    {"samuel", "sam", "sammy"},
    {"samantha", "sam", "sammy"},
    {"daniel", "dan", "danny"},
    {"matthew", "matt"},
    {"andrew", "andy", "drew"},
    {"anthony", "tony"},
    {"edward", "ed", "eddie", "ted", "teddy"},
    {"stephen", "steven", "steve"},
    {"stuart", "stewart", "stu"},
    {"patricia", "pat", "patty", "trish"},
    {"deborah", "debora", "deb", "debbie"},
    {"susan", "sue", "susie", "suzy"},
    {"rebecca", "becky", "becca"},
    {"victoria", "vicky", "vicki", "tori"},
    {"nicholas", "nick", "nicky"},
    {"joseph", "joe", "joey"},
    {"david", "dave", "davey"},
    {"charles", "charlie", "chuck"},
    {"benjamin", "ben", "benny"},
    {"timothy", "tim", "timmy"},
    {"carole", "carol", "caroline"},
    {"theodore", "theo", "ted"},
    {"abigail", "abby", "abbie"},
    {"amelia", "amy", "mel"},
    {"anne", "ann", "annie", "anna"},
    {"emily", "em", "emmy"},
    {"frances", "fran", "frankie"},
    {"jonathan", "jon", "jonny"},
    {"joanna", "joan", "jo"},
    {"nathaniel", "nate", "nat"},
    {"philip", "phillip", "phil"},
    {"rachel", "rach"},
    {"sarah", "sara", "sally"},
    {"vincent", "vince", "vinny"},
]


def _same_name_group(a, b):
    a, b = a.lower(), b.lower()
    return a == b or any(a in g and b in g for g in NAME_GROUPS)


# Titles that are not part of a name: "Dr Cole" is Anna Cole, "Miss Hale" is
# Elizabeth Hale. Stripped before matching.
TITLES = {"dr", "doctor", "mr", "mrs", "ms", "miss", "mx", "sir", "dame",
          "lady", "lord", "captain", "capt", "professor", "prof", "father",
          "fr", "rev", "reverend", "constable", "sergeant", "sgt",
          "inspector", "officer", "madam", "master", "colonel", "major"}


def strip_title(name):
    """'Dr. Cole' -> 'Cole' (unchanged when the title is the whole name or
    the name has no title)."""
    words = str(name).strip().split()
    while len(words) > 1 and words[0].lower().rstrip(".") in TITLES:
        words.pop(0)
    return " ".join(words)


def canonical_name(name, canon, aliases=None):
    """Map an extracted name onto the bible's spelling when that is
    unambiguous: exact (any case), a first name, a surname, or the bible name
    plus extra words. Anything unmatched or ambiguous is returned unchanged,
    so a new character is never forced onto an existing one."""
    original = str(name).strip()
    name = strip_title(original)          # match without the title...
    if not name or not canon:
        return original
    for c in canon:
        if c.lower() == name.lower():
            return c
    declared = {a.lower(): c for c, al in (aliases or {}).items()
                for a in al}                     # nicknames from the bible
    if name.lower() in declared and declared[name.lower()] in canon:
        return declared[name.lower()]

    def words(s):
        return re.findall(r"[a-z']+", s.lower())

    nw = words(name)
    matches = [c for c in canon if nw and words(c) and (
        nw[0] == words(c)[0] or set(nw) <= set(words(c))
        or set(words(c)) <= set(nw))]
    if len(matches) != 1:
        # a known nickname ('Liz' for 'Elizabeth'), only if it picks ONE person
        matches = [c for c in canon if nw and words(c)
                   and _same_name_group(nw[0], words(c)[0])]
    return matches[0] if len(matches) == 1 else original  # ...keep it if new


def as_names(value):
    """A clean list of names from whatever a model sent for a names field.

    Models, small ones especially, are loose about types: `"who": "Tom"` or
    `"who": "Tom and Liz"` instead of a list. Iterating that string would
    create one single-letter "character" per letter, so a string is split on
    commas, semicolons, '&', '+' and the word 'and'. List items may be strings
    or {"name": ...} objects; anything else is ignored.
    """
    if value is None:
        return []
    if isinstance(value, str):
        value = re.split(r"\s*(?:,|;|&|\+|\band\b)\s*", value)
    elif isinstance(value, dict):
        value = [value]
    elif not isinstance(value, (list, tuple, set)):
        return []
    names = []
    for item in value:
        if isinstance(item, dict):
            item = item.get("name")
        if isinstance(item, (str, int, float)) and str(item).strip():
            names.append(str(item).strip())
    return names


def normalize_state(number, title, data, canon=None, aliases=None):
    """Coerce extracted JSON into a validated per-chapter state dict.

    canon: the bible's character names; aliases: {name: [nicknames]} from the
    bible. Extracted names are snapped to them so deaths, meetings and
    relationships are tracked under one spelling.
    """
    def fix(names):
        seen, out = set(), []
        for n in names:
            n = canonical_name(n, canon, aliases)
            if n.lower() not in seen:
                seen.add(n.lower())
                out.append(n)
        return out

    events = []
    raw_events = data.get("events")
    for ev in raw_events if isinstance(raw_events, list) else []:
        if not isinstance(ev, dict):
            continue
        etype = str(ev.get("type", "")).strip()
        if etype not in EVENT_TYPES:
            continue
        who = fix(as_names(ev.get("who")))
        if not who:
            continue
        events.append({"type": etype, "who": who,
                       "detail": str(ev.get("detail", "")).strip()})
    return {
        "number": number,
        "title": title,
        "summary": str(data.get("summary", "")).strip(),
        "time": str(data.get("time", "")).strip(),
        "location": str(data.get("location", "")).strip(),
        "present": fix(as_names(data.get("present"))),
        "events": events,
    }


def minimal_state(number, title, summary_text):
    """Fallback state when extraction fails: summary only, no facts."""
    return {"number": number, "title": title, "summary": summary_text,
            "time": "", "location": "", "present": [], "events": []}


def pair_key(a, b):
    """Stable key for a pair of names."""
    return "+".join(sorted([a.lower(), b.lower()]))


def merge_states(chronology, upto=None):
    """Merge per-chapter states into a cumulative story state.

    Args:
        chronology: {number: state} dict
        upto: only include chapters with number < upto (i.e. strictly
              before the chapter being reviewed); None = all

    Returns {"alive", "dead", "met_pairs", "relationships", "timeline",
             "secrets", "injuries"}.
    """
    dead = {}           # name -> death chapter
    met = {}            # pair_key -> first chapter
    relationships = {}  # pair_key -> (detail, chapter)
    secrets = []        # {"who", "detail", "chapter"}
    injuries = []       # {"who", "detail", "chapter"}
    timeline = []       # {"chapter", "time", "location"}
    seen_names = set()

    def register_met(who, n):
        for i in range(len(who)):
            for j in range(i + 1, len(who)):
                key = pair_key(who[i], who[j])
                if key not in met:
                    met[key] = n

    for n in sorted(k for k in chronology if upto is None or k < upto):
        state = chronology[n]
        for name in state.get("present", []):
            seen_names.add(name)
        if state.get("time") or state.get("location"):
            timeline.append({"chapter": n, "time": state.get("time", ""),
                             "location": state.get("location", "")})
        for ev in state.get("events", []):
            who = ev["who"]
            detail = ev.get("detail", "")
            if ev["type"] == "death":
                for name in who:
                    dead.setdefault(name, n)   # the FIRST death chapter stands
            elif len(who) >= 2:
                # any two-person event means they have met
                register_met(who, n)
                if ev["type"] == "relationship_change":
                    for i in range(len(who)):
                        for j in range(i + 1, len(who)):
                            relationships[pair_key(who[i], who[j])] = (detail, n)
            if ev["type"] == "secret_revealed":
                secrets.append({"who": who, "detail": detail, "chapter": n})
            elif ev["type"] == "injury":
                injuries.append({"who": who, "detail": detail, "chapter": n})

    alive = sorted(seen_names - set(dead))
    return {"alive": alive, "dead": dead, "met_pairs": met,
            "relationships": relationships, "timeline": timeline,
            "secrets": secrets, "injuries": injuries}


def _first_sentence(text, limit=220):
    """First sentence of `text`, capped at `limit` characters."""
    text = " ".join(text.split())
    m = re.search(r"(?<=[.!?])\s", text)
    sentence = text[:m.start()] if m else text
    if len(sentence) > limit:
        sentence = sentence[:limit].rsplit(" ", 1)[0] + "..."
    return sentence


def render_story_so_far(summaries, before, window=8):
    """Rolling recap of chapters strictly before `before`.

    The last `window` chapters get their full summary; older ones are cut to
    their first sentence so the prompt stays bounded on long books.
    `summaries` is {chapter_number: text}; returns "" when nothing precedes.
    """
    prior = sorted(k for k in summaries if k < before and summaries.get(k))
    recent = set(prior[-window:]) if window and window > 0 else set(prior)
    return "\n\n".join(
        f"Chapter {k}: " + (summaries[k] if k in recent
                           else _first_sentence(summaries[k]))
        for k in prior)


def pretty_pair(key):
    """'lisa+stuart' -> 'Lisa & Stuart' for display."""
    return " & ".join(part.title() for part in key.split("+"))


def render_story_facts(cumulative):
    """Render the cumulative state as compact text for prompts."""
    lines = []
    if cumulative["timeline"]:
        lines.append("Timeline so far: "
                     + "; ".join(f"Ch{t['chapter']}: {t['time'] or '?'}"
                                 f" @ {t['location'] or '?'}"
                                 for t in cumulative["timeline"]))
    if cumulative["dead"]:
        deaths = ", ".join(f"{name} (died Ch{n})"
                           for name, n in cumulative["dead"].items())
        lines.append(f"DEAD (must not act alive later): {deaths}")
    if cumulative["alive"]:
        lines.append(f"Alive/known so far: {', '.join(cumulative['alive'])}")
    if cumulative["met_pairs"]:
        met = ", ".join(f"{pretty_pair(key)} (Ch{n})"
                        for key, n in sorted(
                            cumulative["met_pairs"].items(),
                            key=lambda kv: kv[1]))
        lines.append(f"Have already met (do NOT play as strangers): {met}")
    if cumulative["relationships"]:
        rel = ", ".join(f"{pretty_pair(key)}: {detail} (since Ch{n})"
                        for key, (detail, n) in sorted(
                            cumulative["relationships"].items(),
                            key=lambda kv: kv[1][1]))
        lines.append(f"Relationships (state must not regress): {rel}")
    if cumulative["secrets"]:
        sec = "; ".join(f"Ch{s['chapter']}: {' & '.join(s['who'])} - "
                        f"{s['detail']}" for s in cumulative["secrets"])
        lines.append(f"Secrets revealed (characters may now know): {sec}")
    if cumulative["injuries"]:
        inj = "; ".join(f"Ch{i['chapter']}: {' & '.join(i['who'])} - "
                        f"{i['detail']}" for i in cumulative["injuries"])
        lines.append(f"Injuries (must still affect the character): {inj}")
    return "\n".join(lines) if lines else "(first chapter - no history yet)"
