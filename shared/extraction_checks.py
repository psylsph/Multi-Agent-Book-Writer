"""Deterministic cross-checks on an extractor's output (pure, no LLM).

The extractor is a language model reading a chapter, and it makes two kinds
of mistake that matter for continuity:

  - it names people who are not in the chapter, or who are only mentioned;
  - it misses a death, or declares a living character dead.

These helpers find the evidence that settles such cases in the chapter text
itself: which names really occur, and where death language sits next to a
character's name. agents/writer.py uses them to drop ungrounded names and to
decide when a focused second question is worth asking.
"""

import re

from shared.story_state import NAME_GROUPS

# Death language. Deliberately broad (it only triggers a cheap confirmation
# question, never a decision by itself) but without words like "body" or "gone"
# on their own, which appear in nearly every chapter. It includes the common
# euphemisms: a death shown only as "He's gone" or a hand on a pulse must still
# reach the confirmation step.
_DEATH_RE = re.compile(
    r"\b(?:died|dies|dying|dead|death|deaths|killed|kills|murdered|slain|"
    r"drowned|drowning|perished|passed away|passed on|lifeless|corpse|"
    r"funeral|buried|burial|laid (?:[a-z']+ ){0,3}to rest|fatal|fatally|stopped breathing|"
    r"not breathing|no longer breathing|last breath|breathed (?:his|her|their) "
    r"last|did not survive|didn't survive|found dead|never woke|"
    r"(?:did not|didn't) wake|pulse|"
    r"(?:he|she|they)(?:'s|\u2019s| is| has| was| are|'re) gone)\b",
    re.IGNORECASE)

# Short forms that are also ordinary words: never counted as a mention of a
# character ("will" is not William).
_COMMON_WORD_NICKNAMES = {
    "will", "bill", "rob", "pat", "sue", "rich", "jack", "bob", "bert", "ed",
    "em", "jo", "mel", "drew", "nat", "fran", "may", "sandra", "kitty",
}


# Events that are about TWO people; with one left they no longer say anything.
PAIR_EVENTS = {"first_meeting", "relationship_change"}


def name_variants(name, aliases=()):
    """Lower-case strings that count as a mention of `name` in a text: the
    full name, each word of it (3+ letters), declared aliases, and the
    common short forms of the first name (Liz for Elizabeth)."""
    name = re.sub(r"\(.*?\)", " ", str(name)).strip()
    words = re.findall(r"[A-Za-z][A-Za-z'’-]*", name)
    found = {name.lower()} if name else set()
    found.update(w.lower() for w in words if len(w) >= 3)
    found.update(str(a).strip().lower() for a in aliases if str(a).strip())
    if words:
        first = words[0].lower()
        for group in NAME_GROUPS:
            if first in group:
                found.update(n for n in group
                             if n not in _COMMON_WORD_NICKNAMES)
    found.discard("")
    return found


def _pattern(variants):
    """Whole-word, case-insensitive alternation, longest variant first."""
    ordered = sorted(variants, key=len, reverse=True)
    return re.compile(r"(?<!\w)(?:" + "|".join(map(re.escape, ordered))
                      + r")(?!\w)", re.IGNORECASE) if ordered else None


def mentions(text, variants):
    """True when any of `variants` occurs in `text` as a whole word."""
    pattern = _pattern(variants)
    return bool(pattern and pattern.search(text))


def ground_state(state, text, variants_by_name):
    """Drop every name the chapter text never mentions.

    A name that does not occur in the chapter (by full name, a word of it, an
    alias or a short form) was invented by the model; keeping it would put a
    ghost in the story state. Events left with nobody are dropped too (a meeting or relationship change
    needs two people).
    variants_by_name maps bible names to their variants; other names are
    checked by their own words. Returns (new_state, dropped_names).
    """
    dropped = []

    def keep(name):
        variants = variants_by_name.get(name) or name_variants(name)
        if mentions(text, variants):
            return True
        if name not in dropped:
            dropped.append(name)
        return False

    events = []
    for event in state["events"]:
        who = [n for n in event["who"] if keep(n)]
        needed = 2 if event["type"] in PAIR_EVENTS else 1
        if len(who) >= needed:
            events.append({**event, "who": who})
    out = {**state, "present": [n for n in state["present"] if keep(n)],
           "events": events}
    return out, dropped


def death_hints(text, variants_by_name, context=600, limit=3):
    """Where death language is, and which characters it might concern.

    Each death word gets a snippet of `context` characters either side
    (overlapping snippets merge; at most `limit` are kept). Returns
    {name: [snippet, ...]} for every character NAMED inside a snippet. The
    snippet is deliberately wide: a death is often written with a pronoun
    ("He's gone") and the name sits a few sentences earlier; the model that
    is shown the snippet resolves who it means.
    """
    spans = sorted((max(0, m.start() - context),
                    min(len(text), m.end() + context))
                   for m in _DEATH_RE.finditer(text))
    merged = []
    for start, end in spans:
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    patterns = {name: _pattern(v) for name, v in variants_by_name.items()}
    hints = {}
    for start, end in merged[:limit]:
        snippet = text[start:end].strip()
        for name, pattern in patterns.items():
            if pattern and pattern.search(snippet):
                hints.setdefault(name, []).append(snippet)
    return hints
