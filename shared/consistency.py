"""Deterministic consistency checks (the machine half of the review step).

Parses author constraints into machine-checkable rules and lints chapter
text against them: banned words, countable quotas (em-dashes, adverbs,
sentence-opening words, quoted phrases), and character-name near-misses.
Pure functions - no LLM, fully unit-testable.
"""

import re
from collections import Counter
from difflib import SequenceMatcher

# tokens that look like names but never are
_NOT_NAMES = {
    "The", "These", "Those", "She", "He", "They", "It", "But", "And",
    "His", "Her", "When", "What", "Then", "There", "This", "That",
    "Chapter", "Part", "Yes", "No", "Wednesday", "Thursday", "Tuesday",
    "Morning", "Night", "Cedar", "Rooms",
}

# words never allowed as 'banned words' when unquoted (from phrases like
# 'never in dialogue', 'never use the')
_BANNED_STOPWORDS = {
    "in", "the", "a", "an", "to", "of", "or", "and", "never", "use",
    "words", "word", "dialogue", "prose", "scene", "narration", "mid",
    "only", "when", "during", "before", "after", "with", "for", "any",
}

_NOT_ADVERBS = {
    "only", "family", "reply", "supply", "early", "holy", "ugly", "july",
    "italy", "anomaly", "homily", "lily", "silly", "filly", "holly", "molly",
    "ply", "imply", "multiply", "apply", "belly", "jelly", "rally", "ally",
    "tally", "fully", "worldly", "goodly", "likely", "timely", "costly",
}

_ONCE_WORDS = {"once": 1, "twice": 2}


# ----------------------------------------------------------- rule extraction

def extract_banned_words(constraints):
    """Pull banned words/phrases out of constraint strings.

    Recognizes patterns like:
      Never the words "unhurried", "unrushed", or "exactly".
      Never "black lace".
      Never use "unhurried".
    """
    banned = []
    for constraint in constraints or []:
        for match in re.finditer(
            r"[Nn]ever\s+(?:use\s+)?(?:the\s+words?\s+)?"
            r"((?:\"[^\"]+\"|'[^']+'|\w+)"
            r"(?:\s*(?:,\s*or|,|or)\s*(?:\"[^\"]+\"|'[^']+'|\w+))*)",
            constraint,
        ):
            for q1, q2, bare in re.findall(
                r'"([^"]+)"|\'([^\']+)\'|(\w+)', match.group(1)
            ):
                word = (q1 or q2 or bare or "").strip()
                if not word:
                    continue
                if word.lower() in _BANNED_STOPWORDS:
                    continue
                if word.lower() in ("never", "the", "words", "or", "use"):
                    continue
                if word not in banned:
                    banned.append(word)
    return banned


def extract_quotas(constraints):
    """Pull countable quotas out of constraint strings.

    Recognized forms (max may be a number or once/twice):
      Em-dash rare - max 8 per scene
      Sentence-opening "And" max 5 per scene
      Adverbs max 8 per 1000 words
      "the thing" max 6
      "collarbone" max once per volume

    Returns a list of {target, phrase, max, scope} where target is one of
    'em_dash', 'sentence_opening', 'adverbs', 'phrase'; scope is
    'chapter' (default), 'book', or 'per_1000_words'.
    """
    quotas = []
    for constraint in constraints or []:
        for m in re.finditer(
            r"max\s+(\d+|once|twice)\s*"
            r"(?:per\s+([a-z0-9 ]+?))?(?=[,.;)]|$)",
            constraint, re.IGNORECASE,
        ):
            maximum = _ONCE_WORDS.get(m.group(1), m.group(1))
            maximum = int(maximum)
            scope = (m.group(2) or "").strip().lower()

            # subject: the text between the last separator and 'max'
            before = constraint[:m.start()].rstrip(" \u2014-:")
            cut = max(before.rfind(sep) for sep in ".;\u2014")
            subject = before[cut + 1:].strip() if cut >= 0 else before
            quoted = re.search(r'"([^"]+)"|\'([^\']+)\'', subject)
            quoted_text = (quoted.group(1) or quoted.group(2)) if quoted else None
            low = subject.lower()

            if "volume" in scope or "book" in scope:
                scope_key = "book"
            elif "1000" in scope:
                scope_key = "per_1000_words"
            else:
                scope_key = "chapter"  # scene/etc. approximated per chapter

            if low.startswith("em-dash") or low.startswith("em dash"):
                target, key_phrase = "em_dash", "\u2014"
            elif low.startswith("sentence-opening") or \
                    low.startswith("sentence opening"):
                target = "sentence_opening"
                key_phrase = quoted_text or "And"
            elif low.startswith("adverb"):
                target, key_phrase = "adverbs", "adverbs"
            elif quoted_text:
                target, key_phrase = "phrase", quoted_text
            else:
                continue  # not something we can count deterministically

            quotas.append({"target": target, "phrase": key_phrase,
                           "max": maximum, "scope": scope_key})
    return quotas


# --------------------------------------------------------------- counting

def _sentences(text):
    return [s.strip() for s in re.split(r"(?<=[.!?\u2026])\s+", text) if s.strip()]


def _count_em_dash(text):
    return text.count("\u2014")


def _count_sentence_opening(text, word):
    word = word.strip()
    pattern = re.compile(rf'^(?:["\u201c(]*)(?:{re.escape(word)})\b',
                         re.IGNORECASE)
    return sum(1 for s in _sentences(text) if pattern.match(s))


def _count_adverbs(text):
    words = re.findall(r"[A-Za-z]+", text.lower())
    return sum(1 for w in words
               if w.endswith("ly") and len(w) > 4 and w not in _NOT_ADVERBS)


def _count_phrase(text, phrase):
    return len(re.findall(re.escape(phrase), text, re.IGNORECASE))


def _check_quota(quota, text):
    """Return (count, detail) for one quota against one text."""
    target = quota["target"]
    if target == "em_dash":
        count = _count_em_dash(text)
    elif target == "sentence_opening":
        count = _count_sentence_opening(text, quota["phrase"])
    elif target == "adverbs":
        count = _count_adverbs(text)
        words = len(text.split())
        if quota["scope"] == "per_1000_words":
            if words < 250:
                return 0, f"adverbs: {count} in {words} words " \
                          f"(too short to judge per-1000 quota)"
            normalized = round(count * 1000 / words)
            return normalized, f"adverbs: {count} in {words} words " \
                               f"(~{normalized} per 1000, max {quota['max']})"
        return count, f"adverbs: {count} (max {quota['max']})"
    else:
        count = _count_phrase(text, quota["phrase"])
    return count, f"{quota['phrase']}: {count} (max {quota['max']})"


# ------------------------------------------------------------------ linting

def _name_findings(text, bible):
    """Flag tokens that are near-misses of bible character names."""
    names = [c["name"] for c in (bible.get("characters") or [])]
    known = {word for name in names for word in name.split()}
    known_lower = {w.lower() for w in known}

    findings = []
    token_counts = Counter(re.findall(r"\b[A-Z][a-z]{2,}\b", text))
    for token, count in token_counts.items():
        if token in _NOT_NAMES or token in known:
            continue
        for name_word in known:
            # require a shared prefix so determiners/plurals never match
            prefix = 0
            for a, b in zip(token.lower(), name_word.lower()):
                if a != b:
                    break
                prefix += 1
            if prefix < 3:
                continue
            ratio = SequenceMatcher(None, token.lower(),
                                    name_word.lower()).ratio()
            if ratio >= 0.75:
                findings.append({
                    "check": "name_mismatch",
                    "detail": f"'{token}' appears {count}x - did you mean "
                              f"'{name_word}'?",
                })
                break
    return findings


def word_count(text):
    """Word count using `wc -w` semantics: maximal runs of non-whitespace
    characters. str.split() with no arguments is exactly equivalent."""
    return len(text.split())


def word_count_finding(text, target_words, tolerance=0.8):
    """Length check for one chapter against the configured target.

    Args:
        text: chapter body (no heading)
        target_words: book.words_per_chapter from config
        tolerance: acceptable fraction of the target (book.
            word_count_tolerance); chapters below target*tolerance are
            flagged as too short.

    Returns a finding dict or None.
    """
    if not text or not target_words:
        return None
    minimum = int(target_words * tolerance)
    count = word_count(text)
    if count < minimum:
        return {
            "check": "word_count",
            "detail": f"chapter is {count} words; minimum is {minimum} "
                      f"({int(tolerance * 100)}% of the {target_words}-word "
                      "target). Expand with SUBSTANCE - deepen existing "
                      "scenes, add dialogue and specific detail, extend "
                      "beats from the outline. Do NOT pad: no repetition, "
                      "no filler adjectives, no summarised skim, no "
                      "restating what the reader already knows.",
        }
    return None


def lint_chapter(number, text, bible, constraints):
    """Run all deterministic checks on one chapter. Returns findings list."""
    findings = []

    banned = extract_banned_words(constraints)
    for word in banned:
        hits = [m.start() for m in re.finditer(re.escape(word), text,
                                               re.IGNORECASE)]
        if hits:
            findings.append({
                "check": "banned_word",
                "detail": f"banned word '{word}' appears {len(hits)}x",
            })

    for quota in extract_quotas(constraints):
        if quota["scope"] == "book":
            continue  # checked on the full book, not per chapter
        count, detail = _check_quota(quota, text)
        if count > quota["max"]:
            findings.append({"check": "quota", "detail": detail})

    findings.extend(_name_findings(text, bible))
    return findings


def lint_book(full_text, bible, constraints):
    """Book-scope checks: banned words and volume-level quotas."""
    findings = []
    for word in extract_banned_words(constraints):
        hits = len(re.findall(re.escape(word), full_text, re.IGNORECASE))
        if hits:
            findings.append({
                "check": "banned_word",
                "detail": f"banned word '{word}' appears {hits}x in the book",
            })
    for quota in extract_quotas(constraints):
        if quota["scope"] != "book":
            continue
        count, detail = _check_quota(quota, full_text)
        if count > quota["max"]:
            findings.append({"check": "quota", "detail": detail})
    findings.extend(_name_findings(full_text, bible))
    return findings


# ------------------------------------------------- chronology (timeline) checks

# A dead character doing something only a living person can do. Bare
# mentions (memories, grief, dialogue ABOUT them) are fine.
_ALIVE_ACTION_RE = re.compile(
    r"\b({name})\b[^.!?\n]{{0,40}}?"
    r"\b(said|says|asked|replied|answered|shouted|whispered|smiled|laughed|"
    r"walked|ran|stood|sat|turned|looked|watched|opened|closed|took|put|"
    r"held|grabbed|touched|kissed|wore|arrived|entered|left)\b"
    r"|\b(said|asked|whispered|replied)\s+{name}\b"
    r"|\b{name}'s\s+(voice|hand|eyes|face|smile)\b",
    re.IGNORECASE,
)

# Strangers-language between people who have already met.
_FIRST_MEETING_RE = re.compile(
    r"pleased to meet you|have we met|do i know you|"
    r"we(?:'ve| have)n['\u2019]t met|i(?:'m| am) [A-Za-z]+['\u2019]?s? "
    r"(?:wife|husband)|introduc(?:ed|ing) (?:himself|herself|themselves)|"
    r"first time (?:she|he|they) (?:had )?(?:met|seen|spoken to)",
    re.IGNORECASE,
)


def check_chronology(number, text, cumulative):
    """Deterministic timeline checks for chapter `number` against the
    cumulative state from all EARLIER chapters.

    Flags:
      - dead characters performing living actions after their death chapter
      - 'first meeting' language between characters who already met

    Returns a findings list (same shape as lint findings).
    """
    findings = []

    for name, died_ch in (cumulative.get("dead") or {}).items():
        if number <= died_ch:
            continue  # death happens here or later; fine
        pattern = _ALIVE_ACTION_RE.pattern.replace("{name}", re.escape(name))
        hits = re.findall(pattern, text, re.IGNORECASE)
        if hits:
            findings.append({
                "check": "dead_character",
                "detail": f"{name} died in Ch{died_ch} but appears alive in "
                          f"Ch{number} ({len(hits)} living-action refs) - "
                          "rewrite as memory/dialogue-about, or remove",
            })

    met_pairs = cumulative.get("met_pairs") or {}
    first_meeting_hits = _FIRST_MEETING_RE.findall(text)
    if met_pairs and first_meeting_hits:
        # only flag if two people who already met are both on page
        present_names = set(re.findall(r"\b[A-Z][a-z]{2,}\b", text))
        for key, met_ch in met_pairs.items():
            a, b = key.split("+")
            if a.capitalize() in present_names and b.capitalize() in present_names:
                findings.append({
                    "check": "already_met",
                    "detail": f"{a.capitalize()} & {b.capitalize()} first met "
                              f"in Ch{met_ch}, but Ch{number} uses "
                              "first-meeting/stranger language between them",
                })
                break  # one flag is enough to trigger a revision
    return findings


def format_findings(lint, issues):
    """Render lint + LLM review findings as markdown for the reviser."""
    lines = []
    if lint:
        lines.append("MACHINE LINT (deterministic - fix these exactly):")
        lines += [f"- [lint] {f['detail']}" for f in lint]
    if issues:
        lines.append("REVIEWER NOTES (address each):")
        lines += [f"- [{i.get('type', 'quality')}] {i.get('description', '')}"
                  + (f" -> {i['fix']}" if i.get("fix") else "")
                  for i in issues]
    return "\n".join(lines)
