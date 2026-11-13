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
    "Chapter", "Part", "Yes", "No", "Morning", "Night", "Evening",
    "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday",
    "Sunday", "January", "February", "March", "April", "May", "June", "July",
    "August", "September", "October", "November", "December",
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

_QUOTED = r"\"[^\"]+\"|'[^']+'"
_SEP = r"\s*(?:,\s*or|,|or)\s*"
# Never [use] the word(s) <quoted-or-bare list>
_BANNED_WORDS_RE = re.compile(
    r"[Nn]ever\s+(?:use\s+)?the\s+words?\s+"
    rf"((?:{_QUOTED}|\w+)(?:{_SEP}(?:{_QUOTED}|\w+))*)")
# Never [use] "quoted" [, "quoted" ...]  (bare words are NOT taken here:
# 'Never mention orange' must not ban 'mention')
_BANNED_QUOTED_RE = re.compile(
    r"[Nn]ever\s+(?:use\s+)?"
    rf"((?:{_QUOTED})(?:{_SEP}(?:{_QUOTED}))*)")


def extract_banned_words(constraints):
    """Pull banned words/phrases out of constraint strings.

    Recognizes patterns like:
      Never the words "unhurried", "unrushed", or "exactly".
      Never use the word unhurried.
      Never "black lace".
      Never use "unhurried".
    Unquoted terms are only accepted after "the word(s)".
    """
    banned = []
    for constraint in constraints or []:
        for regex in (_BANNED_WORDS_RE, _BANNED_QUOTED_RE):
            for match in regex.finditer(constraint):
                for q1, q2, bare in re.findall(
                    r'"([^"]+)"|\'([^\']+)\'|(\w+)', match.group(1)
                ):
                    word = (q1 or q2 or bare or "").strip()
                    if not word or word.lower() in _BANNED_STOPWORDS:
                        continue
                    if word not in banned:
                        banned.append(word)
    return banned


def _term_re(term):
    """Whole-word, case-insensitive matcher ('art' must not hit 'heart')."""
    return re.compile(rf"(?<!\w){re.escape(term)}(?!\w)", re.IGNORECASE)


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

# A sentence ends at . ! ? or an ellipsis, optionally followed by the quote or
# bracket that closes it: dialogue ('"Go."  She went.') is two sentences, not
# one long one.
_SENTENCE_END = re.compile(
    r"(?<=[.!?\u2026])\s+|(?<=[.!?\u2026][\"\u201d\u2019')])\s+")


def _sentences(text):
    return [s.strip() for s in _SENTENCE_END.split(text) if s.strip()]


def _phrase_pattern(phrase):
    """`phrase` as a whole-word, case-insensitive pattern: "sigh" must not
    count "sight". (?<!\\w) rather than \\b so a phrase that starts or ends
    with punctuation still matches."""
    return re.compile(rf"(?<!\w){re.escape(phrase.strip())}(?!\w)",
                      re.IGNORECASE)


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
    return len(_phrase_pattern(phrase).findall(text))


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

def _is_name_variant(token, name_word):
    """Plural, possessive or suffixed form of a name ('Stuarts', 'Marks',
    'Stuartson'): a different, legitimate word, not a misspelling."""
    t, n = token.lower(), name_word.lower()
    return len(n) >= 3 and t.startswith(n) and len(t) - len(n) >= 1 \
        and t != n and (t in (n + "s", n + "es") or len(t) - len(n) >= 2)


def _name_findings(text, bible, ignore=()):
    """Flag tokens that are near-misses of bible character names.

    ignore: words never flagged (book.name_lint_ignore), e.g. a real word
    that happens to resemble a character's name."""
    ignored = {str(w).lower() for w in ignore}
    names = [c["name"] for c in (bible.get("characters") or [])]
    known = {word for name in names for word in name.split()}

    findings = []
    token_counts = Counter(re.findall(r"\b[A-Z][a-z]{2,}\b", text))
    for token, count in token_counts.items():
        if token in _NOT_NAMES or token in known or token.lower() in ignored:
            continue
        for name_word in known:
            if _is_name_variant(token, name_word):
                continue
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


# The writer is asked for up to 1.2x the target, so a "too long" threshold
# below that would send back chapters that did what they were told.
MIN_MAX_LENGTH_RATIO = 1.2


def word_count_finding(text, target_words, tolerance=0.8, max_ratio=0):
    """Length check for one chapter against the configured target.

    Args:
        text: chapter body (no heading)
        target_words: book.words_per_chapter from config
        tolerance: acceptable fraction of the target (book.
            word_count_tolerance); chapters below target*tolerance are
            flagged as too short.
        max_ratio: book.max_length_ratio; chapters above target*max_ratio
            are flagged as too long (0 = never). A 27B model asked for
            1,400-1,700 words wrote 1,803-3,189 on every chapter.

    Returns a finding dict or None.
    """
    if not text or not target_words:
        return None
    minimum = int(target_words * tolerance)
    count = word_count(text)
    if count < minimum:
        return {
            "check": "word_count",
            "count": count,
            "minimum": minimum,
            "detail": f"chapter is {count} words; minimum is {minimum} "
                      f"({int(tolerance * 100)}% of the {target_words}-word "
                      "target). Expand with SUBSTANCE - deepen existing "
                      "scenes, add dialogue and specific detail, extend "
                      "beats from the outline. Do NOT pad: no repetition, "
                      "no filler adjectives, no summarised skim, no "
                      "restating what the reader already knows.",
        }
    maximum = max_length(target_words, max_ratio)
    if maximum and count > maximum:
        return {
            "check": "too_long",
            "count": count,
            "maximum": maximum,
            "target": target_words,
            "detail": f"chapter is {count} words; the target is "
                      f"{target_words} and the most is {maximum}. TIGHTEN: "
                      "cut repetition, restated feelings, over-described "
                      "beats and redundant dialogue tags. Keep every scene, "
                      "every event, every character and every fact; do not "
                      "summarise a scene to save words.",
        }
    return None


def max_length(target_words, max_ratio):
    """Most words a chapter may have before it is sent back to be tightened
    (book.max_length_ratio, never under MIN_MAX_LENGTH_RATIO); 0 = no limit."""
    ratio = float(max_ratio or 0)
    if ratio <= 0 or not target_words:
        return 0
    return int(target_words * max(ratio, MIN_MAX_LENGTH_RATIO))


def dropped_names(before, after, names):
    """Character names `before` mentions and `after` no longer does. A
    tightened chapter must keep everyone in it: a revision that cut a
    character's scene to save words has lost story, not padding. A name
    counts as present when any part of it (Mara of "Mara Vance") appears."""
    def present(text, name):
        parts = [p for p in re.findall(r"[A-Za-z][A-Za-z'-]+", name or "")
                 if len(p) > 2 and p[0].isupper()]
        return any(re.search(rf"\b{re.escape(p)}\b", text) for p in parts)
    return [n for n in dict.fromkeys(names)
            if n and present(before, n) and not present(after, n)]


# --------------------------------------------- repetition across chapters

_FUNCTION_WORDS = {
    "the", "a", "an", "and", "or", "but", "of", "in", "on", "at", "to", "for",
    "with", "by", "from", "as", "was", "were", "is", "are", "be", "been",
    "had", "has", "have", "it", "its", "he", "she", "they", "them", "his",
    "her", "their", "i", "you", "we", "me", "my", "your", "that", "this",
    "then", "than", "so", "not", "no", "up", "down", "out", "into", "over",
    "like", "just", "there", "what", "when", "how", "if", "do", "did",
    "said", "all", "one", "back", "off", "would", "could", "she's", "he's",
}


def _tokens(text):
    return re.findall(r"[a-z]+(?:'[a-z]+)?", text.lower())


def _informative(gram, names):
    """A repeated phrase only matters if it carries content: at least three
    words that are neither function words nor character names."""
    return sum(1 for w in gram if w not in _FUNCTION_WORDS
               and w not in names) >= 3


def _name_parts(names):
    return {part.lower() for nm in names for part in re.findall(r"[A-Za-z']+", nm)}


def _grams(tokens, n):
    return [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]


def _spans(flags):
    """Maximal runs of True in `flags` as (start, end) pairs."""
    spans, i = [], 0
    while i < len(flags):
        if flags[i]:
            j = i
            while j < len(flags) and flags[j]:
                j += 1
            spans.append((i, j))
            i = j
        else:
            i += 1
    return spans


def phrase_repeats(text, earlier_texts, names=(), n=6, min_prior=2, limit=5):
    """Phrases in `text` (n or more words) already used at least `min_prior`
    times in earlier chapters, merged into maximal spans.
    Returns [(phrase, prior_count)], longest first."""
    name_set = _name_parts(names)
    prior = Counter()
    for earlier in earlier_texts:
        prior.update(_grams(_tokens(earlier), n))
    tokens = _tokens(text)
    flags, counts = [False] * len(tokens), {}
    for i, gram in enumerate(_grams(tokens, n)):
        if prior.get(gram, 0) >= min_prior and _informative(gram, name_set):
            flags[i:i + n] = [True] * n
            counts[i] = prior[gram]
    found = [(" ".join(tokens[a:b]),
              max(c for k, c in counts.items() if a <= k < b))
             for a, b in _spans(flags)]
    return sorted(found, key=lambda pc: -len(pc[0]))[:limit]


def repeated_phrase_finding(text, earlier_texts, names=(), n=6, min_prior=2):
    """One lint finding listing phrases this chapter reuses from earlier
    ones, or None. The reviser is told to reword them."""
    found = phrase_repeats(text, earlier_texts, names, n, min_prior)
    if not found:
        return None
    listing = "; ".join(f'"{phrase}" (used {count}x before)'
                        for phrase, count in found)
    return {"check": "repeated_phrase",
            "detail": f"{len(found)} phrase(s) already used {min_prior}+ "
                      f"times in earlier chapters; reword them with fresh "
                      f"imagery and phrasing: {listing}"}


def repeated_phrases_in_book(chapter_texts, names=(), n=6, min_total=3,
                             min_chapters=2, limit=8):
    """Phrases used at least `min_total` times across at least
    `min_chapters` chapters. chapter_texts is {number: text}. Returns
    [(phrase, count, chapters)], most frequent first (for the lint report)."""
    name_set = _name_parts(names)
    total, where = Counter(), {}
    token_lists = {k: _tokens(v) for k, v in chapter_texts.items()}
    for k, toks in token_lists.items():
        for gram in _grams(toks, n):
            total[gram] += 1
            where.setdefault(gram, set()).add(k)
    keep = {g for g, c in total.items() if c >= min_total
            and len(where[g]) >= min_chapters and _informative(g, name_set)}
    phrases, chapters_of = Counter(), {}
    for k, toks in token_lists.items():
        flags = [False] * len(toks)
        for i, gram in enumerate(_grams(toks, n)):
            if gram in keep:
                flags[i:i + n] = [True] * n
        for a, b in _spans(flags):
            phrase = " ".join(toks[a:b])
            phrases[phrase] += 1
            chapters_of.setdefault(phrase, set()).add(k)
    ranked = sorted(phrases, key=lambda ph: (-phrases[ph], -len(ph)))
    return [(ph, phrases[ph], sorted(chapters_of[ph])) for ph in ranked[:limit]]


def lint_chapter(number, text, bible, constraints, ignore_names=(),
                 style=None):
    """Run all deterministic checks on one chapter. Returns findings list.

    style: the book.style config dict (see style_findings); None disables the
    style checks, which is the default so existing configs behave identically.
    """
    findings = []

    banned = extract_banned_words(constraints)
    for word in banned:
        hits = [m.start() for m in _term_re(word).finditer(text)]
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

    findings.extend(_name_findings(text, bible, ignore_names))
    if style:
        findings.extend(style_findings(text, style))
    return findings


def lint_book(full_text, bible, constraints, ignore_names=(),
              chapter_texts=None, style=None):
    """Book-scope checks: banned words and volume-level quotas."""
    findings = []
    for word in extract_banned_words(constraints):
        hits = len(_term_re(word).findall(full_text))
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
    findings.extend(_name_findings(full_text, bible, ignore_names))
    if chapter_texts:
        names = [c.get("name", "") for c in bible.get("characters") or []]
        for phrase, count, chapters in repeated_phrases_in_book(
                chapter_texts, names):
            findings.append({
                "check": "repeated_phrase",
                "detail": f'"{phrase}" appears {count}x across chapters '
                          f"{', '.join(map(str, chapters))}"})
    if style:
        findings.extend(style_findings(full_text, style, book_scope=True))
    return findings


# ------------------------------------------------------ polish guard: names

# A capitalised word after a lowercase word or a comma is not sentence-initial,
# so it is a name (or a proper noun): "she told Marcus", "the tower, Veyl".
_MID_SENTENCE_CAP = re.compile(r"(?<=[a-z,;]\s)([A-Z][a-z]{2,})\b")


_WORD_RE = re.compile(r"[A-Za-z]+")
# A run of lowercase letters then a capital inside one word: "unremarkableRE".
# Three lowercase letters first, so "McGregor" and "iPhone" do not match.
_GARBLED_RE = re.compile(r"^[A-Za-z]*[a-z]{3}[A-Z][A-Za-z]*$")


def repair_fused_words(before, after):
    """Split words in `after` that fuse two words written apart in `before`.

    A model rewriting text it was given sometimes drops the space between
    two words it is copying ("against her" -> "againsther"); repetition
    penalties make it likelier. A word is repaired only when it is new (not
    in `before`) and `before` contains exactly that pair of words side by
    side, so the fix is the author's own text. Returns (text, [(fused,
    repaired)])."""
    known = {w.lower() for w in _WORD_RE.findall(before)}
    pairs = {(a.lower(), b.lower()) for a, b in re.findall(
        r"\b([A-Za-z]+)\s+(?=([A-Za-z]+)\b)", before)}
    fixes = {}
    for word in set(_WORD_RE.findall(after)):
        low = word.lower()
        if len(word) < 5 or low in known:
            continue
        for i in range(2, len(word) - 1):
            if (low[:i], low[i:]) in pairs:
                fixes[word] = f"{word[:i]} {word[i:]}"
                break
    text = after
    for fused, repaired in fixes.items():
        text = re.sub(rf"\b{re.escape(fused)}\b", repaired, text)
    return text, sorted(fixes.items())


def garbled_words(before, after):
    """Words new in `after` with a capital in the middle of lowercase
    ("unremarkableRE"): token debris, never a real word the copy should
    have. Sorted, unique."""
    known = {w.lower() for w in _WORD_RE.findall(before)}
    return sorted({w for w in _WORD_RE.findall(after)
                   if w.lower() not in known and _GARBLED_RE.match(w)})


def introduced_names(before, after):
    """Names `after` uses mid-sentence that `before` never contains in any
    case. The polish pass must not change any name or fact, and the reviewer
    does not see the polished text, so a polish that renames a character or
    invents a place is caught here, in code. Recapitalising a word the draft
    already had ("london" -> "London") is not a new name."""
    known = {w.lower() for w in re.findall(r"[A-Za-z]+", before)}
    return sorted({m for m in _MID_SENTENCE_CAP.findall(after)
                   if m.lower() not in known})


# ------------------------------------------------------------- style checks

# Hedges: language that softens a concrete image into a guess. Kept narrow on
# purpose - "as if" is often good prose, so this list is the vague-narration
# subset, not every qualifier. Counted per 1000 words and off by default.
STYLE_HEDGES = (
    "seemed to", "somehow", "a sense of", "couldn't help but",
    "could not help but", "almost as if", "as if to", "some part of",
    "somewhere inside", "something inside", "a kind of", "almost like",
)

# A style check only runs when its option is configured, so an empty style
# section changes nothing. Every threshold is opt-in for that reason: a
# false finding costs a revision round and can push the reviser to over-edit.
STYLE_DEFAULTS = {
    "max_phrase_hits": 1,          # per chapter, per banned phrase
    "max_hedges_per_1000": 8.0,
    "min_sentence_cv": 0.30,
    "min_pov_share": 0.60,
    "min_tense_share": 0.60,
    "min_sentences": 6,            # below this, variance/tense are not judged
}


def sentence_length_cv(text):
    """Coefficient of variation of sentence lengths (stdev / mean).

    Flat, metronomic prose sits near 0.25-0.35; varied prose sits at 0.45+.
    Returns 0.0 when the text is too short to judge."""
    lengths = [len(s.split()) for s in _sentences(text) if len(s.split()) > 2]
    if len(lengths) < 3:
        return 0.0
    mean = sum(lengths) / len(lengths)
    if not mean:
        return 0.0
    variance = sum((n - mean) ** 2 for n in lengths) / len(lengths)
    return round((variance ** 0.5) / mean, 3)


def hedge_count(text):
    return sum(_count_phrase(text, hedge) for hedge in STYLE_HEDGES)


def hedges_per_1000(text):
    words = word_count(text)
    return round(hedge_count(text) * 1000 / words, 2) if words else 0.0


def dialogue_ratio(text):
    """Share of non-empty lines that open with a quote mark."""
    lines = [line for line in text.splitlines() if line.strip()]
    if not lines:
        return 0.0
    quoted = sum(1 for line in lines
                 if line.lstrip()[:1] in ('"', "\u201c", "\u2018", "'"))
    return round(quoted / len(lines), 3)


# Quoted speech, removed before POV and tense are judged: a past-tense,
# third-person narration is full of "I", "you" and "is" inside its dialogue.
# Straight quotes pair within one paragraph; curly quotes pair directly.
_DIALOGUE_RE = re.compile(r"\u201c[^\u201d]*\u201d|\"[^\"\n]*\"")

# Past-tense markers: common irregular pasts plus regular -ed verbs. A bare
# \w+ed counted "need", "indeed", "speed" and "hundred" as past, which tilted
# present-tense books towards "past"; those shapes are excluded.
_PAST_IRREGULAR = (
    "was", "were", "had", "said", "went", "came", "took", "knew", "saw",
    "thought", "felt", "made", "got", "gave", "found", "told", "left", "stood",
    "sat", "ran", "began", "brought", "held", "kept", "turned", "looked",
    "heard", "did", "could", "would",
)
_NOT_PAST_ED = {
    "hundred", "sacred", "naked", "wicked", "kindred", "rugged", "ragged",
    "jagged", "crooked", "shed", "embed", "infrared", "sled", "bled", "wed",
    "bed", "red", "led", "fed",
}
_PRESENT = (
    "is", "are", "am", "has", "does", "says", "walks", "looks", "takes",
    "comes", "goes", "knows", "sees", "feels", "thinks", "stands", "turns",
    "makes", "gets", "tells", "leaves", "sits", "runs", "holds", "hears",
)
_PAST_RE = re.compile(r"\b(?:" + "|".join(_PAST_IRREGULAR) + r")\b")
_PRESENT_RE = re.compile(r"\b(?:" + "|".join(_PRESENT) + r")\b")


def narration(text):
    """`text` with quoted dialogue removed (see _DIALOGUE_RE)."""
    return _DIALOGUE_RE.sub(" ", text)


def _regular_pasts(low):
    return sum(1 for w in re.findall(r"\b[a-z]{2,}ed\b", low)
               if not w.endswith("eed") and w not in _NOT_PAST_ED)


def pov_tense(text):
    """Crude dominant POV ('first'/'third'/'second') and tense
    ('past'/'present'/'mixed') of the NARRATION - a lint signal, not a
    parser. Dialogue is left out, since characters speak in the first person
    and the present tense whatever the book's voice. Reported so a chapter
    that slips POV can be sent back; the share decides confidence."""
    low = narration(text).lower()
    pronouns = {
        "first": len(re.findall(r"\b(?:i|me|my|mine|myself)\b", low)),
        "third": len(re.findall(
            r"\b(?:he|she|they|him|her|them|his|their|himself|herself)\b",
            low)),
        "second": len(re.findall(r"\b(?:you|your|yours|yourself)\b", low)),
    }
    total = sum(pronouns.values())
    pov = max(pronouns, key=pronouns.get) if total else ""
    share = round(pronouns[pov] / total, 3) if total else 0.0

    past = _regular_pasts(low) + len(_PAST_RE.findall(low))
    present = len(_PRESENT_RE.findall(low))
    if past > present * 1.3:
        tense, tense_share = "past", round(past / max(past + present, 1), 3)
    elif present > past * 1.3:
        tense = "present"
        tense_share = round(present / max(past + present, 1), 3)
    else:
        tense, tense_share = "mixed", 0.5
    return pov, share, tense, tense_share


# Narrative person, judged by first-person pronouns in the NARRATION. The
# dominant-pronoun share above is a weak signal for this: a first-person
# narrator talks about "she" and "her" constantly, so a chapter that slipped
# into "I" mid-way still read 69% third person (a real 10-chapter run: chapters
# 1-8 had 0 first-person pronouns per 1000 narration words, the two that
# slipped had 25 and 28). Italic spans are left out too: deep third person
# writes thoughts as *I can't*.
_ITALIC_RE = re.compile(r"\*[^*\n]+\*|(?<!\w)_[^_\n]+_(?!\w)")
_FIRST_PERSON_RE = re.compile(r"\b(?:i|me|my|mine|myself)\b")
FIRST_PERSON_MIN = 10.0    # per 1000 narration words: a first-person chapter
THIRD_PERSON_MAX = 2.0     # ... at most this many: a third-person chapter
POV_SLIP_MIN_HITS = 5      # fewer is a stray letter or quotation, not a slip
POV_MIN_WORDS = 150        # narration shorter than this is not judged
# "third-person limited", "POV: first person" - but not "the first person to
# arrive", which would turn the third-person default into first
_POV_WORDS = (r"(?:pov|point[- ]of[- ]view|narrat\w*|perspective|voice|limited"
              r"|omniscient|close|deep|told|written|tense)")
_STATED_POV_RE = re.compile(
    r"\b(first|second|third)[- ]person[\s,:;)-]{0,3}" + _POV_WORDS + r"\b"
    r"|\b" + _POV_WORDS + r"\b[^.\n]{0,30}?\b(first|second|third)[- ]person\b",
    re.I)


def first_person_rate(text):
    """(first-person pronouns, per 1000 words, narration words) in the
    narration: dialogue and italic thoughts removed."""
    low = _ITALIC_RE.sub(" ", narration(text)).lower()
    words = len(low.split())
    hits = len(_FIRST_PERSON_RE.findall(low))
    return hits, (round(hits * 1000 / words, 1) if words else 0.0), words


def chapter_person(text):
    """'first', 'third', 'mixed', or '' (too short to judge)."""
    hits, rate, words = first_person_rate(text)
    if words < POV_MIN_WORDS:
        return ""
    if rate >= FIRST_PERSON_MIN:
        return "first"
    if rate <= THIRD_PERSON_MAX:
        return "third"
    return "mixed"


def stated_pov(*texts):
    """'first' or 'third' when the brief names one person ("third-person
    limited", "first person"), else None. Several different ones: None."""
    found = {(m.group(1) or m.group(2)).lower() for t in texts if t
             for m in _STATED_POV_RE.finditer(str(t))}
    return found.pop() if len(found) == 1 and found <= {"first", "third"} \
        else None


DEFAULT_POV = "third"     # the house voice: every book unless told otherwise


def book_pov(style=None, stated_in=()):
    """(person, why) the book is narrated in: book.style.pov, else the person
    the brief names ("first-person narrator"), else DEFAULT_POV. A style.pov
    of 'second' is the style lint's to check, so it returns (None, '')."""
    wanted = str((style or {}).get("pov") or "").strip().lower()
    if wanted in ("first", "third"):
        return wanted, "book.style.pov"
    if wanted:
        return None, ""
    stated = stated_pov(*stated_in)
    if stated:
        return stated, "the seed"
    return DEFAULT_POV, "the default"


def pov_slip_finding(text, pov, why=""):
    """A 'pov' finding when the chapter's narration is not in `pov` person
    ('first' or 'third'), quoting where the slip starts; None otherwise."""
    if pov not in ("first", "third"):
        return None
    hits, rate, words = first_person_rate(text)
    if words < POV_MIN_WORDS:
        return None
    if pov == "third" and (hits < POV_SLIP_MIN_HITS or rate <= THIRD_PERSON_MAX):
        return None
    if pov == "first" and rate >= FIRST_PERSON_MIN / 2:
        return None
    where = ""
    if pov == "third":
        paras = [p for p in text.split("\n\n") if p.strip()]
        for i, para in enumerate(paras, 1):
            bare = _ITALIC_RE.sub(" ", narration(para))
            if _FIRST_PERSON_RE.search(bare.lower()):
                first = next((x for x in _sentences(bare)
                              if _FIRST_PERSON_RE.search(x.lower())), bare)
                where = (f" from paragraph {i} of {len(paras)}: "
                         f"\"{' '.join(first.split()[:14])} ...\"")
                break
        what = (f"the narration uses the first person ({hits} I/me/my outside "
                f"dialogue{where})")
    else:
        what = (f"the narration is barely first person ({rate} I/me/my per "
                "1000 words)")
    return {
        "check": "pov",
        "detail": f"{what} but the book is narrated in the {pov} person"
                  + (f" ({why})" if why else "")
                  + "; rewrite the narration in the "
                  f"{pov} person, keeping the dialogue as it is",
    }


def style_metrics(text, style=None):
    """Measured style numbers for one chapter (used by the lint and by
    tools/style_eval.py, so the eval scores what the pipeline enforces)."""
    style = style or {}
    phrases = {phrase: _count_phrase(text, phrase)
               for phrase in (style.get("banned_phrases") or [])
               if str(phrase).strip()}
    pov, pov_share, tense, tense_share = pov_tense(text)
    return {
        "sentences": len(_sentences(text)),
        "sentence_cv": sentence_length_cv(text),
        "hedges_per_1000": hedges_per_1000(text),
        "dialogue_ratio": dialogue_ratio(text),
        "pov": pov,
        "pov_share": pov_share,
        "tense": tense,
        "tense_share": tense_share,
        "phrase_hits": {p: n for p, n in phrases.items() if n},
    }


def _style_number(style, key, default=None):
    """style[key] as a float, or `default` when it is unset or not a number.
    config_schema.fix_types() already repairs book.style at load time; this
    keeps callers that pass a style dict directly (tools/style_eval.py) from
    crashing a lint on a typo."""
    value = style.get(key)
    if value is None or isinstance(value, bool):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def style_rules(style, constraints=()):
    """The checkable rules as prompt lines: what the lint will hold a chapter
    to. The writer and the polish pass were linted against book.style (and
    the polish against the banned words) without ever being told the rules,
    so a polish that reintroduced a banned phrase was simply thrown away.
    `constraints`: also list the banned words and quotas parsed from them
    (callers whose prompt does not already carry the bible's constraints)."""
    lines = []
    banned = extract_banned_words(constraints)
    if banned:
        lines.append("Never use these words: "
                     + ", ".join(f'"{w}"' for w in banned) + ".")
    for quota in extract_quotas(constraints):
        what = {"em_dash": "em dashes",
                "sentence_opening": f'sentences opening with "{quota["phrase"]}"',
                "adverbs": "-ly adverbs"}.get(quota["target"],
                                               f'"{quota["phrase"]}"')
        per = {"book": "in the whole book", "per_1000_words": "per 1000 words",
               }.get(quota["scope"], "per chapter")
        lines.append(f"At most {quota['max']} {what} {per}.")
    if not isinstance(style, dict) or style.get("enabled") is False:
        return lines
    phrases = [p for p in style.get("banned_phrases") or [] if str(p).strip()]
    if phrases:
        limit = int(_style_number(style, "max_phrase_hits",
                                  STYLE_DEFAULTS["max_phrase_hits"]))
        lines.append("Stock phrases to avoid (at most "
                     f"{limit} each per chapter): "
                     + ", ".join(f'"{p}"' for p in phrases) + ".")
    if _style_number(style, "max_hedges_per_1000") is not None:
        lines.append("Few hedges: state what happens rather than what "
                     "\"seemed to\" or \"somehow\" happened.")
    if _style_number(style, "min_sentence_cv") is not None:
        lines.append("Vary sentence length: short sentences among long ones.")
    pov = str(style.get("pov") or "").strip().lower()
    if pov in ("first", "second", "third"):
        lines.append(f"Narrate in the {pov} person.")
    tense = str(style.get("tense") or "").strip().lower()
    if tense in ("past", "present"):
        lines.append(f"Narrate in the {tense} tense.")
    from shared.scene_checks import spelling_rule  # it imports this module
    rule = spelling_rule(style.get("spelling"))
    if rule:
        lines.append(rule)
    return lines


def style_findings(text, style, book_scope=False):
    """Findings for the configured style rules.

    A rule is checked only when its option is present, so `style: {}` (the
    default) produces nothing. `book_scope` limits the checks to the ones
    that make sense summed over a whole book."""
    if not isinstance(style, dict) or style.get("enabled") is False:
        return []
    findings = []
    limit = int(_style_number(style, "max_phrase_hits",
                              STYLE_DEFAULTS["max_phrase_hits"]))
    metrics = style_metrics(text, style)

    for phrase, count in metrics["phrase_hits"].items():
        if count > limit:
            findings.append({
                "check": "banned_phrase",
                "detail": f'stock phrase "{phrase}" appears {count}x '
                          f"(max {limit}); replace it with specific "
                          "imagery from this scene",
            })
    if book_scope:
        return findings  # the remaining rules are per-chapter

    max_hedges = _style_number(style, "max_hedges_per_1000")
    if max_hedges is not None and metrics["hedges_per_1000"] > max_hedges:
        findings.append({
            "check": "hedging",
            "detail": f"{metrics['hedges_per_1000']} hedge phrases per 1000 "
                      f"words (max {max_hedges}); state what happens instead "
                      "of what seemed or felt like it"
                      f" -- {hedge_count(text)} hits from "
                      + ", ".join(f'"{h}"' for h in STYLE_HEDGES[:5]) + " ...",
        })

    min_cv = _style_number(style, "min_sentence_cv")
    if (min_cv is not None
            and metrics["sentences"] >= STYLE_DEFAULTS["min_sentences"]
            and metrics["sentence_cv"] < min_cv):
        findings.append({
            "check": "flat_prose",
            "detail": f"sentence lengths are uniform (CV "
                      f"{metrics['sentence_cv']}, min {min_cv}); vary the "
                      "rhythm - a short sentence after a long one, a "
                      "fragment at the beat that matters",
        })

    wanted_pov = str(style.get("pov") or "").strip().lower()
    min_share = _style_number(style, "min_pov_share",
                              STYLE_DEFAULTS["min_pov_share"])
    if (wanted_pov in ("first", "third", "second")
            and metrics["sentences"] >= STYLE_DEFAULTS["min_sentences"]
            and metrics["pov"] != wanted_pov
            and metrics["pov_share"] >= min_share):
        findings.append({
            "check": "pov",
            "detail": f"chapter reads as {metrics['pov']}-person "
                      f"({int(metrics['pov_share'] * 100)}% of pronouns) "
                      f"but the book is {wanted_pov}-person; align it",
        })
    elif wanted_pov in ("first", "third"):
        # the share misses a chapter that slips part-way (see FIRST_PERSON_MIN)
        slip = pov_slip_finding(text, wanted_pov, "book.style.pov")
        if slip:
            findings.append(slip)

    wanted_tense = str(style.get("tense") or "").strip().lower()
    min_tense = _style_number(style, "min_tense_share",
                              STYLE_DEFAULTS["min_tense_share"])
    if (wanted_tense in ("past", "present")
            and metrics["sentences"] >= STYLE_DEFAULTS["min_sentences"]
            and metrics["tense"] != "mixed"
            and metrics["tense"] != wanted_tense
            and metrics["tense_share"] >= min_tense):
        findings.append({
            "check": "tense",
            "detail": f"chapter reads as {metrics['tense']} tense "
                      f"({int(metrics['tense_share'] * 100)}% confidence) "
                      f"but the book is {wanted_tense} tense; align it",
        })

    from shared.scene_checks import spelling_finding  # it imports this module
    spelling = spelling_finding(text, style.get("spelling"))
    if spelling:
        findings.append(spelling)
    return findings


# ------------------------------------------------- chronology (timeline) checks

# A dead character doing something only a living person can do. Bare
# mentions (memories, grief, dialogue ABOUT them) are fine.
_ALIVE_VERBS = (
    "said|says|asked|replied|answered|shouted|whispered|smiled|laughed|"
    "walked|ran|stood|sat|turned|looked|watched|opened|closed|took|put|"
    "held|grabbed|touched|kissed|wore|arrived|entered|left"
)


def _alive_action_re(name):
    """Regex for `name` doing something only a living person can do.

    The name is matched case-sensitively (a dead 'Rose' must not match 'the
    sun rose'); verbs are case-insensitive. Possessives ("Anna's mother
    walked") are excluded from the verb branch, and the gap between name and
    verb may not cross sentence punctuation, semicolons or dialogue quotes.
    """
    n = re.escape(name)
    return re.compile(
        rf"\b{n}\b(?![\'\u2019]s)[^.!?;\n\"\u201c\u201d]{{0,30}}?"
        rf"\b(?i:{_ALIVE_VERBS})\b"
        rf"|(?i:\b(?:said|asked|whispered|replied)\s+){n}\b"
        rf"|\b{n}[\'\u2019]s\s+(?i:voice|hand|eyes|face|smile)\b"
    )


# Strangers-language between people who have already met.
_FIRST_MEETING_RE = re.compile(
    r"pleased to meet you|have we met|do i know you|"
    r"we(?:'ve| have)n['\u2019]t met|i(?:'m| am) [A-Za-z]+['\u2019]?s? "
    r"(?:wife|husband)|introduc(?:ed|ing) (?:himself|herself|themselves)|"
    r"first time (?:she|he|they) (?:had )?(?:met|seen|spoken to)",
    re.IGNORECASE,
)


def check_chronology(number, text, cumulative, aliases=None):
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
        # a multi-word bible name is also referred to by its first name
        variants = [name] + ([name.split()[0]] if len(name.split()) > 1
                             and len(name.split()[0]) > 2 else [])
        variants += (aliases or {}).get(name, [])  # declared nicknames
        hits = [h for v in variants for h in _alive_action_re(v).findall(text)]
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
            a, b = (part.split()[0].capitalize() for part in key.split("+"))
            if a in present_names and b in present_names:
                findings.append({
                    "check": "already_met",
                    "detail": f"{a} & {b} first met "
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
