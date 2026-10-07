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
    return None


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


def lint_chapter(number, text, bible, constraints, ignore_names=()):
    """Run all deterministic checks on one chapter. Returns findings list."""
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
    return findings


def lint_book(full_text, bible, constraints, ignore_names=(),
              chapter_texts=None):
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
