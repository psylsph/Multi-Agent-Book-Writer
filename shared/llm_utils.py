"""Parsing and cleanup helpers for LLM output and seed prompts.

Pure functions, unit-testable without a model.
"""

import json
import re

# Seed sections the architect already maps to bible keys; anything else in
# the seed is treated as author notes (explicitness, style rules, ...).
KNOWN_SECTIONS = ("premise", "character", "world", "setting", "tone",
                  "style", "outline", "constraint", "synopsis", "plot")

_FENCE_RE = re.compile(r"^\s*```[\w-]*\s*$", re.MULTILINE)

# A leading line that is meta commentary rather than content, e.g.
# "Here is the edited chapter:" or a bare duplicated heading "Chapter 3:".
_META_LINE_RE = re.compile(
    r"^(?:here(?:'s| is| below)?|sure|certainly|of course|as requested"
    r"|i(?:'ve| have)? .{0,40}(?:edited|written|rewritten|polished))"
    r"[^:]{0,100}:$"
    r"|^(?:#+\s*)?\**\s*chapter\s*\d+\s*\**\s*(?:[:.\-\u2013\u2014]\s*.{0,80})?\s*$",
    re.IGNORECASE,
)


def strip_code_fences(text):
    """Remove markdown code fence lines while keeping their content."""
    return _FENCE_RE.sub("", text).strip()


def _balanced_end(text, start, open_ch, close_ch):
    """Index of the bracket closing the one at `start` (strings and escapes
    respected), or None when it is never closed."""
    depth, in_str, esc = 0, False, False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
        elif ch == '"':
            in_str = True
        elif ch == open_ch:
            depth += 1
        elif ch == close_ch:
            depth -= 1
            if depth == 0:
                return i
    return None


def extract_json(text, expect="object"):
    """Extract and parse the first JSON object or array from LLM output.

    Args:
        text: raw model output (may contain code fences or commentary)
        expect: "object" or "array"

    Returns the parsed Python object.

    Commentary before the JSON may itself contain brackets ("use {name} for
    ..."): a balanced span that is not valid JSON is skipped and the search
    goes on AFTER it, never inside it, so a malformed object can't be
    mistaken for one of its own members.

    Raises ValueError if no balanced JSON of the expected kind is found.
    """
    cleaned = strip_code_fences(text)
    open_ch, close_ch = ("{", "}") if expect == "object" else ("[", "]")
    pos, failure = 0, None
    while True:
        start = cleaned.find(open_ch, pos)
        if start == -1:
            if failure:
                raise ValueError(f"could not parse JSON from model: {failure}")
            raise ValueError(f"no JSON {expect} found in model output")
        end = _balanced_end(cleaned, start, open_ch, close_ch)
        if end is None:
            raise ValueError("unbalanced JSON in model output")
        candidate = cleaned[start:end + 1]
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            # small models sometimes leave trailing commas
            fixed = re.sub(r",\s*([}\]])", r"\1", candidate)
            try:
                return json.loads(fixed)
            except json.JSONDecodeError as e:
                failure = e
        pos = end + 1


def clean_llm_text(text, max_dropped=2):
    """Strip code fences and leading meta-commentary lines.

    Drops up to ``max_dropped`` short preamble lines such as
    "Here is the edited version:" or a duplicated "Chapter 3:" heading.
    """
    cleaned = strip_code_fences(text)
    lines = cleaned.splitlines()
    dropped = 0
    while (
        lines
        and dropped < max_dropped
        and len(lines[0].strip()) <= 120
        and _META_LINE_RE.match(lines[0].strip())
    ):
        lines.pop(0)
        dropped += 1
        while lines and not lines[0].strip():
            lines.pop(0)
    return "\n".join(lines).strip()


def split_sections(seed_text):
    """Split a markdown seed into (title, body) pairs per heading.

    Headings of level 1-2 start a new section; deeper sub-headings stay in
    the current section's body (so '### Part One' stays inside Outline).
    """
    sections, title, body = [], None, []
    for line in seed_text.splitlines():
        m = re.match(r"^\s*#{1,2}\s+(.+)$", line)
        if m:
            if title is not None:
                sections.append((title, "\n".join(body).strip()))
            title, body = m.group(1).strip(), []
        elif title is not None:
            body.append(line)
    if title is not None:
        sections.append((title, "\n".join(body).strip()))
    return sections


def extract_seed_extras(seed_text):
    """Deterministically pull constraints and author notes from a seed.

    Verbatim fidelity matters for style constraints, so they are taken from
    the raw seed (never LLM-transcribed). Wrapped continuation lines are
    joined onto the bullet they belong to. Returns (constraints, notes):
      - constraints: bullet entries of the 'Constraints' section, if present
      - notes: bodies of any non-standard sections (e.g. 'Explicitness'),
        including the preamble under the document title, if any
    """
    constraints, notes = [], []
    for title, body in split_sections(seed_text):
        if not body:
            continue
        key = title.lower()
        if any(word in key for word in KNOWN_SECTIONS):
            if "constraint" in key:
                for line in body.splitlines():
                    stripped = line.strip()
                    if _HEADING_RE.match(stripped):
                        continue
                    if _BOLD_HEADING_RE.match(stripped):
                        # "**Plot Roadmap:**" on a line of its own is a
                        # heading written in bold: the constraints end here.
                        # Read on, a seed's whole outline became constraints.
                        break
                    if _BULLET_RE.match(stripped):
                        text = _BULLET_RE.sub("", stripped, count=1).strip()
                        if text:
                            constraints.append(text)
                    elif constraints and stripped:
                        # wrapped continuation of the previous constraint
                        constraints[-1] += " " + stripped
        else:
            notes.append(f"{title}:\n{body}")
    return constraints, "\n\n".join(notes).strip()


_CHAR_LINE_RE = re.compile(
    r"^(?:[-*\u2022]\s*|\d+[.)]\s*)?\*\*(.+?)\*\*\s*"
    r"(?:\((.+?)\))?\s*[-\u2013\u2014:]\s*(.+)$"
)
_HEADING_RE = re.compile(r"^#{1,6}\s")
# A bullet marker. A '*' followed by another '*' opens bold text, not a bullet
# ("**POV** - single" lost its first star and stopped matching anything).
_BULLET_RE = re.compile(r"^(?:[-\u2022]\s*|\*(?!\*)\s*|\d+[.)]\s*)")
# A line that is nothing but bold text: "**Plot Roadmap:**", "**Chapter 1**".
_BOLD_HEADING_RE = re.compile(r"^\*\*[^*]+\*\*:?$|^__[^_]+__:?$")


_ALIAS_INTRO = r"(?:a\.?k\.?a\.?|known as|called|nicknamed|goes by|also called)"
_ALIAS_DESC_RE = re.compile(
    rf"\b(?i:{_ALIAS_INTRO})\s+[\"'\u201c]?([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)")
_ALIAS_QUAL_RE = re.compile(rf"^{_ALIAS_INTRO}\s+(.+)$", re.IGNORECASE)


def find_aliases(qualifier, description):
    """Nicknames declared in a seed character line.

    Recognised: a bracketed qualifier that is a single capitalised word
    ('**Elizabeth (Liz)**') or starts with aka / known as / called / goes by,
    and the same phrases inside the description ('...known as Liz...').
    Returns (aliases, qualifier_was_alias).
    """
    aliases, used = [], False
    qualifier = (qualifier or "").strip()
    m = _ALIAS_QUAL_RE.match(qualifier)
    if m:
        aliases.append(m.group(1).strip(" \"'"))
        used = True
    elif re.fullmatch(r"[A-Z][a-z]+", qualifier):
        aliases.append(qualifier)
        used = True
    aliases += [a.strip() for a in _ALIAS_DESC_RE.findall(description or "")]
    seen, out = set(), []
    for a in aliases:
        if a and a.lower() not in seen:
            seen.add(a.lower())
            out.append(a)
    return out, used


def extract_seed_characters(seed_text):
    """Deterministically extract '**Name** - description' bullets from a
    Characters section. Wrapped continuation lines are joined onto the
    description. Returns a list of {name, role, description}."""
    characters = []
    for title, body in split_sections(seed_text):
        if "character" not in title.lower():
            continue
        current = None
        for line in body.splitlines():
            stripped = line.strip()
            if _HEADING_RE.match(stripped):        # e.g. '### The trio'
                current = None
                continue
            m = _CHAR_LINE_RE.match(stripped)
            if m:
                name = m.group(1).strip()
                desc = (m.group(3) or "").strip()
                # a bracketed qualifier is not part of the name:
                # '**Carole (from No.12)**' -> 'Carole' + '(from No.12)'
                qualifier = (m.group(2) or "").strip()
                paren = re.match(r"^(.+?)\s*\((.+?)\)\s*$", name)
                if paren:
                    name = paren.group(1).strip()
                    qualifier = qualifier or paren.group(2).strip()
                # strip a leading role word like 'protagonist.' / 'antagonist.'
                role = ""
                role_m = re.match(r"^(protagonist|antagonist|supporting)\b",
                                  desc, re.IGNORECASE)
                if role_m:
                    role = role_m.group(1).lower()
                aliases, qualifier_is_alias = find_aliases(qualifier, desc)
                if qualifier and not qualifier_is_alias:
                    desc = f"{desc} ({qualifier})"
                if name and name.lower() not in \
                        (c["name"].lower() for c in characters):
                    current = {"name": name, "role": role,
                               "description": desc, "aliases": aliases}
                    characters.append(current)
                else:
                    current = None
            elif current is not None and stripped:
                # wrapped continuation of the previous description
                current["description"] += " " + stripped
    for c in characters:  # aliases mentioned on wrapped description lines
        found, _ = find_aliases("", c["description"])
        c["aliases"] = list(dict.fromkeys(c.get("aliases", []) + found))
    return characters


def same_character(a, b):
    """True when two names plausibly denote one bible character: equal
    ignoring case and bracketed qualifiers, or sharing a first name
    ('Lisa' and 'Lisa Hale')."""
    def words(name):
        return re.sub(r"\(.*?\)", " ", str(name)).lower().split()
    wa, wb = words(a), words(b)
    return bool(wa and wb and (wa == wb or wa[0] == wb[0]))


def render_bible(bible):
    """Render the story bible as compact text for prompts."""
    if not bible:
        return "(no story bible available)"

    char_lines = "\n".join(
        "- {name}{role}{desc}".format(
            name=c.get("name", "?")
            + (f" (also called {', '.join(c['aliases'])})"
               if c.get("aliases") else ""),
            role=f" ({c['role']})" if c.get("role") else "",
            desc=f": {c['description']}" if c.get("description") else "",
        )
        for c in bible.get("characters") or []
    )

    parts = []
    if bible.get("genre"):
        parts.append(f"Genre: {bible['genre']}")
    if bible.get("tone"):
        parts.append(f"Tone/Style: {bible['tone']}")
    if bible.get("premise"):
        parts.append(f"Premise: {bible['premise']}")
    if char_lines:
        parts.append("Characters (use these names EXACTLY):\n" + char_lines)
    if bible.get("world"):
        parts.append(f"World:\n{bible['world']}")
    constraints = bible.get("constraints") or []
    if constraints:
        parts.append("Constraints (must remain true):\n"
                     + "\n".join(f"- {c}" for c in constraints))
    if bible.get("notes"):
        parts.append(f"Author notes:\n{bible['notes']}")
    clarifications = [c for c in bible.get("clarifications") or []
                      if isinstance(c, dict) and c.get("answer")]
    if clarifications:
        parts.append("Author's clarifications (part of the brief):\n"
                     + "\n".join(f"- {c.get('question', '')} {c['answer']}"
                                  for c in clarifications))
    return "\n\n".join(parts) if parts else "(no story bible available)"


# A per-chapter length the author states: "1,500-1,700 words per chapter",
# "2000 words a chapter", "chapters of about 1,800 words". Parsed here, not
# by the seed reviewer: the model's JSON has one number per field, so a RANGE
# came back as null and the configured words_per_chapter (2000) silently
# replaced the author's 1,500-1,700.
_NUM = r"(\d{1,2}(?:[,.]\d{3})+|\d{3,5}|\d{1,2}(?:\.\d)?\s*k)"
_SPAN = (_NUM + r"(?:\s*(?:-|–|—|to)\s*" + _NUM + r")?")
_CHAPTER_WORDS_RES = (
    # "1,500-1,700 words per chapter", "2000 words a chapter"
    re.compile(_SPAN + r"\s*words?\s*(?:per|a|each|every|for each|/)\s*chapter",
               re.I),
    # "chapters of about 1,800 words"
    re.compile(r"chapters?\s+(?:of|at|around|about|averaging|of about|of around)"
               r"\s+(?:about\s+|around\s+|roughly\s+|~\s*)?" + _SPAN + r"\s*words",
               re.I),
    # "Words per chapter: 1,500", "word count per chapter - 2000"
    re.compile(r"words?\s*(?:count\s*)?(?:per|a|each|every)\s*chapter\s*"
               r"(?:\*\*)?\s*[:=\-\u2013\u2014]?\s*(?:\*\*)?\s*"
               r"(?:about\s+|around\s+|roughly\s+|~\s*)?" + _SPAN, re.I),
    # "each chapter should be around 2,000 words", "chapter length: 1500 words"
    re.compile(r"\b(?:each|every|per|a)\s+chapter\b[^.\n]{0,40}?\b" + _SPAN
               + r"\s*words", re.I),
    re.compile(r"\bchapter\s+length\b[^.\n]{0,20}?" + _SPAN, re.I),
)


def _words_number(text):
    t = text.lower().replace(" ", "")
    if t.endswith("k"):
        return int(float(t[:-1]) * 1000)
    return int(re.sub(r"[,.]", "", t))


def stated_chapter_words(seed_text):
    """Every per-chapter length the seed states, as [(low, high), ...] in
    order (low == high for a single number). Implausible numbers (outside
    300-20,000) are skipped."""
    found = []
    for pattern in _CHAPTER_WORDS_RES:
        for m in pattern.finditer(seed_text or ""):
            low = _words_number(m.group(1))
            high = _words_number(m.group(2)) if m.group(2) else low
            low, high = min(low, high), max(low, high)
            if 300 <= low and high <= 20000:
                found.append((m.start(), (low, high)))
    out = []
    for _, span in sorted(found):
        if span not in out:
            out.append(span)
    return out


# A length statement inside brackets: "(about 2,000 words)" in an outline line.
_BRACKETED_RE = re.compile(r"\s*[(\[][^()\[\]\n]{0,60}?\b" + _NUM
                           + r"[^()\[\]\n]{0,40}?\bwords?\b[^()\[\]\n]{0,30}[)\]]",
                           re.I)
# The sentence pieces of one line (the split keeps the punctuation).
_PIECE_RE = re.compile(r"[^.!?;\n]*(?:[.!?;]+|$)")
# A leading label: "* **Word Count & Depth:**", "- Length:", "1. Style -"
_LABEL_RE = re.compile(
    r"^(\s*(?:[-*+]\s+|\d+[.)]\s+)?(?:\*\*[^*\n]{1,60}\*\*\s*:?|"
    r"__[^_\n]{1,60}__\s*:?|[A-Za-z][^:\n]{0,40}:(?!\d)))\s*")


# An outline entry ("- Chapter 3: Title - ..."): a bracketed word count on it
# is that chapter's length
_OUTLINE_ENTRY_RE = re.compile(r"^\s*(?:[-*+]\s+|#+\s*)?(?:\*\*)?chapter\s+\w+",
                               re.I)


def _states_chapter_length(text):
    return any(p.search(text) for p in _CHAPTER_WORDS_RES)


def strip_chapter_length(seed_text):
    """The seed with every per-chapter length statement taken out, and the
    pieces removed. Returns (text, removed).

    The book's size is settled once (book.words_per_chapter, the page's
    settings, the seed review) and handed to every stage as a number. Left in
    the seed, a length rode into every prompt through the bible, the author
    notes and the writer's brief - in one run as "1,500-1,700 words per
    chapter. 2,000 words per chapter." against a 2,000-word target. Read it
    first (stated_chapter_words), then strip it.

    The whole sentence that states the length goes (a sentence is cut at
    . ! ? ;), and a bracketed "(about 2,000 words)" goes on its own. A line's
    label ("**Word Count & Depth:**") stays when other sentences follow it; a
    line with nothing left goes. Total book lengths ("about 50,000 words")
    are not per-chapter and stay."""
    removed = []
    out = []
    for line in (seed_text or "").split("\n"):
        outline = bool(_OUTLINE_ENTRY_RE.match(line))
        if not (_states_chapter_length(line)
                or outline and _BRACKETED_RE.search(line)):
            out.append(line)
            continue

        def cut_bracket(m):
            if outline or _states_chapter_length(m.group(0)):
                removed.append(m.group(0).strip())
                return ""
            return m.group(0)

        line = _BRACKETED_RE.sub(cut_bracket, line)
        if not _states_chapter_length(line):
            out.append(line.rstrip())
            continue
        label_m = _LABEL_RE.match(line)
        label = label_m.group(1).rstrip() if label_m else ""
        body = line[label_m.end():] if label_m else line
        kept = []
        for piece in _PIECE_RE.findall(body):
            if not piece.strip():
                continue
            if _states_chapter_length(piece) or (
                    label and _states_chapter_length(label + " " + piece)
                    and not kept):
                removed.append(piece.strip())
            else:
                kept.append(piece.strip())
        if kept:
            out.append((label + " " if label else "") + " ".join(kept))
        elif label and not _states_chapter_length(label) and not re.search(
                r"\b(?:length|word)", label, re.I):
            out.append(label.rstrip())       # a label for something else
        # else: the line was only the length; drop it
    text = "\n".join(out)
    if (seed_text or "").endswith("\n") and not text.endswith("\n"):
        text += "\n"
    return text, removed
