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


def extract_json(text, expect="object"):
    """Extract and parse the first JSON object or array from LLM output.

    Args:
        text: raw model output (may contain code fences or commentary)
        expect: "object" or "array"

    Returns the parsed Python object.

    Raises ValueError if no balanced JSON of the expected kind is found.
    """
    cleaned = strip_code_fences(text)
    open_ch, close_ch = ("{", "}") if expect == "object" else ("[", "]")
    start = cleaned.find(open_ch)
    if start == -1:
        raise ValueError(f"no JSON {expect} found in model output")

    depth, in_str, esc = 0, False, False
    for i in range(start, len(cleaned)):
        ch = cleaned[i]
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
                candidate = cleaned[start:i + 1]
                try:
                    return json.loads(candidate)
                except json.JSONDecodeError:
                    # small models sometimes leave trailing commas
                    fixed = re.sub(r",\s*([}\]])", r"\1", candidate)
                    try:
                        return json.loads(fixed)
                    except json.JSONDecodeError as e:
                        raise ValueError(f"could not parse JSON from model: {e}")
    raise ValueError("unbalanced JSON in model output")


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
                    if re.match(r"^(?:[-*\u2022]\s*|\d+[.)]\s*)", stripped):
                        text = re.sub(
                            r"^(?:[-*\u2022]\s*|\d+[.)]\s*)", "", stripped
                        ).strip()
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
                # strip a leading role word like 'protagonist.' / 'antagonist.'
                role = ""
                role_m = re.match(r"^(protagonist|antagonist|supporting)\b",
                                  desc, re.IGNORECASE)
                if role_m:
                    role = role_m.group(1).lower()
                if name and name.lower() not in \
                        (c["name"].lower() for c in characters):
                    current = {"name": name, "role": role,
                               "description": desc}
                    characters.append(current)
                else:
                    current = None
            elif current is not None and stripped:
                # wrapped continuation of the previous description
                current["description"] += " " + stripped
    return characters


def render_bible(bible):
    """Render the story bible as compact text for prompts."""
    if not bible:
        return "(no story bible available)"

    char_lines = "\n".join(
        "- {name}{role}{desc}".format(
            name=c.get("name", "?"),
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
    return "\n\n".join(parts) if parts else "(no story bible available)"
