"""
Planner Agent
Produces the chapter outline.

If the seed prompt contains an explicit outline, it is honored directly
(only expanding it via the LLM when the user asks for more chapters).
Otherwise the planner generates a JSON outline from the story bible.
"""

import json
import re

from shared.context import context, update_context
from shared.llm_utils import extract_json, strip_code_fences
from shared.llm_client import generate, get_config
from shared.output import save_interim

# "## Outline", "# Story Outline:", "Outline:" section headers
_SECTION_HEADER_RE = re.compile(
    r"^\s*#{0,4}\s*\**\s*(?:the\s+)?(?:story\s+)?outline\s*\**\s*:?\s*$",
    re.IGNORECASE,
)
# "Chapter 3: Title - summary" (number optional after we strip bullets)
_CHAPTER_LINE_RE = re.compile(
    r"^(?:[-*\u2022]\s*|\d+[.)]\s*)?\**\s*[Cc]hapter\s+(\d+)\s*\**\s*[:.\-\u2013\u2014]?\s*(.+)$"
)


def _parse_chapter_line(line):
    """Parse an explicit 'Chapter N' line into (number, title, summary) or None."""
    s = line.strip()
    if not s:
        return None
    m = _CHAPTER_LINE_RE.match(s)
    if not m:
        return None
    num, rest = int(m.group(1)), m.group(2).strip()
    rest = rest.strip("*").strip()
    rest = re.sub(r"^[:\-\u2013\u2014\s]+", "", rest).strip()
    title, summary = _split_title_summary(rest)
    if not title:
        return None
    return (num, title, summary)


def _split_title_summary(rest):
    """Split 'Title - summary' on the first separator."""
    rest = rest.strip("*").strip()
    title, summary = rest, ""
    for sep in (" - ", " \u2014 ", " \u2013 ", ": "):
        if sep in rest:
            title, summary = rest.split(sep, 1)
            break
    return title.strip().strip("*").strip(), summary.strip()


def _header_level(line):
    """Markdown heading level of a line, or None if not a heading."""
    m = re.match(r"^\s*(#{1,6})\s", line)
    return len(m.group(1)) if m else None


def _find_outline_span(lines):
    """Return the lines of the Outline section (None if there isn't one).

    The section ends at the first heading of the same or higher level; deeper
    sub-headings (e.g. '### Part One') are kept inside the span.
    """
    for idx, line in enumerate(lines):
        if _SECTION_HEADER_RE.match(line):
            level = _header_level(line) or 0
            for end in range(idx + 1, len(lines)):
                lvl = _header_level(lines[end])
                if lvl is not None and lvl <= level:
                    return lines[idx + 1:end]
            return lines[idx + 1:]
    return None


def _parse_outline_span(span):
    """Parse the lines of an outline section into chapter dicts.

    - explicit 'Chapter N' lines are chapters
    - bare '- Title - summary' bullets are chapters ONLY when the section
      contains no explicit 'Chapter N' lines (otherwise they are treated as
      fold-in beats/notes and appended to the previous chapter)
    - deeper sub-headings (e.g. '### Part One') become 'part' labels
    - any other non-blank line is a wrapped continuation of the previous
      chapter's summary
    """
    has_numbered = any(_parse_chapter_line(l) for l in span)
    chapters, current_part = [], ""
    for line in span:
        stripped = line.strip()
        if _header_level(line) is not None:
            current_part = re.sub(r"^#+\s*", "", stripped).strip()
            continue
        if not stripped:
            continue
        parsed = _parse_chapter_line(line)
        if parsed:
            chapters.append({"number": parsed[0], "title": parsed[1],
                             "summary": parsed[2], "part": current_part})
            continue
        m = re.match(r"^(?:[-*\u2022]\s*|\d+[.)]\s*)(.+)$", stripped)
        if m and not has_numbered:
            title, summary = _split_title_summary(m.group(1).strip())
            chapters.append({"number": None, "title": title,
                             "summary": summary, "part": current_part})
            continue
        if chapters:  # wrapped continuation or fold-in beat
            chapters[-1]["summary"] = (chapters[-1]["summary"]
                                       + " " + stripped).strip()
    return chapters


def _sanitize_numbers(chapters):
    """Ensure chapter numbers are unique and strictly increasing."""
    out, prev = [], 0
    for ch in chapters:
        n = ch.get("number")
        n = n if isinstance(n, int) and n > prev else prev + 1
        out.append({"number": n, "title": ch["title"],
                    "summary": ch.get("summary", ""),
                    "part": ch.get("part", "")})
        prev = n
    return out


def extract_seed_outline(seed_text):
    """Pull an explicit chapter outline out of the seed prompt, if any.

    First looks for a dedicated Outline section (supporting Parts,
    wrapped summaries and fold-in beats); otherwise scans the whole seed
    for explicit 'Chapter N:' lines (requiring at least two).

    Returns a list of {"number", "title", "summary", "part"} (possibly empty).
    """
    if not seed_text:
        return []

    lines = seed_text.splitlines()

    # Pass 1: dedicated outline section
    span = _find_outline_span(lines)
    if span is not None:
        chapters = _parse_outline_span(span)
        if chapters:
            return _sanitize_numbers(chapters)

    # Pass 2: loose scan for explicit "Chapter N" lines anywhere
    loose = [p for p in (_parse_chapter_line(l) for l in lines) if p]
    if len(loose) >= 2:
        return _sanitize_numbers(
            [{"number": n, "title": t, "summary": s, "part": ""}
             for n, t, s in loose])
    return []


def _fallback_line_parse(raw):
    """Last-resort outline parsing from free-form LLM text."""
    chapters = []
    for line in strip_code_fences(raw).splitlines():
        parsed = _parse_chapter_line(line)
        if parsed:
            chapters.append({"number": parsed[0], "title": parsed[1],
                             "summary": parsed[2], "part": ""})
    return _sanitize_numbers(chapters)


def _plan_with_llm(bible, target, fixed_prefix=None):
    """Ask the LLM for a JSON outline, honoring any fixed seed chapters."""
    characters = "\n".join(
        f"- {c.get('name', '?')}"
        + (f" ({c.get('role')})" if c.get("role") else "")
        + (f": {c.get('description')}" if c.get("description") else "")
        for c in bible.get("characters") or []
    )

    fixed_block = ""
    if fixed_prefix:
        fixed = [{"number": c["number"], "title": c["title"],
                  "summary": c["summary"]} for c in fixed_prefix]
        fixed_block = (
            "These chapters are FIXED (they come from the author's outline) "
            "and must appear first, unchanged:\n"
            + json.dumps(fixed, indent=2)
            + f"\nGenerate {target - len(fixed)} more chapters that continue the "
              "story naturally, up to the story's conclusion.\n\n"
        )

    prompt = f"""You are planning the chapter outline of a novel.

TITLE: {bible.get('title', '')}
GENRE: {bible.get('genre', '')}
PREMISE: {bible.get('premise', '')}
CHARACTERS:
{characters or '(see premise)'}

WORLD:
{bible.get('world', '')}

{fixed_block}Return ONLY a JSON array of exactly {target} objects, one per chapter in story order:
[{{"number": 1, "title": "short evocative title", "summary": "1-2 sentences: what happens and how the arc advances"}}]

The chapters must form a complete dramatic arc (setup, rising action, climax, resolution).
No markdown fences, no commentary - JSON only."""

    raw = generate(prompt, agent="planner")
    try:
        chapters = extract_json(raw, expect="array")
    except ValueError:
        print("[PLANNER] JSON parse failed; falling back to line parsing.")
        return _fallback_line_parse(raw)

    if not isinstance(chapters, list) or not chapters:
        raise ValueError("planner returned an empty outline")

    normalized = []
    for ch in chapters:
        if not isinstance(ch, dict) or not str(ch.get("title", "")).strip():
            continue
        normalized.append({
            "title": str(ch["title"]).strip(),
            "summary": str(ch.get("summary", "")).strip(),
        })
    if not normalized:
        raise ValueError("planner outline had no usable chapters")

    # enforce the requested count
    if len(normalized) > target:
        normalized = normalized[:target]
    elif len(normalized) < target:
        print(f"[PLANNER] Warning: asked for {target} chapters, "
              f"got {len(normalized)}; continuing with {len(normalized)}.")

    return _assign_numbers([(None, c["title"], c["summary"]) for c in normalized])


def _outline_markdown(chapters):
    """Human-readable outline for interim output."""
    lines = [f"# Outline ({len(chapters)} chapters)", ""]
    for ch in chapters:
        part = f"  *({ch['part']})*" if ch.get("part") else ""
        lines.append(f"{ch['number']}. **{ch['title']}**{part}")
        if ch.get("summary"):
            lines.append(f"   {ch['summary']}")
        lines.append("")
    return "\n".join(lines)


def run_planner(num_chapters=None):
    """
    Produce the chapter outline for the book.

    Priority for chapter count:
      1. explicit num_chapters argument (from --chapters / CLI)
      2. the number of chapters in the seed's own outline
      3. config book.num_chapters

    Returns the chapter list (possibly empty on failure).
    """
    print("[PLANNER] Planning chapter outline...")
    cfg = get_config()
    bible = context.get("bible") or {}
    seed_outline = extract_seed_outline(context.get("seed", ""))

    target = num_chapters
    if target is None:
        target = len(seed_outline) if seed_outline else cfg["book"]["num_chapters"]

    chapters = []
    if seed_outline:
        if len(seed_outline) >= target:
            if len(seed_outline) > target:
                print(f"[PLANNER] Seed outline has {len(seed_outline)} chapters; "
                      f"using the first {target}. "
                      f"(Run with --chapters {len(seed_outline)} to keep them all.)")
            chapters = seed_outline[:target]
            print(f"[PLANNER] Using the seed prompt's outline "
                  f"({len(chapters)} chapters).")
        else:
            print(f"[PLANNER] Seed outline has {len(seed_outline)} chapters; "
                  f"expanding to {target} with the LLM.")
            try:
                chapters = _plan_with_llm(bible, target, fixed_prefix=seed_outline)
            except Exception as e:
                print(f"[PLANNER] Expansion failed ({e}); "
                      "keeping the seed outline as-is.")
                chapters = seed_outline
    else:
        print(f"[PLANNER] No outline found in seed; "
              f"generating {target} chapters from the story bible.")
        try:
            chapters = _plan_with_llm(bible, target)
        except Exception as e:
            print(f"[PLANNER] Error: {e}")
            return []

    update_context("chapters", chapters)
    save_interim("outline.md", _outline_markdown(chapters))
    print(f"[PLANNER] Outline ready ({len(chapters)} chapters):")
    for ch in chapters:
        print(f"  {ch['number']}. {ch['title']}"
              + (f" - {ch['summary'][:70]}" if ch.get("summary") else ""))
    return chapters
