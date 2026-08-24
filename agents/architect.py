"""
Architect Agent
Converts a raw seed prompt (premise, characters, world, outline, ...) into a
structured story bible that every later agent uses for consistency.
"""

from shared.context import context, update_context
from shared.llm_utils import (extract_json, extract_seed_characters,
                              extract_seed_extras)
from shared.ollama_client import generate
from shared.output import format_bible_markdown, save_interim


def _fallback_title(seed_text):
    """Best-effort title: first markdown heading, else first non-empty line."""
    for line in seed_text.splitlines():
        stripped = line.strip().lstrip("#").strip()
        if stripped:
            return stripped[:80]
    return "Untitled Story"


def _minimal_bible(seed_text):
    """Bible used when the LLM is unavailable or returns unusable JSON.
    The raw seed is kept under 'seed' so downstream agents still see it."""
    return {
        "title": _fallback_title(seed_text),
        "genre": "",
        "tone": "",
        "premise": seed_text.strip()[:600],
        "characters": [],
        "world": "",
        "constraints": [],
        "notes": "",
    }


def _normalize(bible, seed_text):
    """Coerce the model's JSON into the shape we rely on downstream."""
    if not isinstance(bible, dict):
        raise ValueError("bible is not a JSON object")

    characters = []
    for c in bible.get("characters") or []:
        if isinstance(c, dict) and c.get("name"):
            characters.append({
                "name": str(c["name"]).strip(),
                "role": str(c.get("role", "")).strip(),
                "description": str(c.get("description", "")).strip(),
            })
        elif isinstance(c, str) and c.strip():
            characters.append({"name": c.strip(), "role": "", "description": ""})

    title = str(bible.get("title") or "").strip() or _fallback_title(seed_text)
    constraints = [str(c) for c in (bible.get("constraints") or []) if str(c).strip()]

    return {
        "title": title,
        "genre": str(bible.get("genre", "")).strip(),
        "tone": str(bible.get("tone", "")).strip(),
        "premise": str(bible.get("premise", "")).strip(),
        "characters": characters,
        "world": str(bible.get("world", "")).strip(),
        "constraints": constraints,
        "notes": str(bible.get("notes", "")).strip(),
    }


def run_architect(seed_text):
    """
    Build a story bible from the seed prompt.

    Tries a structured LLM extraction first; falls back to a minimal bible
    built straight from the seed text so the pipeline never dead-ends.
    """
    print("[ARCHITECT] Building story bible from seed prompt...")

    prompt = f"""You are a story architect. Convert the creative seed below into a JSON story bible.

Return ONLY valid JSON (no markdown fences, no commentary) with exactly these keys:
{{
  "title": "short book title",
  "genre": "e.g. fantasy mystery, techno-thriller",
  "tone": "narrative voice and style guidance",
  "premise": "2-3 sentence summary of the story",
  "characters": [{{"name": "...", "role": "protagonist/antagonist/supporting", "description": "appearance, personality, motivation"}}],
  "world": "setting, rules of the world, important background",
  "constraints": ["facts that must stay consistent: names, dates, magic/tech rules, tone"],
  "notes": "any other author instructions from the seed: explicitness level, banned words, style rules"
}}

Extract faithfully from the seed - do not invent major new characters or change names.
If the seed lacks a section, fill it in minimally and sensibly.

Creative seed:
\"\"\"{seed_text}\"\"\"
"""
    bible = None
    try:
        raw = generate(prompt, agent="architect")
        bible = _normalize(extract_json(raw, expect="object"), seed_text)
        print(f"[ARCHITECT] Story bible ready: '{bible['title']}' "
              f"({len(bible['characters'])} characters)")
    except Exception as e:
        print(f"[ARCHITECT] LLM bible extraction failed ({e}). "
              "Falling back to raw seed as bible.")
        bible = _minimal_bible(seed_text)

    # Style constraints and author notes must be VERBATIM, so take them
    # straight from the seed rather than trusting the LLM's transcription.
    raw_constraints, raw_notes = extract_seed_extras(seed_text)
    if raw_constraints:
        bible["constraints"] = raw_constraints
    if raw_notes:
        bible["notes"] = (bible.get("notes", "") + "\n\n" + raw_notes).strip() \
            if bible.get("notes") else raw_notes

    # Same for characters: '**Name** - description' bullets from the seed
    # are authoritative; LLM-extracted characters are only extras.
    seed_chars = extract_seed_characters(seed_text)
    if seed_chars:
        seed_names = {c["name"].lower() for c in seed_chars}
        extras = [c for c in bible.get("characters", [])
                  if c["name"].lower() not in seed_names]
        bible["characters"] = seed_chars + extras
        print(f"[ARCHITECT] {len(seed_chars)} characters taken verbatim "
              f"from the seed" + (f", {len(extras)} from the LLM" if extras else ""))

    bible["seed"] = seed_text
    update_context("bible", bible)
    update_context("title", bible["title"])
    update_context("seed", seed_text)
    save_interim("story_bible.md", format_bible_markdown(bible))
    return bible
