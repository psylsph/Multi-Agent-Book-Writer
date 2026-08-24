"""
Editor Agent
Polishes each chapter and checks it for consistency against the story bible
(character names, world facts, constraints). Saves the assembled book and a
human-readable copy of the story bible to the output directory.
"""

from pathlib import Path

from shared.context import context, update_context
from shared.llm_utils import clean_llm_text, render_bible
from shared.ollama_client import get_config, generate
from shared.output import chapter_filename, format_bible_markdown, save_interim


def _bible_reference(bible):
    """Compact bible summary for the editor's consistency check."""
    names = ", ".join(c["name"] for c in bible.get("characters") or [])
    parts = []
    if names:
        parts.append(f"Character names (must be spelled exactly): {names}")
    if bible.get("world"):
        parts.append(f"World: {bible['world'][:800]}")
    constraints = bible.get("constraints") or []
    if constraints:
        parts.append("Constraints:\n" + "\n".join(f"- {c}" for c in constraints))
    if bible.get("notes"):
        parts.append(f"Author notes: {bible['notes']}")
    return "\n".join(parts) or "(no bible)"


def run_editor():
    """
    Edit each draft for grammar, flow, and bible consistency, then save.
    On per-chapter failure the unedited draft is kept (never lost).
    """
    print("[EDITOR] Starting editing phase...")
    cfg = get_config()
    bible = context.get("bible") or {}
    chapters = context.get("chapters", [])
    drafts = context.get("drafts", {})
    reference = _bible_reference(bible)

    final = []
    for chapter in chapters:
        n, title = chapter["number"], chapter["title"]
        draft = drafts.get(n)
        if not draft:
            print(f"[EDITOR] No draft for chapter {n}; skipping.")
            continue

        draft_with_heading = f"## Chapter {n}: {title}\n\n{draft}"
        print(f"[EDITOR] Editing chapter {n}/{len(chapters)}...")

        prompt = f"""You are a professional fiction editor. Edit the chapter below.

STYLE RULES
- Fix grammar, spelling, punctuation, and awkward phrasing.
- Improve clarity and flow while preserving the author's voice and ALL story
  content. Do NOT summarize, shorten, or rewrite the plot.
- Fix any inconsistencies with the story bible (character names, world facts,
  constraints) by aligning the text to the bible.

STORY BIBLE REFERENCE
{reference}

CHAPTER (begins with its markdown heading - keep that heading unchanged):
{draft_with_heading}

Return ONLY the edited chapter, starting with its original heading."""

        try:
            edited = clean_llm_text(generate(prompt, agent="editor"))
            if not edited.startswith("##"):
                # model dropped the heading; restore it
                edited = f"## Chapter {n}: {title}\n\n{edited}"
            final.append(edited)
            save_interim(chapter_filename("edited", n), edited)
            print(f"[EDITOR] Chapter {n} editing completed.")
        except Exception as e:
            print(f"[EDITOR] Error editing chapter {n}: {e}; keeping draft.")
            final.append(draft_with_heading)

    update_context("final", final)
    return save_book(final)


def save_book(final_chapters):
    """Assemble and write the book + story bible to the output directory."""
    cfg = get_config()
    out_dir = Path(cfg["output"]["directory"])
    out_dir.mkdir(parents=True, exist_ok=True)

    filename = cfg["output"].get("filename", "draft.md")
    out_path = out_dir / filename
    if not cfg["output"].get("overwrite", True) and out_path.exists():
        stem, suffix = out_path.stem, out_path.suffix or ".txt"
        i = 1
        while (out_dir / f"{stem}-{i}{suffix}").exists():
            i += 1
        out_path = out_dir / f"{stem}-{i}{suffix}"

    title = context.get("title", "Untitled")
    book = f"# {title}\n\n" + "\n\n".join(final_chapters) + "\n"
    try:
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(book)
        print(f"[EDITOR] Final book saved to {out_path}")
    except OSError as e:
        print(f"[EDITOR] Error saving book: {e}")
        return book

    # human-readable copy of the story bible for easy inspection
    bible = context.get("bible") or {}
    try:
        bible_path = out_dir / "story_bible.md"
        bible_path.write_text(format_bible_markdown(bible), encoding="utf-8")
        print(f"[EDITOR] Story bible saved to {bible_path}")
    except OSError as e:
        print(f"[EDITOR] Warning: could not save story bible: {e}")

    return book
