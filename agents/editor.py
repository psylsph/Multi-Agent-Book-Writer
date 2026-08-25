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
from shared.consistency import (check_chronology, format_findings,
                                lint_book, lint_chapter)
from shared.story_state import merge_states
from agents.reviewer import run_reviewer


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


def _revise(number, title, draft, lint, issues):
    """Ask the model to fix specific findings in a draft."""
    prompt = f"""You are revising Chapter {number} ("{title}") to fix specific, verified findings.

FINDINGS TO FIX
{format_findings(lint, issues)}

Rules:
- Fix ONLY the listed findings. Do not rewrite style, add scenes, or change plot beyond the fixes.
- For banned words, replace them naturally; for quotas, cut the least-needed instances down to the limit.
- For continuity findings, apply the suggested fix or an equivalent minimal change.
- Keep the chapter's voice, length, and all unaffected content intact.

CHAPTER (current draft):
\"\"\"{draft}\"\"\"

Return ONLY the full revised chapter prose - no headings, no commentary."""
    revised = clean_llm_text(generate(prompt, agent="editor"))
    if len(revised.split()) < 100:
        print(f"[EDITOR] Revision for chapter {number} too short "
              f"({len(revised.split())} words); discarding.")
        return None
    return revised


def run_editor():
    """
    Review-and-revise loop per chapter, then a final polish pass:
      1. deterministic lint (banned words, quotas, name & chronology checks)
      2. reviewer agent (dead characters, relationship/knowledge/timeline)
      3. bounded revision rounds until findings are fixed
      4. style polish, then a final lint guard
    On any failure the best version so far is kept (nothing is ever lost).
    """
    print("[EDITOR] Starting review & edit phase...")
    cfg = get_config()
    bible = context.get("bible") or {}
    chapters = context.get("chapters", [])
    drafts = context.get("drafts", {})
    chronology = context.get("chronology") or {}
    constraints = bible.get("constraints") or []
    max_rounds = int(cfg["book"].get("revision_rounds", 2))
    reviewer_on = cfg.get("agents", {}).get("reviewer", {}).get("enabled", True)
    reference = _bible_reference(bible)

    final = []
    for chapter in chapters:
        n, title = chapter["number"], chapter["title"]
        draft = drafts.get(n)
        if not draft:
            print(f"[EDITOR] No draft for chapter {n}; skipping.")
            continue

        print(f"[EDITOR] Chapter {n}/{len(chapters)}: review & edit")

        def full_lint(text):
            findings = lint_chapter(n, text, bible, constraints)
            findings += check_chronology(
                n, text, merge_states(chronology, upto=n))
            return findings

        # ---- review & revise rounds
        lint = full_lint(draft)
        verdict, issues = ("pass", [])
        if reviewer_on:
            print(f"[REVIEWER] Reviewing chapter {n}...")
            verdict, issues = run_reviewer(n, title, draft)
            print(f"[REVIEWER] Chapter {n}: {verdict}"
                  + (f" ({len(issues)} issues)" if issues else ""))
        else:
            print(f"[REVIEWER] disabled; deterministic lint only "
                  f"({len(lint)} findings)")

        for rnd in range(1, max_rounds + 1):
            if not lint and verdict == "pass":
                break
            print(f"[EDITOR] Revision round {rnd}/{max_rounds} for chapter "
                  f"{n}: {len(lint)} lint, {len(issues)} review issues")
            try:
                revised = _revise(n, title, draft, lint, issues)
            except Exception as e:
                print(f"[EDITOR] Revision failed for chapter {n}: {e}")
                break
            if revised is None:
                break
            draft = revised
            lint = full_lint(draft)
            if issues or verdict != "pass":
                verdict, issues = run_reviewer(n, title, draft)

        if lint:
            print(f"[EDITOR] Chapter {n}: {len(lint)} findings remain after "
                  f"revision (see lint_report.md)")

        # ---- final polish pass
        draft_with_heading = f"## Chapter {n}: {title}\n\n{draft}"
        print(f"[EDITOR] Polishing chapter {n}...")
        prompt = f"""You are a professional fiction editor. Edit the chapter below.

STYLE RULES
- Fix grammar, spelling, punctuation, and awkward phrasing.
- Improve clarity and flow while preserving the author's voice and ALL story
  content. Do NOT summarize, shorten, or rewrite the plot.
- Do NOT introduce any continuity changes: keep every name, fact, and event
  exactly as written.

STORY BIBLE REFERENCE
{reference}

CHAPTER (begins with its markdown heading - keep that heading unchanged):
{draft_with_heading}

Return ONLY the edited chapter, starting with its original heading."""
        try:
            edited = clean_llm_text(generate(prompt, agent="editor"))
            if not edited.startswith("##"):
                edited = f"## Chapter {n}: {title}\n\n{edited}"
        except Exception as e:
            print(f"[EDITOR] Error editing chapter {n}: {e}; keeping draft.")
            edited = draft_with_heading

        # deterministic guard: keep whichever version has fewer lint findings
        body = edited.split("\n\n", 1)[1] if "\n\n" in edited else edited
        if len(full_lint(body)) > len(full_lint(draft)):
            print(f"[EDITOR] Polish introduced lint findings for chapter {n}; "
                  "keeping pre-polish version.")
            edited = draft_with_heading

        final.append(edited)
        save_interim(chapter_filename("edited", n), edited)
        print(f"[EDITOR] Chapter {n} complete.")

    update_context("final", final)
    _write_lint_report(final)
    return save_book(final)


def _write_lint_report(final_chapters):
    """Final deterministic lint report across the whole book."""
    bible = context.get("bible") or {}
    constraints = bible.get("constraints") or []
    full_text = "\n\n".join(final_chapters)
    findings = lint_book(full_text, bible, constraints)

    lines = ["# Lint Report (final book)", "",
             f"Chapters: {len(final_chapters)}  "
             f"Words: {len(full_text.split())}", ""]
    if not findings:
        lines.append("No deterministic findings. "
                     "(LLM continuity reviews are in review_chapter_NN.md)")
    else:
        lines += [f"- [{f['check']}] {f['detail']}" for f in findings]
    save_interim("lint_report.md", "\n".join(lines) + "\n")


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
