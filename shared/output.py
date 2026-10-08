"""Output helpers: interim pipeline artifacts + story bible formatting.

Interim artifacts (story bible, outline, lore briefs, chapter drafts,
rolling summaries, edited chapters) are written to <output dir>/interim/ as
soon as they are produced, so you can watch a run take shape instead of
waiting for the final book. All interim writes are best-effort: a failure
never breaks the pipeline.

Durable per-chapter files: every chapter is also written to
<output dir>/chapters/chapter_NN.md the moment it is created (draft first,
replaced by the edited version when it finishes polishing). Unlike the
interim directory these files are ALWAYS written (even with output.interim
off) and are never wiped by clear_interim(); a fresh start archives them (see
archive_previous_run()) instead of deleting them.
"""

import json
import os
import shutil
import time
from pathlib import Path

from shared.llm_client import get_config


def atomic_write_text(path, text):
    """Write text so a crash mid-write can never leave a truncated file:
    write a sibling temp file, then atomically swap it into place."""
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def interim_enabled():
    """Whether interim output is on (config output.interim, default True)."""
    return bool(get_config()["output"]["interim"])


def interim_dir():
    """Return (and create) the interim output directory."""
    path = Path(get_config()["output"]["directory"]) / "interim"
    path.mkdir(parents=True, exist_ok=True)
    return path


def clear_interim():
    """Remove leftover interim files from previous runs."""
    if not interim_enabled():
        return
    path = Path(get_config()["output"]["directory"]) / "interim"
    if path.exists():
        shutil.rmtree(path, ignore_errors=True)


def archive_previous_run():
    """Move the previous run out of the way WITHOUT deleting it.

    A fresh start used to wipe interim/ (and leave stale chapters/ behind).
    Now interim/, chapters/ and the assembled book are moved into
    <output dir>/archive/<timestamp>/, so hours of generated text are never
    lost to a restart. Returns the archive path, or None when there was
    nothing to archive. Never raises on a missing file; a failed move is
    reported and the run continues.
    """
    cfg = get_config()["output"]
    out = Path(cfg["directory"])
    candidates = [out / "interim", out / "chapters", out / "story_bible.md"]
    if cfg.get("overwrite", True):          # otherwise the -N naming applies
        candidates.append(out / cfg.get("filename", "draft.md"))
    present = [p for p in candidates if p.exists()
               and (not p.is_dir() or any(p.iterdir()))]
    if not present:
        return None
    dest = out / "archive" / time.strftime("%Y%m%d-%H%M%S")
    suffix = 0
    while dest.exists():
        suffix += 1
        dest = out / "archive" / f"{time.strftime('%Y%m%d-%H%M%S')}-{suffix}"
    dest.mkdir(parents=True)
    for path in present:
        try:
            shutil.move(str(path), str(dest / path.name))
        except OSError as e:
            print(f"[PIPELINE] Could not archive {path}: {e}")
    return dest


def chapters_dir():
    """Return (and create) the durable per-chapter output directory."""
    path = Path(get_config()["output"]["directory"]) / "chapters"
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_chapter(number, title, text):
    """Write a chapter to <output dir>/chapters/chapter_NN.md.

    Called as soon as a chapter exists (draft) and again when its edited
    version is ready. Always writes - independent of the interim flag - and
    never raises, so a save failure cannot break the pipeline.
    """
    if not text:
        return None
    try:
        path = chapters_dir() / f"chapter_{number:02d}.md"
        atomic_write_text(path, f"# Chapter {number}: {title}\n\n{text}\n")
        print(f"[CHAPTER] Saved {path}")
        return path
    except OSError as e:
        print(f"[CHAPTER] Could not save chapter {number}: {e}")
        return None


def save_interim(filename, text):
    """Write an interim artifact; returns its path or None. Never raises."""
    if not interim_enabled() or not text:
        return None
    try:
        path = interim_dir() / filename
        atomic_write_text(path, text)
        print(f"[INTERIM] Saved {path}")
        return path
    except OSError as e:
        print(f"[INTERIM] Could not save {filename}: {e}")
        return None


def save_interim_json(filename, obj):
    """Write a Python object as pretty JSON to the interim directory."""
    try:
        text = json.dumps(obj, indent=2, ensure_ascii=False)
    except (TypeError, ValueError) as e:
        print(f"[INTERIM] Could not serialise {filename}: {e}")
        return None
    return save_interim(filename, text)


def chapter_filename(prefix, number):
    """Stable per-chapter filename, e.g. draft_chapter_03.md."""
    return f"{prefix}_chapter_{number:02d}.md"


def format_bible_markdown(bible):
    """Human-readable story bible (used for interim + final saves)."""
    title = bible.get("title", "Untitled")
    lines = [
        f"# {title} - Story Bible",
        "",
        "## Premise",
        "",
        bible.get("premise", "") or "(none)",
        "",
        "## Genre / Tone",
        "",
        f"{bible.get('genre', '') or '(unset)'} / "
        f"{bible.get('tone', '') or '(unset)'}",
        "",
        "## Characters",
        "",
    ]
    characters = bible.get("characters") or []
    if characters:
        for c in characters:
            role = f" ({c['role']})" if c.get("role") else ""
            also = (f" (also called {', '.join(c['aliases'])})"
                    if c.get("aliases") else "")
            lines.append(f"- **{c['name']}**{also}{role}: "
                         f"{c.get('description', '')}")
    else:
        lines.append("(none)")
    lines += ["", "## World", "", bible.get("world", "") or "(none)", ""]

    constraints = bible.get("constraints") or []
    if constraints:
        lines += ["## Constraints", ""]
        lines += [f"- {c}" for c in constraints]
        lines.append("")

    if bible.get("notes"):
        lines += ["## Author Notes", "", bible["notes"], ""]

    clarifications = bible.get("clarifications") or []
    if clarifications:
        lines += ["## Clarifications", ""]
        lines += [f"- **{c.get('question', '')}** {c.get('answer', '')}"
                  + ("" if c.get("answered") else " *(assumed)*")
                  for c in clarifications]
        lines.append("")
    return "\n".join(lines)
