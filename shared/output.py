"""Output helpers: interim pipeline artifacts + story bible formatting.

Interim artifacts (story bible, outline, lore briefs, chapter drafts,
rolling summaries, edited chapters) are written to <output dir>/interim/ as
soon as they are produced, so you can watch a run take shape instead of
waiting for the final book. All interim writes are best-effort: a failure
never breaks the pipeline.
"""

import json
import shutil
from pathlib import Path

from shared.llm_client import get_config


def interim_enabled():
    """Whether interim output is on (config output.interim, default True)."""
    return bool(get_config().get("output", {}).get("interim", True))


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


def save_interim(filename, text):
    """Write an interim artifact; returns its path or None. Never raises."""
    if not interim_enabled() or not text:
        return None
    try:
        path = interim_dir() / filename
        path.write_text(text, encoding="utf-8")
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
            lines.append(f"- **{c['name']}**{role}: {c.get('description', '')}")
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
    return "\n".join(lines)
