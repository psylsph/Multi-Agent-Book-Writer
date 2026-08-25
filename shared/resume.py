"""Resume support: load persisted state from output/interim/ so a crashed
or interrupted run can continue from where it stopped.

The app writes three kinds of artifacts:
  - JSON snapshots of the context (bible, outline, summaries, chronology)
  - Per-chapter text artifacts (lore/draft/edited markdown, keyed by chapter #)
  - Human-readable mirrors (story_bible.md, outline.md, etc.) for inspection

On startup, load_state() rebuilds everything from disk so the per-agent
pipelines can skip chapters they already finished.
"""

import json
from pathlib import Path

from shared.llm_client import get_config


def resume_dir():
    """Path to the interim directory."""
    return Path(get_config()["output"]["directory"]) / "interim"


def has_resume():
    """True when there is at least one snapshot from a previous run."""
    return (resume_dir() / "bible.json").exists()


def _read_json(path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"[RESUME] Warning: failed to load {path}: {e}")
        return None


def _strip_heading(text):
    """Remove a leading '## Chapter N: Title\\n\\n' if present."""
    parts = text.split("\n\n", 1)
    return parts[1] if len(parts) == 2 else text


def _strip_lore_heading(text):
    """Remove a leading '# Lore Brief - Chapter N: Title\\n\\n' if present."""
    parts = text.split("\n\n", 1)
    return parts[1] if len(parts) == 2 else text


def load_state():
    """Read all persisted state from output/interim/.

    Returns a dict with keys:
      bible, chapters, summaries, chronology  (None if absent)
      research, drafts                          ({chapter_number: text})
      final                                     ([(chapter_number, text), ...]
                                                in chapter order)
      completed_chapters                        (set of ints where edited)
    Missing keys are None or empty - callers decide whether to skip that
    step or rebuild it from scratch.
    """
    out = {"bible": None, "chapters": None, "summaries": None,
           "chronology": None, "research": {}, "drafts": {}, "final": [],
           "completed_chapters": set()}
    d = resume_dir()
    if not d.exists():
        return out

    for filename, key in [("bible.json", "bible"),
                          ("outline.json", "chapters"),
                          ("summaries.json", "summaries"),
                          ("chronology.json", "chronology")]:
        path = d / filename
        if path.exists():
            out[key] = _read_json(path)

    # JSON object keys are always strings; coerce int-keyed dicts back
    if out["summaries"]:
        out["summaries"] = {int(k): v for k, v in out["summaries"].items()}
    if out["chronology"]:
        out["chronology"] = {int(k): v for k, v in out["chronology"].items()}

    for path in sorted(d.glob("lore_chapter_*.md")):
        try:
            n = int(path.stem.split("_")[-1])
        except ValueError:
            continue
        out["research"][n] = _strip_lore_heading(path.read_text(encoding="utf-8"))

    for path in sorted(d.glob("draft_chapter_*.md")):
        try:
            n = int(path.stem.split("_")[-1])
        except ValueError:
            continue
        out["drafts"][n] = _strip_heading(path.read_text(encoding="utf-8"))

    finals = []
    for path in sorted(d.glob("edited_chapter_*.md")):
        try:
            n = int(path.stem.split("_")[-1])
        except ValueError:
            continue
        finals.append((n, path.read_text(encoding="utf-8")))
    out["final"] = sorted(finals, key=lambda nv: nv[0])
    out["completed_chapters"] = {n for n, _ in finals}
    return out


def summarize_for_log(state):
    """One-line summary of what was loaded, for the startup banner."""
    bits = []
    if state["bible"]:
        bits.append(f"bible '{state['bible'].get('title', '?')}'")
    if state["chapters"]:
        bits.append(f"plan {len(state['chapters'])} chapters")
    if state["drafts"]:
        bits.append(f"{len(state['drafts'])} drafted")
    if state["final"]:
        bits.append(f"{len(state['final'])} edited")
    return ", ".join(bits) if bits else "nothing"
