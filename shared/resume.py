"""The run's state store, and resume from it after a crash or Ctrl-C.

Everything a later stage (or a rerun) needs is saved to <output dir>/state/
as soon as it exists: the story bible, the plan, the outline, and per-chapter
briefs, drafts, summaries, story state and edited text. One JSON file per key,
each replaced atomically. Unlike the human-readable files in interim/, these
are always written, whatever output.interim says, and a write that fails
stops the run (StateWriteError): carrying on would spend hours of model time
on work that a crash could no longer resume from.

On startup, load_state() rebuilds everything from disk so the per-agent
pipelines can skip chapters they already finished. Runs saved by older
versions, which kept their state in interim/, are still read.
"""

import json
from pathlib import Path

from shared.llm_client import AbortRun, get_config
from shared.output import atomic_write_text, strip_heading

# key -> True when the value is a {chapter number: ...} dict (JSON keys are
# strings, so these are turned back into ints when loaded)
STATE_KEYS = {
    "seed": False,         # the master seed the seed review grew, with the
                           # author's original: {original, current, pending}
    "bible": False,        # story bible (includes the seed)
    "plan": False,         # book size + clarifications from the seed review
    "outline": False,      # [{"number", "title", "summary", "part"}]
    "research": True,      # lore briefs
    "drafts": True,        # first drafts
    "summaries": True,     # rolling chapter summaries
    "chronology": True,    # structured story state per chapter
    "final": True,         # edited chapter bodies
    "unreviewed": False,   # [chapter numbers] finished without a review
}


class StateWriteError(AbortRun):
    """The run's state could not be saved (disk full, permissions...)."""


def state_dir():
    """Path to the state directory (not created)."""
    return Path(get_config()["output"]["directory"]) / "state"


def resume_dir():
    """Path to the interim directory (where older versions kept state)."""
    return Path(get_config()["output"]["directory"]) / "interim"


def save_state(key, value):
    """Save one piece of the run's state. Raises StateWriteError."""
    if key not in STATE_KEYS:
        raise ValueError(f"unknown state key '{key}'")
    path = state_dir() / f"{key}.json"
    try:
        text = json.dumps(value, indent=2, ensure_ascii=False)
        path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_text(path, text)
    except (OSError, TypeError, ValueError) as e:
        raise StateWriteError(f"could not save the run's {key} to {path} "
                              f"({e})") from e


def _legacy():
    """True when the only saved run is in the old interim/ layout."""
    return (not (state_dir() / "bible.json").exists()
            and (resume_dir() / "bible.json").exists())


def has_resume():
    """True when there is a saved run to continue: a bible, or a seed review
    that expanded the seed and stopped before the bible was built."""
    return ((state_dir() / "bible.json").exists()
            or (state_dir() / "seed.json").exists() or _legacy())


def load_plan():
    """The book size and clarifications the seed review settled, or None
    when there is none."""
    path = (resume_dir() if _legacy() else state_dir()) / "plan.json"
    return _read_json(path) if path.exists() else None


def _read_json(path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"[RESUME] Warning: failed to load {path}: {e}")
        return None


def _int_keys(value):
    """{"1": x} -> {1: x}; anything that isn't such a dict -> None."""
    if not isinstance(value, dict):
        return None
    try:
        return {int(k): v for k, v in value.items()}
    except ValueError:
        return None


def _empty():
    return {"seed": None, "bible": None, "chapters": None, "summaries": None,
            "chronology": None, "research": {}, "drafts": {}, "final": {},
            "unreviewed": set()}


def load_state():
    """Read the saved run.

    Returns a dict with keys:
      seed                                      ({original, current,
                                                pending, rounds} or None)
      bible, chapters, summaries, chronology  (None if absent)
      research, drafts, final                   ({chapter_number: text})
      unreviewed                                (set of chapter numbers)
    Missing keys are None or empty - callers decide whether to skip that
    step or rebuild it from scratch.
    """
    if _legacy():
        print("[RESUME] Reading a run saved by an older version "
              f"({resume_dir()}/); copying it to {state_dir()}/.")
        out = _load_legacy()
        _migrate(out)
        return out
    out = _empty()
    d = state_dir()
    if not d.exists():
        return out
    for key, by_chapter in STATE_KEYS.items():
        path = d / f"{key}.json"
        if key == "plan" or not path.exists():
            continue
        value = _read_json(path)
        if by_chapter:
            value = _int_keys(value)
            if value is None:
                continue
        target = "chapters" if key == "outline" else key
        if key == "unreviewed":
            value = {n for n in value or () if isinstance(n, int)}
        out[target] = value
    return out


def _load_legacy():
    """load_state() for runs saved in interim/ by older versions."""
    out = _empty()
    d = resume_dir()
    for filename, key in [("bible.json", "bible"),
                          ("outline.json", "chapters"),
                          ("summaries.json", "summaries"),
                          ("chronology.json", "chronology")]:
        path = d / filename
        if path.exists():
            out[key] = _read_json(path)

    # JSON object keys are always strings; coerce int-keyed dicts back
    for key in ("summaries", "chronology"):
        if out[key]:
            out[key] = _int_keys(out[key])

    for prefix, key in (("lore", "research"), ("draft", "drafts"),
                        ("edited", "final")):
        for path in sorted(d.glob(f"{prefix}_chapter_*.md")):
            try:
                n = int(path.stem.split("_")[-1])
            except ValueError:
                continue
            out[key][n] = strip_heading(path.read_text(encoding="utf-8"))
    return out


def _migrate(state):
    """Save a run loaded from the old layout into state/, so later saves and
    resumes all use one place. (A resumed run skips the architect, so
    nothing else would ever write state/bible.json.)"""
    if not state["bible"]:
        return      # unreadable: main refuses to resume it anyway
    plan = load_plan()
    if plan is not None:
        save_state("plan", plan)
    for key, value in (("outline", state["chapters"]),
                       ("research", state["research"]),
                       ("drafts", state["drafts"]),
                       ("summaries", state["summaries"]),
                       ("chronology", state["chronology"]),
                       ("final", state["final"])):
        if value:
            save_state(key, value)
    save_state("bible", state["bible"])     # last: it marks the new layout


def summarize_for_log(state):
    """One-line summary of what was loaded, for the startup banner."""
    bits = []
    if (state.get("seed") or {}).get("current"):
        bits.append(f"a seed expanded in {state['seed'].get('rounds', '?')} "
                    "review round(s)")
    if state["bible"]:
        bits.append(f"bible '{state['bible'].get('title', '?')}'")
    if state["chapters"]:
        bits.append(f"plan {len(state['chapters'])} chapters")
    if state["drafts"]:
        bits.append(f"{len(state['drafts'])} drafted")
    if state["final"]:
        bits.append(f"{len(state['final'])} edited")
    return ", ".join(bits) if bits else "nothing"
