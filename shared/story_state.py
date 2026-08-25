"""Story-state tracking: the structured chronology used for continuity.

After each chapter is written, the writer agent extracts structured facts
(when/where, who is present, first meetings, relationship changes, deaths,
injuries, secrets revealed). This module merges those per-chapter states
into a cumulative story state, renders it for prompts, and exposes it to
the deterministic checks. Pure functions - unit-testable without a model.
"""

import re

EVENT_TYPES = ("first_meeting", "relationship_change", "death", "injury",
               "departure", "secret_revealed")


def normalize_state(number, title, data):
    """Coerce extracted JSON into a validated per-chapter state dict."""
    events = []
    for ev in data.get("events") or []:
        if not isinstance(ev, dict):
            continue
        etype = str(ev.get("type", "")).strip()
        if etype not in EVENT_TYPES:
            continue
        who = [str(w).strip() for w in (ev.get("who") or [])
               if str(w).strip()]
        if not who:
            continue
        events.append({"type": etype, "who": who,
                       "detail": str(ev.get("detail", "")).strip()})
    return {
        "number": number,
        "title": title,
        "summary": str(data.get("summary", "")).strip(),
        "time": str(data.get("time", "")).strip(),
        "location": str(data.get("location", "")).strip(),
        "present": [str(p).strip() for p in (data.get("present") or [])
                    if str(p).strip()],
        "events": events,
    }


def minimal_state(number, title, summary_text):
    """Fallback state when extraction fails: summary only, no facts."""
    return {"number": number, "title": title, "summary": summary_text,
            "time": "", "location": "", "present": [], "events": []}


def pair_key(a, b):
    """Stable key for a pair of names."""
    return "+".join(sorted([a.lower(), b.lower()]))


def merge_states(chronology, upto=None):
    """Merge per-chapter states into a cumulative story state.

    Args:
        chronology: {number: state} dict
        upto: only include chapters with number < upto (i.e. strictly
              before the chapter being reviewed); None = all

    Returns {"alive", "dead", "met_pairs", "relationships", "timeline",
             "secrets", "injuries"}.
    """
    dead = {}           # name -> death chapter
    met = {}            # pair_key -> first chapter
    relationships = {}  # pair_key -> (detail, chapter)
    secrets = []        # {"who", "detail", "chapter"}
    injuries = []       # {"who", "detail", "chapter"}
    timeline = []       # {"chapter", "time", "location"}
    seen_names = set()

    def register_met(who, n):
        for i in range(len(who)):
            for j in range(i + 1, len(who)):
                key = pair_key(who[i], who[j])
                if key not in met:
                    met[key] = n

    for n in sorted(k for k in chronology if upto is None or k < upto):
        state = chronology[n]
        for name in state.get("present", []):
            seen_names.add(name)
        if state.get("time") or state.get("location"):
            timeline.append({"chapter": n, "time": state.get("time", ""),
                             "location": state.get("location", "")})
        for ev in state.get("events", []):
            who = ev["who"]
            detail = ev.get("detail", "")
            if ev["type"] == "death":
                for name in who:
                    dead[name] = n
            elif len(who) >= 2:
                # any two-person event means they have met
                register_met(who, n)
                if ev["type"] == "relationship_change":
                    for i in range(len(who)):
                        for j in range(i + 1, len(who)):
                            relationships[pair_key(who[i], who[j])] = (detail, n)
            if ev["type"] == "secret_revealed":
                secrets.append({"who": who, "detail": detail, "chapter": n})
            elif ev["type"] == "injury":
                injuries.append({"who": who, "detail": detail, "chapter": n})

    alive = sorted(seen_names - set(dead))
    return {"alive": alive, "dead": dead, "met_pairs": met,
            "relationships": relationships, "timeline": timeline,
            "secrets": secrets, "injuries": injuries}


def pretty_pair(key):
    """'lisa+stuart' -> 'Lisa & Stuart' for display."""
    return " & ".join(part.title() for part in key.split("+"))


def render_story_facts(cumulative):
    """Render the cumulative state as compact text for prompts."""
    lines = []
    if cumulative["timeline"]:
        lines.append("Timeline so far: "
                     + "; ".join(f"Ch{t['chapter']}: {t['time'] or '?'}"
                                 f" @ {t['location'] or '?'}"
                                 for t in cumulative["timeline"]))
    if cumulative["dead"]:
        deaths = ", ".join(f"{name} (died Ch{n})"
                           for name, n in cumulative["dead"].items())
        lines.append(f"DEAD (must not act alive later): {deaths}")
    if cumulative["alive"]:
        lines.append(f"Alive/known so far: {', '.join(cumulative['alive'])}")
    if cumulative["met_pairs"]:
        met = ", ".join(f"{pretty_pair(key)} (Ch{n})"
                        for key, n in sorted(
                            cumulative["met_pairs"].items(),
                            key=lambda kv: kv[1]))
        lines.append(f"Have already met (do NOT play as strangers): {met}")
    if cumulative["relationships"]:
        rel = ", ".join(f"{pretty_pair(key)}: {detail} (since Ch{n})"
                        for key, (detail, n) in sorted(
                            cumulative["relationships"].items(),
                            key=lambda kv: kv[1][1]))
        lines.append(f"Relationships (state must not regress): {rel}")
    if cumulative["secrets"]:
        sec = "; ".join(f"Ch{s['chapter']}: {' & '.join(s['who'])} - "
                        f"{s['detail']}" for s in cumulative["secrets"])
        lines.append(f"Secrets revealed (characters may now know): {sec}")
    if cumulative["injuries"]:
        inj = "; ".join(f"Ch{i['chapter']}: {' & '.join(i['who'])} - "
                        f"{i['detail']}" for i in cumulative["injuries"])
        lines.append(f"Injuries (must still affect the character): {inj}")
    return "\n".join(lines) if lines else "(first chapter - no history yet)"
