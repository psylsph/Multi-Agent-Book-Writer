"""
Seed Review
Checks, before anything is written, that the seed can carry the book's length.

An LLM reads the seed and lists the scenes it really supplies (each backed by a
quote that must be found in the seed), the story's timespan, and the length
the material would naturally fill. Code does the arithmetic (words per scene,
chapters per scene) rather than trusting the model with it.

On a terminal the review is a loop: the author answers the critique's
questions (and adds notes), the LLM expands the seed with them, the author
keeps or undoes the expansion, and the expanded seed is critiqued again,
until the author is done. Code checks that an expansion kept every character,
constraint and outline chapter. A kept expansion becomes the run's MASTER
seed (state/seed.json and seed.md) that every later stage and any restart
uses. Finally the author keeps the requested size, takes the recommended one,
chooses their own, or stops.

Modes (book.seed_review, or --seed-review): ask (default; falls back to warn
without a terminal), warn (report only, never prompts), off.
"""

import difflib
import sys
from pathlib import Path

from agents.planner import extract_seed_outline
from shared import prompts
from shared.llm_client import AbortRun, generate_prose, generate_with_wait, \
    get_config
from shared.llm_utils import (clean_llm_text, extract_json,
                              extract_seed_characters, extract_seed_extras)
from shared.output import atomic_write_text, save_interim
from shared.resume import save_state
from shared.web_search import quote_in_text

VERDICTS = ("too_long", "about_right", "too_short")
# A scene in commercial fiction usually runs about 1,000-3,000 words. Far
# above that per scene the chapters will be padded; far below, rushed.
STRETCHED_WORDS_PER_SCENE = 4000
CRAMPED_WORDS_PER_SCENE = 700
SEED_CHARS = 24000

_input = input          # replaced in tests
_INPUT = {"closed": False}  # stdin hit EOF: no one is there to answer


def mode():
    """'ask', 'warn' or 'off'. 'ask' becomes 'warn' without a terminal."""
    value = str(get_config()["book"]["seed_review"]).strip().lower()
    if value in ("false", "no", "none"):
        value = "off"
    if value not in ("ask", "warn", "off"):
        print(f"[SEED REVIEW] Unknown book.seed_review '{value}'; using 'ask'.")
        value = "ask"
    if value == "ask" and not interactive():
        value = "warn"
    return value


def interactive():
    try:
        return sys.stdin.isatty() and sys.stdout.isatty()
    except (AttributeError, ValueError):
        return False


def _ask(question, default=""):
    """input() that treats Ctrl-D / a closed stdin as 'use the default'."""
    try:
        answer = _input(question).strip()
    except EOFError:
        print()
        _INPUT["closed"] = True
        return default
    return answer or default


# ------------------------------------------------------------- the LLM call

def _size_text(size):
    if not size:
        return "not fixed: recommend one"
    return (f"{size['chapters']} chapters of about {size['words_per_chapter']:,} "
            f"words (about {size['chapters'] * size['words_per_chapter']:,} "
            f"words in all), from {size['source']}")


def _clarification_block(clarifications):
    if not clarifications:
        return ""
    lines = [f"- Q: {c['question']}\n  A: {c['answer']}"
             + ("" if c.get("answered") else " (assumed; the author did not answer)")
             for c in clarifications]
    return ("The author has answered these questions; treat the answers as "
            "part of the brief:\n" + "\n".join(lines) + "\n\n")


def build_prompt(seed_text, size, questions, clarifications=()):
    return f"""Assess this creative brief before a book is planned from it.

REQUESTED SIZE: {_size_text(size)}

Rules of thumb:
- A scene in commercial fiction usually runs 1,000-3,000 words; a chapter holds one to three scenes.
- Length must come from story material: distinct scenes, events, turning points, subplots and character arcs. Extra description, repetition and recap are padding.
- A short story timespan is fine if it holds enough distinct events: judge the material, not the hours.

Return ONLY JSON:
{{
  "stated": {{"total_words": null, "words_per_chapter": null, "chapters_min": null, "chapters_max": null, "quote": "the exact words in the brief that state a length, or empty"}},
  "timespan": "how much story time the brief covers, e.g. 'Friday evening to Sunday evening (about 48 hours)'",
  "scenes": [{{"what": "one line", "quote": "a short phrase copied exactly from the brief that supplies this scene"}}],
  "threads": ["each subplot or character arc the brief gives"],
  "natural_words": {{"low": 0, "high": 0}},
  "verdict": "too_long | about_right | too_short",
  "reasons": ["specific reasons, each tied to something in the brief"],
  "recommended": {{"chapters": 0, "words_per_chapter": 0}},
  "to_fill_requested": ["what the brief would need to add to support the requested size without padding"],
  "questions": [{{"question": "...", "why": "how the answer changes the book", "default": "what will be assumed if the author does not answer"}}]
}}

- "stated": fill in only what the brief itself says about length (null otherwise). A total word count goes in total_words; a per-chapter length in words_per_chapter.
- "scenes": every distinct scene or event the brief describes or clearly implies, in story order. Do not split one event into several or invent any.
- "verdict" is about the REQUESTED size: too_long = more words than the material supports (it would need padding); too_short = more material than fits (it would be rushed). With no requested size, judge your own recommendation (about_right).
- "recommended": the size the material supports as the brief stands now.
- "questions": at most {questions}; none if the brief settles everything. Ask only what changes the book's shape: missing scenes or subplots that could carry the length, an unclear ending, point of view, how much time the explicit or key scenes should take.

{_clarification_block(clarifications)}Creative brief:
\"\"\"{seed_text[:SEED_CHARS]}\"\"\"
"""


def _int(value, lo=1, hi=10**7):
    try:
        n = int(float(value))
    except (TypeError, ValueError):
        return None
    return n if lo <= n <= hi else None


def parse_assessment(raw, seed_text, max_questions):
    """The model's JSON -> a checked assessment dict. Quotes are verified
    against the seed: a scene whose quote is not in the seed is kept but
    marked, and a stated length whose quote is not in the seed is dropped."""
    data = extract_json(raw, expect="object")
    if not isinstance(data, dict):
        raise ValueError("the assessment is not a JSON object")

    stated = data.get("stated") if isinstance(data.get("stated"), dict) else {}
    stated_quote = str(stated.get("quote") or "").strip()
    stated_ok = bool(stated_quote) and quote_in_text(stated_quote, seed_text,
                                                     min_chars=8)
    out_stated = {k: (_int(stated.get(k)) if stated_ok else None)
                  for k in ("total_words", "words_per_chapter",
                            "chapters_min", "chapters_max")}
    out_stated["quote"] = stated_quote if stated_ok else ""

    scenes = []
    for s in data.get("scenes") or []:
        if isinstance(s, str):
            s = {"what": s, "quote": ""}
        if not isinstance(s, dict) or not str(s.get("what", "")).strip():
            continue
        quote = str(s.get("quote") or "").strip()
        scenes.append({"what": " ".join(str(s["what"]).split()),
                       "quote": quote,
                       "found": bool(quote) and quote_in_text(quote, seed_text,
                                                              min_chars=8)})

    natural = data.get("natural_words") if isinstance(
        data.get("natural_words"), dict) else {}
    rec = data.get("recommended") if isinstance(
        data.get("recommended"), dict) else {}
    verdict = str(data.get("verdict") or "").strip().lower()
    questions = []
    for q in data.get("questions") or []:
        if isinstance(q, str):
            q = {"question": q}
        if isinstance(q, dict) and str(q.get("question", "")).strip():
            questions.append({k: " ".join(str(q.get(k) or "").split())
                              for k in ("question", "why", "default")})

    def strings(key):
        return [" ".join(str(x).split()) for x in data.get(key) or []
                if str(x).strip()]

    return {
        "stated": out_stated,
        "timespan": " ".join(str(data.get("timespan") or "").split()),
        "scenes": scenes,
        "threads": strings("threads"),
        "natural_words": {"low": _int(natural.get("low")),
                          "high": _int(natural.get("high"))},
        "verdict": verdict if verdict in VERDICTS else "about_right",
        "reasons": strings("reasons"),
        "recommended": {"chapters": _int(rec.get("chapters"), 1, 200),
                        "words_per_chapter": _int(rec.get("words_per_chapter"),
                                                  300, 20000)},
        "to_fill_requested": strings("to_fill_requested"),
        "questions": questions[:max_questions],
    }


def assess(seed_text, size, max_questions=5, clarifications=()):
    """One assessment call. Raises on an unusable reply (and on an outage)."""
    raw = generate_with_wait(
        build_prompt(seed_text, size, max_questions, clarifications),
        system=prompts.SEED_REVIEWER, agent="seed_reviewer", json_mode=True)
    return parse_assessment(raw, seed_text, max_questions)


# ------------------------------------------------------- sizes and arithmetic

def requested_size(num_chapters, outline_chapters, stated, words_per_chapter):
    """What the author asked for, or None when nothing fixes the size.

    In priority order: -c on the command line, the seed's own outline, a
    total length stated in the seed. (A stated chapter RANGE is a request
    too, but a loose one; it is applied to the recommendation instead.)"""
    wpc = (stated or {}).get("words_per_chapter") or words_per_chapter
    if num_chapters:
        return {"chapters": num_chapters, "words_per_chapter": wpc,
                "source": "your --chapters option"}
    if outline_chapters:
        return {"chapters": outline_chapters, "words_per_chapter": wpc,
                "source": "the seed's own outline"}
    total = (stated or {}).get("total_words")
    if total:
        return {"chapters": max(1, round(total / wpc)), "words_per_chapter": wpc,
                "source": "the length stated in the seed"}
    return None


def recommended_size(assessment, words_per_chapter):
    """The model's recommendation, kept inside the configured chapter limits
    and any chapter range the seed states."""
    book = get_config()["book"]
    lo = int(book["min_chapters"])
    hi = max(lo, int(book["max_chapters"]))
    stated = assessment["stated"]
    lo = max(lo, stated.get("chapters_min") or 0)
    if stated.get("chapters_max"):
        hi = min(hi, stated["chapters_max"])
    hi = max(hi, lo)
    rec = assessment["recommended"]
    wpc = rec.get("words_per_chapter") or stated.get(
        "words_per_chapter") or words_per_chapter
    chapters = rec.get("chapters")
    if not chapters:
        natural = assessment["natural_words"]
        mid = ((natural.get("low") or 0) + (natural.get("high") or 0)) / 2
        chapters = round(mid / wpc) if mid else None
    if not chapters:
        return None
    return {"chapters": min(max(chapters, lo), hi), "words_per_chapter": wpc,
            "source": "the seed review's recommendation"}


def arithmetic(assessment, size):
    """Facts about the size computed in code. Returns (lines, flag) where flag
    is 'stretched', 'cramped' or None."""
    if not size:
        return [], None
    found = [s for s in assessment["scenes"] if s["found"]]
    total = size["chapters"] * size["words_per_chapter"]
    lines, flag = [], None
    if not found:
        return ["No scene quotes could be found in the seed, so the size "
                "could not be checked against the material."], None
    per_scene = total / len(found)
    lines.append(f"The seed supplies {len(found)} distinct scenes/events. At "
                 f"{total:,} words that is about {per_scene:,.0f} words per "
                 "scene (a scene usually runs 1,000-3,000).")
    if per_scene > STRETCHED_WORDS_PER_SCENE:
        flag = "stretched"
    elif per_scene < CRAMPED_WORDS_PER_SCENE:
        flag = "cramped"
    if size["chapters"] > len(found) * 1.5:
        lines.append(f"{size['chapters']} chapters for {len(found)} scenes: "
                     "many chapters would have less than one scene of story.")
        flag = flag or "stretched"
    return lines, flag


def needs_decision(assessment, flag):
    return assessment["verdict"] != "about_right" or flag is not None


# ----------------------------------------------------------------- reporting

def _size_line(size):
    return (f"{size['chapters']} chapters x {size['words_per_chapter']:,} words "
            f"(about {size['chapters'] * size['words_per_chapter']:,})")


def report(assessment, requested, recommended, notes, flag):
    """Human-readable assessment (printed and saved)."""
    verdict = {"too_long": "TOO LONG for the material (it would need padding)",
               "too_short": "TOO SHORT for the material (it would be rushed)",
               "about_right": "about right"}[assessment["verdict"]]
    lines = ["# Seed review", ""]
    lines.append(f"**Requested:** {_size_line(requested)}, from "
                 f"{requested['source']}" if requested else
                 "**Requested:** nothing fixes the size")
    lines.append(f"**Verdict:** {verdict}")
    if flag:
        lines.append(f"**Arithmetic:** the requested size looks {flag}")
    if recommended:
        lines.append(f"**Recommended:** {_size_line(recommended)}")
    natural = assessment["natural_words"]
    if natural.get("low") and natural.get("high"):
        lines.append(f"**Natural length of the material:** about "
                     f"{natural['low']:,}-{natural['high']:,} words")
    if assessment["timespan"]:
        lines.append(f"**Story timespan:** {assessment['timespan']}")
    lines.append("")
    for title, items in (("Why", assessment["reasons"]),
                         ("Checks", notes),
                         ("To support the requested size, the seed would need",
                          assessment["to_fill_requested"]
                          if assessment["verdict"] == "too_long" else []),
                         ("Subplots and arcs", assessment["threads"])):
        if items:
            lines += [f"## {title}", ""] + [f"- {x}" for x in items] + [""]
    if assessment["scenes"]:
        lines += ["## Scenes the seed supplies", ""]
        lines += [f"{i}. {s['what']}" + ("" if s["found"] else
                                         " *(quote not found in the seed)*")
                  for i, s in enumerate(assessment["scenes"], 1)]
        lines.append("")
    return "\n".join(lines)


def clarifications_markdown(clarifications):
    lines = ["# Clarifications", "",
             "Add these to your seed if you want them kept for a fresh run.", ""]
    for c in clarifications:
        lines.append(f"- **{c['question']}** {c['answer']}"
                     + ("" if c.get("answered") else " *(assumed)*"))
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------- interaction

def ask_questions(questions):
    """Ask each question; Enter keeps the stated assumption, 'skip' accepts
    the assumptions for the rest. Returns the clarifications."""
    out = []
    if not questions:
        return out
    print(f"\n[SEED REVIEW] {len(questions)} question(s). Type your answer, "
          "or press Enter to accept the suggestion shown; type 'skip' to "
          "accept all the rest. Answers and accepted suggestions are built "
          "into the seed.")
    skipping = False
    for i, q in enumerate(questions, 1):
        answer = ""
        if not skipping:
            print(f"\n  {i}. {q['question']}")
            if q.get("why"):
                print(f"     (why it matters: {q['why']})")
            if q.get("default"):
                print(f"     [suggestion, used if you press Enter: "
                      f"{q['default']}]")
            answer = _ask("     > ")
            if answer.lower() == "skip":
                skipping, answer = True, ""
        if answer:
            out.append({"question": q["question"], "answer": answer,
                        "answered": True})
        elif q.get("default"):
            out.append({"question": q["question"], "answer": q["default"],
                        "answered": False})
    return out


def _choose_custom(current):
    chapters = _int(_ask(f"  Chapters [{current['chapters']}]: ",
                         str(current["chapters"])), 1, 200)
    wpc = _int(_ask(f"  Words per chapter [{current['words_per_chapter']}]: ",
                    str(current["words_per_chapter"])), 300, 20000)
    if not chapters or not wpc:
        print("  Not a usable number; keeping "
              f"{_size_line(current)}.")
        return current
    return {"chapters": chapters, "words_per_chapter": wpc,
            "source": "your choice"}


def decide(requested, recommended):
    """Ask which size to use. Returns a size dict, or None to stop."""
    options = []
    if requested:
        options.append(("k", f"keep the requested {_size_line(requested)}",
                        requested))
    if recommended and recommended != requested:
        options.append(("r", f"use the recommended {_size_line(recommended)}",
                        recommended))
    default = options[0][0] if options else "c"
    print("\n[SEED REVIEW] Which size should the book be?")
    for key, text, _ in options:
        print(f"  [{key}] {text}" + ("  (Enter)" if key == default else ""))
    print("  [c] choose your own")
    print("  [s] stop here: no book is written; edit the seed and rerun (an "
          "expanded seed is kept)")
    while True:
        choice = _ask("  > ", default).lower()[:1]
        for key, _, size in options:
            if choice == key:
                return size
        if choice == "c":
            return _choose_custom(requested or recommended or {
                "chapters": int(get_config()["book"]["num_chapters"]),
                "words_per_chapter": int(
                    get_config()["book"]["words_per_chapter"])})
        if choice == "s":
            return None
        print("  Please type one of the letters shown.")


# ------------------------------------------------------- expanding the seed

# An expansion shorter than this has dropped the author's material.
EXPANSION_MIN_RATIO = 0.9
SHOWN_CHANGES = 40      # changed lines printed after an expansion


def master_path():
    """The human-readable copy of the master seed (reusable with --seed)."""
    return Path(get_config()["output"]["directory"]) / "seed.md"


def build_expansion_prompt(seed_text, answers):
    lines = "\n".join(
        f"- Q: {a['question']}\n  A"
        + ("" if a.get("answered", True)
           else " (the author accepted this suggestion)")
        + f": {a['answer']}" for a in answers)
    return f"""Expand the creative brief below with the author's answers and notes.

THE AUTHOR'S ANSWERS AND NOTES
{lines}

Rules:
- Keep EVERY line of the brief as written. You may add to a section, add bullets, or add a new section; never delete, shorten or reword what is there.
- Keep the brief's Markdown structure: the '# Title' line, '## Section' headings, '- **Name** - description' character bullets, and, if the brief has an outline, its '- Chapter N: Title - what happens' lines.
- Put each answer where it belongs: a new character under Characters, a new rule under Constraints, a new event in the Outline or the premise.
- Add only what the answers and notes call for or plainly imply. Do not invent other characters, twists or an ending.
- Write brief notes, not story prose.

Return ONLY the complete expanded brief in Markdown.

THE BRIEF
\"\"\"{seed_text}\"\"\"
"""


def expand(seed_text, answers):
    """The seed rewritten with `answers` built in (one LLM call)."""
    text = clean_llm_text(generate_prose(
        build_expansion_prompt(seed_text, answers),
        system=prompts.SEED_EXPANDER, agent="seed_reviewer"))
    lines = text.splitlines()
    while lines and lines[0].strip() in ('"""', ""):    # echoed delimiters
        lines.pop(0)
    while lines and lines[-1].strip() in ('"""', ""):
        lines.pop()
    return "\n".join(lines).strip() + "\n"


def _flat(text):
    return " ".join(str(text).lower().split())


def lost_material(before, after):
    """What `after` dropped from the author's `before`, as reasons (empty
    when nothing was lost). Checked in code, not trusted to the model: the
    characters, constraints and outline chapters are parsed verbatim later,
    so losing one would change the book."""
    problems = []
    words_before, words_after = len(before.split()), len(after.split())
    if words_after < EXPANSION_MIN_RATIO * words_before:
        problems.append(f"it is shorter ({words_before:,} -> "
                        f"{words_after:,} words)")
    kept = {c["name"].lower() for c in extract_seed_characters(after)}
    gone = [c["name"] for c in extract_seed_characters(before)
            if c["name"].lower() not in kept]
    if gone:
        problems.append("character(s) no longer listed: " + ", ".join(gone))
    constraints, _ = extract_seed_extras(before)
    flat_after = _flat(after)
    changed = [c for c in constraints if _flat(c) not in flat_after]
    if changed:
        problems.append(f"{len(changed)} constraint(s) reworded or removed "
                        f"(e.g. '{changed[0][:60]}')")
    outline_before = len(extract_seed_outline(before))
    outline_after = len(extract_seed_outline(after))
    if outline_after < outline_before:
        problems.append(f"outline chapters dropped ({outline_before} -> "
                        f"{outline_after})")
    return problems


def show_changes(before, after, rounds):
    """Print the lines an expansion added or changed."""
    changes = [line for line in difflib.unified_diff(
        before.splitlines(), after.splitlines(), lineterm="", n=0)
        if line.startswith(("+", "-")) and not line.startswith(("+++", "---"))]
    print(f"\n[SEED REVIEW] The expanded seed ({len(before.split()):,} -> "
          f"{len(after.split()):,} words). What changed:")
    for line in changes[:SHOWN_CHANGES]:
        print("  " + line)
    if len(changes) > SHOWN_CHANGES:
        print(f"  ... and {len(changes) - SHOWN_CHANGES} more changed lines "
              f"(full text: interim/seed_round_{rounds:02d}.md)")


def save_master(original, seed, pending, rounds):
    """Save the expanded seed as the run's master seed: in the state store
    (what a restart uses; raises StateWriteError) and as seed.md beside the
    book (best effort; reusable with --seed)."""
    save_state("seed", {"original": original, "current": seed,
                        "pending": pending, "rounds": rounds})
    try:
        atomic_write_text(master_path(), seed)
    except OSError as e:
        print(f"[SEED REVIEW] Could not write {master_path()}: {e}")


def _round_menu(assessment, flag):
    """[a]nswer and expand, [d]one, or [s]top."""
    open_points = bool(assessment["questions"]) or needs_decision(
        assessment, flag)
    default = "a" if open_points else "d"
    marks = {k: "  (Enter)" if k == default else "" for k in "ads"}
    print("\n[SEED REVIEW] What next?")
    print("  [a] " + ("answer the questions and/or add your own notes"
                      if assessment["questions"] else "add your own notes")
          + "; the seed is then expanded with them" + marks["a"])
    print("  [d] done: the seed is ready; go on to the book's size"
          + marks["d"])
    print("  [s] stop here: anything expanded so far is saved; rerun to "
          "continue" + marks["s"])
    while True:
        choice = _ask("  > ", default).lower()[:1]
        if choice in ("a", "d", "s"):
            return choice
        print("  Please type a, d or s.")


def _keep_expansion():
    print("\n  [k] keep it and review the expanded seed  (Enter)")
    print("  [u] undo: keep the previous version (your answers are passed on)")
    while True:
        choice = _ask("  > ", "k").lower()[:1]
        if choice in ("k", "u"):
            return choice == "k"
        print("  Please type k or u.")


# ------------------------------------------------------------------- the step

def run_seed_review(seed_text, num_chapters=None, original=None,
                    clarifications=()):
    """Review the seed, grow it with the author, and settle the book's size.

    In ask mode each round critiques the current seed; the author answers
    its questions (and adds notes), the LLM expands the seed with them, the
    author keeps or undoes that, and the expanded seed is critiqued again,
    until the author is done. Every kept expansion becomes the run's master
    seed (state + seed.md), so a restart continues from it.

    seed_text: the seed to review (on resume: the master seed so far)
    original: the author's own seed, when seed_text is an expansion of it
    clarifications: answers not yet built into seed_text (resume)

    Returns {"seed": the seed to write the book from, "chapters": int or
    None, "words_per_chapter": int, "clarifications": [...], "stop": bool}.
    chapters None means "let the planner decide as before" (review off or
    failed with nothing requested). Never raises except AbortRun.
    """
    book = get_config()["book"]
    wpc = int(book["words_per_chapter"])
    original = original or seed_text
    result = {"seed": seed_text, "chapters": None, "words_per_chapter": wpc,
              "clarifications": list(clarifications), "stop": False}
    how = mode()
    if how == "off":
        return result
    max_q = int(book["seed_questions"]) if how == "ask" else 0
    print("[SEED REVIEW] Checking that the seed can carry the book's length...")

    def checked(seed, pending):
        # the call is told the size we already know (-c or the outline); a
        # length stated in the seed is found by the model itself
        outline = len(extract_seed_outline(seed))
        hint = requested_size(num_chapters, outline, {}, wpc)
        a = assess(seed, hint, max_q, pending)
        req = requested_size(num_chapters, outline, a["stated"], wpc)
        rec = recommended_size(a, (req or {}).get("words_per_chapter", wpc))
        notes, flag = arithmetic(a, req or rec)
        return a, req, rec, notes, flag

    seed, pending = seed_text, list(clarifications)
    rounds = 0
    _INPUT["closed"] = False
    assessment = None
    while True:
        try:
            assessment, requested, recommended, notes, flag = checked(
                seed, pending)
        except AbortRun:
            raise
        except Exception as e:
            print(f"[SEED REVIEW] Could not review the seed ({e}); planning "
                  "as before.")
            result.update(seed=seed, clarifications=pending)
            return result
        text = report(assessment, requested, recommended, notes, flag)
        print("\n" + text)
        save_interim("seed_review.md", text + (
            "\n" + clarifications_markdown(pending) if pending else ""))
        if how != "ask":
            break

        choice = _round_menu(assessment, flag)
        if _INPUT["closed"]:
            # every prompt would return its default ("answer", "keep") and
            # the loop would expand the seed forever: treat it as done
            print("[SEED REVIEW] No more input; the seed stands as it is.")
            break
        if choice == "s":
            result.update(seed=seed, clarifications=pending, stop=True)
            return result
        if choice == "d":
            break
        # an accepted assumption is an answer too: build it in
        answers = ask_questions(assessment["questions"])
        note = _ask("\n  Anything else to add or change in the seed? "
                    "(Enter for nothing)\n  > ")
        if note:
            answers.append({"question": "The author adds", "answer": note,
                            "answered": True})
        if not answers or _INPUT["closed"]:
            print("[SEED REVIEW] Nothing to add; the seed stands as it is.")
            break

        to_build = pending + answers
        rounds += 1
        print(f"\n[SEED REVIEW] Round {rounds}: expanding the seed with your "
              "answers...")
        try:
            expanded = expand(seed, to_build)
            problems = lost_material(seed, expanded)
        except AbortRun:
            raise
        except Exception as e:
            problems = [f"the expansion failed ({e})"]
        if problems:
            print("[SEED REVIEW] The expansion was not used: "
                  + "; ".join(problems) + ". Your answers are kept and "
                  "passed on.")
            pending = to_build
            continue
        save_interim(f"seed_round_{rounds:02d}.md", expanded)
        show_changes(seed, expanded, rounds)
        if _keep_expansion():
            seed, pending = expanded, []
            save_master(original, seed, pending, rounds)
            print(f"[SEED REVIEW] The expanded seed is now the master seed "
                  f"({master_path()}). Reviewing it again...")
        else:
            pending = to_build
            if seed != original:
                save_master(original, seed, pending, rounds)
            print("[SEED REVIEW] Kept the previous version; your answers are "
                  "passed on.")

    result.update(seed=seed, clarifications=pending)
    if how == "ask" and (needs_decision(assessment, flag) or not requested):
        chosen = decide(requested, recommended)
        if chosen is None:
            result["stop"] = True
            return result
    else:
        chosen = requested or recommended
        if requested and needs_decision(assessment, flag):
            print("[SEED REVIEW] Warning: the requested size does not fit the "
                  "seed (see above). Continuing with it; rerun on a terminal "
                  "with book.seed_review: ask to choose another.")
    if chosen:
        result["chapters"] = chosen["chapters"]
        result["words_per_chapter"] = chosen["words_per_chapter"]
        print(f"[SEED REVIEW] Book size: {_size_line(chosen)}.")
    save_state("plan", {
        "chapters": result["chapters"],
        "words_per_chapter": result["words_per_chapter"],
        "clarifications": result["clarifications"]})
    return result
