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
without a terminal), warn (report only, never prompts), auto (accept every
suggested answer, expand, and review again until the reviewer asks nothing
more), off.
"""

import difflib
import sys
from pathlib import Path

from agents.planner import extract_seed_outline
from shared import prompts
from shared.llm_client import AbortRun, generate_prose, generate_with_wait, \
    get_config
from shared.llm_utils import (clean_llm_text, extract_json,
                              extract_seed_characters, extract_seed_extras,
                              stated_chapter_words, strip_chapter_length)
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
# Every question of a round as one form: [{question, why, default}] -> a reply
# per field ('' = take the default), or None when no one is there. None here
# means "ask one at a time" (the terminal); the web GUI installs a form
# (shared/webanswer.py), so a phone shows the round's questions together.
_ask_form = None
NOTE_PROMPT = "Anything else to add or change in the seed?"

# auto mode keeps expanding while the reviewer still asks questions. The cap
# (book.seed_auto_rounds) only exists so a model that asks something every
# round cannot run forever; each round is a full assessment + expansion, so
# five is already generous.
AUTO_MAX_ROUNDS = 5


def auto_rounds():
    """book.seed_auto_rounds: most expansion rounds in auto mode (>= 0)."""
    try:
        value = int(get_config()["book"].get("seed_auto_rounds",
                                             AUTO_MAX_ROUNDS))
    except (TypeError, ValueError):
        value = AUTO_MAX_ROUNDS
    return max(value, 0)


def mode():
    """'ask', 'warn', 'auto' or 'off'. 'ask' becomes 'warn' without a terminal.

    'auto' is never downgraded: it needs no terminal by design.
    """
    value = str(get_config()["book"]["seed_review"]).strip().lower()
    if value in ("false", "no", "none"):
        value = "off"
    if value in ("accept", "automatic", "yes", "true"):
        value = "auto"
    if value not in ("ask", "warn", "auto", "off"):
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


def _ask(question, default="", details="", choices=()):
    """input() that treats Ctrl-D / a closed stdin as 'use the default'.

    details (text the answer depends on, e.g. the changes to keep or undo)
    and choices ([(key, label)] of a menu) are for the browser, which shows
    them beside the question; on a terminal they are already printed."""
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
{prompts.fence(seed_text[:SEED_CHARS], "brief")}
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

def seed_chapter_words(original, seed=""):
    """The per-chapter length the author's seed states, as (low, high), or
    None. The author's own seed wins: an expansion only builds answers in,
    and the first statement there is the author's. A range is used as is;
    its middle is the target (see chapter_words_target)."""
    spans = stated_chapter_words(original) or stated_chapter_words(seed)
    return spans[0] if spans else None


def chapter_words_target(span):
    """The words-per-chapter target for a stated (low, high): the middle,
    to the nearest 50."""
    return int(round((span[0] + span[1]) / 2 / 50) * 50) or span[0]


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
    span = stated.get("words_range")
    if span:
        # the author's own range wins over the model's taste (a run asked for
        # 1,500-1,700 words a chapter was recommended 2,000)
        wpc = min(max(wpc, span[0]), span[1])
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

def _clarification(q, answer):
    """The record for one question: the author's answer, or the suggestion
    they accepted by leaving it empty (None when there is neither)."""
    if answer:
        return {"question": q["question"], "answer": answer, "answered": True}
    if q.get("default"):
        return {"question": q["question"], "answer": q["default"],
                "answered": False}
    return None


def ask_questions(questions):
    """Ask each question; Enter keeps the stated assumption, 'skip' accepts
    the assumptions for the rest, 'back' returns to the previous question.
    Returns the clarifications."""
    if not questions:
        return []
    print(f"\n[SEED REVIEW] {len(questions)} question(s). Type your answer, "
          "or press Enter to accept the suggestion shown; 'back' returns to "
          "the previous question, 'skip' accepts all the rest. Answers and "
          "accepted suggestions are built into the seed.")
    if len(questions) > 1:                 # the whole round, before the first
        for i, q in enumerate(questions, 1):
            print(f"  {i}. {q['question']}")
    replies = [""] * len(questions)
    i = 0
    while i < len(questions):
        q = questions[i]
        print(f"\n  {i + 1}/{len(questions)}. {q['question']}")
        if q.get("why"):
            print(f"     (why it matters: {q['why']})")
        if q.get("default"):
            print(f"     [suggestion, used if you press Enter: "
                  f"{q['default']}]")
        answer = _ask("     > ")
        if answer.lower() == "skip":
            replies[i] = ""
            break
        if answer.lower() == "back":
            i = max(i - 1, 0)
            continue
        replies[i] = answer
        if _INPUT["closed"]:
            break
        i += 1
    return [c for q, r in zip(questions, replies)
            if (c := _clarification(q, r))]


def collect_answers(questions):
    """A round's answers: one per question plus the author's free note, from
    a form when one is installed (the browser), else asked in turn."""
    if _ask_form is not None:
        fields = [{"question": q["question"], "why": q.get("why", ""),
                   "default": q.get("default", "")} for q in questions]
        fields.append({"question": NOTE_PROMPT, "why": "", "default": "",
                       "note": True})
        replies = _ask_form(fields)
        if replies is None:                # no one answered: like Ctrl-D
            _INPUT["closed"] = True
            return []
        replies = [str(r or "").strip() for r in replies]
        replies += [""] * (len(fields) - len(replies))
        answers = [c for q, r in zip(questions, replies)
                   if (c := _clarification(q, r))]
        note = replies[len(questions)]
    else:
        answers = ask_questions(questions)
        note = "" if _INPUT["closed"] else _ask(
            f"\n  {NOTE_PROMPT} (Enter for nothing)\n  > ")
    if note:
        answers.append({"question": "The author adds", "answer": note,
                        "answered": True})
    return answers


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
    extra = [("c", "choose your own"),
             ("s", "stop here: no book is written; edit the seed and rerun "
                   "(an expanded seed is kept)")]
    for key, text in extra:
        print(f"  [{key}] {text}" + ("  (Enter)" if key == default else ""))
    choices = [(key, text) for key, text, _ in options] + extra
    while True:
        choice = _ask("  > ", default, choices=choices).lower()[:1]
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
- Do not add or change a word count or chapter length unless the author's own answer gives one: the book's size is settled separately.
- Write brief notes, not story prose.

Return ONLY the complete expanded brief in Markdown.

THE BRIEF
{prompts.fence(seed_text, "brief")}
"""


def expand(seed_text, answers, rejected=()):
    """The seed rewritten with `answers` built in (one LLM call).
    rejected: why the previous attempt was not used (it is told, and shown
    the lines it must keep word for word)."""
    prompt = build_expansion_prompt(seed_text, answers)
    if rejected:
        prompt += ("\nYOUR PREVIOUS ATTEMPT WAS REJECTED: "
                   + "; ".join(rejected) + ". Keep every character, every "
                   "outline chapter and these constraint lines exactly as "
                   "written:\n" + "\n".join(
                       f"- {c}" for c in extract_seed_extras(seed_text)[0]) + "\n")
    text = clean_llm_text(generate_prose(
        prompt, system=prompts.SEED_EXPANDER, agent="seed_reviewer"))
    inner = prompts.unfence(text, "brief")        # echoed the <brief> fence
    if inner is not None:
        text = inner
    lines = text.splitlines()
    while lines and lines[0].strip() in ('"""', "<brief>", ""):  # stray marks
        lines.pop(0)
    while lines and lines[-1].strip() in ('"""', "</brief>", ""):
        lines.pop()
    return "\n".join(lines).strip() + "\n"


def checked_expansion(seed, answers, rounds):
    """Expand, check it in code, and try once more - told what was lost - if
    the check fails. Returns (expanded, problems): problems is empty when the
    expansion can be used. A rejected attempt is saved, so the author can see
    what the model did (interim/seed_round_NN_rejected.md)."""
    problems = []
    for attempt in (1, 2):
        try:
            expanded = expand(seed, answers, rejected=problems)
            expanded = _strip_length(expanded, "the expansion")
            problems = lost_material(seed, expanded)
        except AbortRun:
            raise
        except Exception as e:
            return None, [f"the expansion failed ({e})"]
        if not problems:
            return expanded, []
        save_interim(f"seed_round_{rounds:02d}_rejected.md", expanded)
        if attempt == 1:
            print("[SEED REVIEW] The expansion lost material ("
                  + "; ".join(problems) + "); asking once more...")
    return None, problems


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


def _span_text(span):
    low, high = span
    return f"{low:,}" if low == high else f"{low:,}-{high:,}"


def _strip_length(text, what):
    """`text` without per-chapter length statements, saying what went. An
    expansion that writes one in (an accepted suggestion once added "2,000
    words per chapter." beside the author's 1,500-1,700) is stripped the same
    way rather than rejected: the size is settled separately."""
    stripped, removed = strip_chapter_length(text)
    if removed:
        print(f"[SEED REVIEW] Took the chapter length out of {what} (the book's "
              "size is set once, as a number): "
              + "; ".join(f'"{r}"' for r in removed))
    return stripped


def show_changes(before, after, rounds):
    """Print the lines an expansion added or changed. Returns every changed
    line, for the keep/undo question (the browser shows them all beside it,
    where the log above would have cut them off)."""
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
    return (f"{len(before.split()):,} -> {len(after.split()):,} words\n"
            + "\n".join(changes))


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
    labels = {
        "a": ("answer the questions and/or add your own notes"
              if assessment["questions"] else "add your own notes")
        + "; the seed is then expanded with them",
        "d": "done: the seed is ready; go on to the book's size",
        "s": "stop here: anything expanded so far is saved; rerun to continue"}
    print("\n[SEED REVIEW] What next?")
    for key, label in labels.items():
        print(f"  [{key}] {label}{marks[key]}")
    while True:
        choice = _ask("  > ", default,
                      choices=list(labels.items())).lower()[:1]
        if choice in ("a", "d", "s"):
            return choice
        print("  Please type a, d or s.")


def _keep_expansion(changes=""):
    labels = {"k": "keep it and review the expanded seed",
              "u": "undo: keep the previous version (your answers are "
                   "passed on)"}
    print(f"\n  [k] {labels['k']}  (Enter)")
    print(f"  [u] {labels['u']}")
    while True:
        choice = _ask("  > ", "k", details=changes,
                      choices=list(labels.items())).lower()[:1]
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
    # Read the author's chapter length, then take it out of the seed: from
    # here on the size is a number (words_per_chapter), not text that every
    # prompt would carry beside it. `original` keeps it, for the resume check.
    span = seed_chapter_words(original, seed_text)
    if span:
        wpc = chapter_words_target(span)
        print(f"[SEED REVIEW] The seed asks for {_span_text(span)} words per "
              f"chapter; aiming for {wpc:,}.")
    seed_text = _strip_length(seed_text, "the seed")
    result = {"seed": seed_text, "chapters": None, "words_per_chapter": wpc,
              "clarifications": list(clarifications), "stop": False}
    how = mode()
    if how == "off":
        if span:
            save_state("plan", {"chapters": None, "words_per_chapter": wpc,
                                "clarifications": result["clarifications"]})
        return result
    max_q = int(book["seed_questions"]) if how in ("ask", "auto") else 0
    print("[SEED REVIEW] Checking that the seed can carry the book's length...")

    def checked(seed, pending):
        # the call is told the size we already know (-c, the outline, a
        # per-chapter length the seed states - parsed in code, see
        # stated_chapter_words); a total length is found by the model itself
        outline = len(extract_seed_outline(seed))
        span = seed_chapter_words(original, seed)
        known = ({"words_per_chapter": chapter_words_target(span)}
                 if span else {})
        hint = requested_size(num_chapters, outline, known, wpc)
        a = assess(seed, hint, max_q, pending)
        if span:
            a["stated"].update(known, words_range=span)
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
        if how == "warn":
            break

        if how == "auto":
            # Accept every suggestion the reviewer offered, build it into the
            # seed, and critique the expanded seed again. Stop when there is
            # nothing left to ask (or at the safety cap).
            if not assessment["questions"]:
                print("[SEED REVIEW] auto: no questions left; the seed stands "
                      "as it is.")
                break
            answers = [{"question": q["question"], "answer": q["default"],
                        "answered": False}
                       for q in assessment["questions"] if q.get("default")]
            if not answers:
                print("[SEED REVIEW] auto: the remaining questions have no "
                      "suggested answer; the seed stands as it is.")
                break
            if rounds >= auto_rounds():
                print(f"[SEED REVIEW] auto: stopping after {rounds} expansion "
                      "round(s); the reviewer is still asking questions "
                      "(answer the remaining questions in the seed by hand and rerun).")
                break
            print(f"[SEED REVIEW] auto: accepting {len(answers)} suggested "
                  "answer(s); expanding the seed...")
            to_build = pending + answers
            rounds += 1
            expanded, problems = checked_expansion(seed, to_build, rounds)
            if problems:
                # It failed twice; a third try would most likely fail too. Keep them
                # as pending clarifications so later prompts still see them.
                print("[SEED REVIEW] auto: the expansion was not used: "
                      + "; ".join(problems) + ". The suggestions are kept "
                      "and passed on.")
                pending = to_build
                break
            save_interim(f"seed_round_{rounds:02d}.md", expanded)
            show_changes(seed, expanded, rounds)
            seed, pending = expanded, []
            save_master(original, seed, pending, rounds)
            print(f"[SEED REVIEW] auto: expanded seed kept and saved as the "
                  f"master seed ({master_path()}); reviewing it again...")
            continue

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
        answers = collect_answers(assessment["questions"])
        if not answers or _INPUT["closed"]:
            print("[SEED REVIEW] Nothing to add; the seed stands as it is.")
            break

        to_build = pending + answers
        rounds += 1
        print(f"\n[SEED REVIEW] Round {rounds}: expanding the seed with your "
              "answers...")
        expanded, problems = checked_expansion(seed, to_build, rounds)
        if problems:
            print("[SEED REVIEW] The expansion was not used: "
                  + "; ".join(problems) + ". Your answers are kept and "
                  "passed on.")
            pending = to_build
            continue
        save_interim(f"seed_round_{rounds:02d}.md", expanded)
        changes = show_changes(seed, expanded, rounds)
        if _keep_expansion(changes):
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
            if how == "auto" and recommended:
                # The whole point of the review is to not write a padded book;
                # a human accepting the on-screen suggestion would take this.
                chosen = recommended
                print(f"[SEED REVIEW] auto: the requested size does not fit "
                      f"the seed; using the recommendation "
                      f"{_size_line(recommended)}.")
            else:
                print("[SEED REVIEW] Warning: the requested size does not fit "
                      "the seed (see above). Continuing with it; rerun on a "
                      "terminal with book.seed_review: ask to choose another.")
    if chosen:
        result["chapters"] = chosen["chapters"]
        result["words_per_chapter"] = chosen["words_per_chapter"]
        print(f"[SEED REVIEW] Book size: {_size_line(chosen)}.")
    save_state("plan", {
        "chapters": result["chapters"],
        "words_per_chapter": result["words_per_chapter"],
        "clarifications": result["clarifications"]})
    return result
