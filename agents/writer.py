"""
Writer Agent
Drafts each chapter with continuity: the writer sees the story bible, the
chapter's lore brief, established story FACTS (who is alive, who has met
whom, relationships, timeline), and a rolling summary of earlier chapters.
After each chapter one call produces both the summary and a structured
story state (chronology) used by every later continuity check.
"""

import re

from shared import prompts
from shared.context import context, update_context
from shared.llm_utils import clean_llm_text, extract_json
from shared.llm_client import (EndpointUnavailable, generate_prose,
                               generate_with_wait, get_config)
from shared.output import chapter_filename, save_chapter, save_interim, \
    save_interim_json
from shared.story_state import (merge_states, minimal_state, normalize_state,
                                render_story_facts, render_story_so_far)

# A reply that opens like this and is shorter than the chapter minimum is a
# refusal, not a chapter ("I can't write that..."). Anything longer, or a
# first-person story that merely starts "I can't believe...", passes.
_REFUSAL_RE = re.compile(
    r"^\W*(?:i\s+(?:can[\u2019']?t|cannot|won[\u2019']?t|apologi[sz]e"
    r"|am\s+(?:unable|not\s+able|sorry))"
    r"|i[\u2019']m\s+(?:sorry|unable|not\s+able)|sorry\b|as\s+an\s+ai\b"
    r"|unfortunately,?\s+i\b)", re.IGNORECASE)

_RETRY_NOTE = (
    "\n\nNOTE: your previous reply was not a usable chapter (too short, "
    "empty, or a refusal). Write the complete chapter now, exactly as the "
    "author's brief asks."
)
PREVIOUS_ENDING_WORDS = 180


def _bad_draft(draft, min_words):
    """Why `draft` can't be a chapter, or None when it can."""
    count = len(draft.split())
    if count == 0:
        return "the model returned nothing"
    if count < min_words and _REFUSAL_RE.match(draft):
        return f"the reply looks like a refusal ({count} words)"
    if count < max(50, int(min_words * 0.4)):
        return f"the draft is only {count} words"
    return None


def _tail_words(text, count=PREVIOUS_ENDING_WORDS):
    """The last `count` words, starting at a paragraph or sentence break
    when one is close, so the excerpt doesn't begin mid-sentence."""
    words = text.split()
    if len(words) <= count:
        return text.strip()
    tail = " ".join(words[-count:])
    m = re.search(r"(?<=[.!?\"\u201d])\s+(?=[A-Z\"\u201c])", tail[:300])
    return tail[m.end():] if m else tail


def _draft_chapter(n, prompt, system, min_words):
    """Draft one chapter; one retry if the reply is unusable (empty, a
    refusal, far too short). Returns the draft, or None if both attempts
    failed. Raises on transport errors, like generate_prose()."""
    for attempt in (1, 2):
        text = prompt if attempt == 1 else prompt + _RETRY_NOTE
        draft = clean_llm_text(
            generate_prose(text, system=system, agent="writer"))
        problem = _bad_draft(draft, min_words)
        if not problem:
            return draft
        print(f"[WRITER] Chapter {n}: {problem}"
              + ("; retrying once." if attempt == 1 else "."))
    return None


def _summarize_and_extract(number, title, draft, fallback=True):
    """One call: prose summary (for later chapters' prompts) + structured
    story state (timeline, present characters, meetings, deaths, ...).

    On failure returns a facts-free fallback state, or None when
    fallback=False (used to refresh existing state without ever replacing
    good facts with empty ones)."""
    canon, aliases = _canonical_names(), _alias_map()
    listing = ", ".join(
        n + (f" (also called {', '.join(aliases[n])})" if n in aliases else "")
        for n in canon)
    names_line = (
        "Known characters - when the text refers to one of these people (by "
        "first name, nickname or surname), use EXACTLY the first spelling "
        "given for them: " + listing + ". Anyone else: the name as written.\n"
        if canon else "")
    prompt = f"""Analyse this chapter and return ONLY JSON (no fences):
{{
  "summary": "3-4 factual sentences: key events, new characters or settings, how the chapter ends",
  "time": "when it takes place (e.g. 'Thursday afternoon, week one')",
  "location": "primary location",
  "present": ["every named character on page"],
  "events": [
    {{"type": "first_meeting", "who": ["A", "B"]}},
    {{"type": "relationship_change", "who": ["A", "B"], "detail": "e.g. became lovers / had a row / first flirtation"}},
    {{"type": "death", "who": ["X"], "detail": "how"}},
    {{"type": "injury", "who": ["X"], "detail": "what injury"}},
    {{"type": "secret_revealed", "who": ["X"], "detail": "what and to whom"}}
  ]
}}
Only include event types that actually happened; use [] if none. Use character names exactly as written.
{names_line}
Chapter {number}: {title}

\"\"\"{draft}\"\"\""""
    last_error = None
    for attempt in (1, 2):  # one retry: a bad JSON reply loses continuity facts
        try:
            raw = generate_with_wait(prompt, system=prompts.EXTRACTOR,
                                     agent="extractor", json_mode=True)
            data = extract_json(raw, expect="object")
            summary = (str(data.get("summary", "")).strip()
                       or _truncate(draft))
            state = normalize_state(number, title, data, canon, aliases)
            state["summary"] = summary
            return summary, state
        except EndpointUnavailable:
            raise  # the draft is saved; abort and let a rerun resume
        except Exception as e:
            last_error = e
            if attempt == 1:
                print(f"[WRITER] State extraction failed for chapter "
                      f"{number} ({e}); retrying once.")
    if not fallback:
        print(f"[WRITER] Could not refresh the story state for chapter "
              f"{number} ({last_error}); keeping the earlier state.")
        return None
    print(f"[WRITER] Warning: state extraction failed for chapter "
          f"{number}: {last_error}. Continuity facts for this chapter "
          "are missing.")
    summary = _truncate(draft)
    return summary, minimal_state(number, title, summary)


def _canonical_names():
    """Character names from the story bible, bracketed qualifiers removed."""
    names = []
    for c in (context.get("bible") or {}).get("characters") or []:
        name = re.sub(r"\(.*?\)", " ", str(c.get("name", ""))).strip()
        if name and name not in names:
            names.append(name)
    return names


def _alias_map():
    """{name: [nicknames]} declared in the bible (for characters with any)."""
    out = {}
    for c in (context.get("bible") or {}).get("characters") or []:
        name = re.sub(r"\(.*?\)", " ", str(c.get("name", ""))).strip()
        if name and c.get("aliases"):
            out[name] = [str(a) for a in c["aliases"]]
    return out


def _truncate(text, limit=400):
    """Cut at a word boundary, not mid-word."""
    text = " ".join(text.split())
    if len(text) <= limit:
        return text
    return text[:limit].rsplit(" ", 1)[0] + "..."


def _summaries_markdown(chapters, summaries):
    """Rolling chapter summaries, rewritten as each chapter finishes."""
    lines = ["# Chapter Summaries (rolling)", ""]
    for ch in chapters:
        s = summaries.get(ch["number"])
        if s:
            lines += [f"## Chapter {ch['number']}: {ch['title']}", "", s, ""]
    return "\n".join(lines)


def _states_markdown(chronology):
    """Rolling story-state log, rewritten as each chapter finishes."""
    lines = ["# Story State (rolling chronology)", ""]
    for n in sorted(chronology):
        s = chronology[n]
        lines += [f"## Chapter {n}: {s['title']}", ""]
        lines += [f"*time: {s['time'] or '?'} | location: "
                  f"{s['location'] or '?'} | present: "
                  f"{', '.join(s['present']) or '?'}*", ""]
        for ev in s["events"]:
            who = " & ".join(ev["who"])
            detail = f" - {ev['detail']}" if ev.get("detail") else ""
            lines.append(f"- **{ev['type']}**: {who}{detail}")
        lines.append("")
    return "\n".join(lines)


def _save_state(chapters, drafts, summaries, chronology):
    """Persist the rolling summaries and story state (context + interim)."""
    update_context("summaries", summaries)
    update_context("chronology", chronology)
    update_context("drafts", drafts)
    save_interim("summaries.md", _summaries_markdown(chapters, summaries))
    save_interim("story_state.md", _states_markdown(chronology))
    save_interim_json("summaries.json", summaries)
    save_interim_json("chronology.json", chronology)


def _final_body(number):
    """Body of chapter `number` as edited, or None before it is edited."""
    prefix = f"## Chapter {number}:"
    for entry in context.get("final") or []:
        if entry.startswith(prefix):
            return entry.split("\n\n", 1)[1] if "\n\n" in entry else ""
    return None


def refresh_state(number):
    """Re-extract chapter `number`'s summary and story facts from its FINAL
    (edited) text, so later chapters and reviews build on what the book now
    says rather than on the unrevised draft. Keeps the old state if the
    extraction fails."""
    chapters = context.get("chapters", [])
    chapter = next((c for c in chapters if c["number"] == number), None)
    drafts = context.get("drafts", {})
    text = _final_body(number) or drafts.get(number)
    if not chapter or not text:
        return
    result = _summarize_and_extract(number, chapter["title"], text,
                                    fallback=False)
    if result is None:
        return
    summaries = context.get("summaries", {})
    chronology = context.get("chronology", {})
    summaries[number], chronology[number] = result
    _save_state(chapters, drafts, summaries, chronology)
    print(f"[WRITER] Story state for chapter {number} refreshed from the "
          "edited text.")


def run_writer(only=None):
    """
    Write drafts for every chapter, in order, feeding each chapter the
    established story facts, summaries, lore brief, and bible.

    only: write just this chapter number (used by the review-as-you-go
    loop in main.py); None writes every chapter that has no draft yet.
    """
    if only is None:
        print("[WRITER] Starting writing phase...")
    cfg = get_config()
    words = int(cfg["book"].get("words_per_chapter", 800))
    tolerance = float(cfg["book"].get("word_count_tolerance", 0.8))
    min_words, max_words = int(words * tolerance), int(words * 1.2)

    chapters = context.get("chapters", [])
    research = context.get("research", {})
    bible = context.get("bible") or {}
    title = context.get("title", "")

    if not chapters:
        print("[WRITER] No chapters found in context. Skipping writing.")
        return

    window = int(cfg["book"].get("summary_window", 8))
    # The bible rides in the system message: identical for every chapter, so
    # servers with prompt caching reuse it instead of re-reading it each time.
    system = prompts.with_bible(prompts.WRITER, bible)

    drafts, summaries, chronology = (context.get("drafts", {}),
                                    context.get("summaries", {}),
                                    context.get("chronology", {}))

    def save_state():
        _save_state(chapters, drafts, summaries, chronology)

    for chapter in chapters:
        n, ch_title = chapter["number"], chapter["title"]
        if only is not None and n != only:
            continue
        if drafts.get(n):
            if chronology.get(n):
                print(f"[WRITER] Chapter {n}: already drafted; skipping.")
            else:
                # draft survived but its state didn't (crash, corrupt
                # snapshot): rebuild the state, never redraft the chapter
                print(f"[WRITER] Chapter {n}: draft found without story "
                      "state; rebuilding the state from the draft.")
                summaries[n], chronology[n] = _summarize_and_extract(
                    n, ch_title, drafts[n])
                save_state()
            continue

        print(f"[WRITER] Writing chapter {n}/{len(chapters)}: {ch_title}")

        lore = research.get(n, "") or "(no lore brief available)"
        story_so_far = (render_story_so_far(summaries, before=n,
                                            window=window)
                        or "(This is the first chapter.)")
        story_facts = render_story_facts(merge_states(chronology, upto=n))

        earlier = [k for k in drafts if k < n]
        prev_block = ""
        if earlier:
            prev = max(earlier)
            prev_text = _final_body(prev) or drafts[prev]  # edited if any
            prev_block = (
                "\nPREVIOUS CHAPTER ENDING (for voice and a smooth "
                "transition; do not repeat or paraphrase it):\n"
                + _tail_words(prev_text) + "\n")

        prompt = f"""Write Chapter {n} of "{title}".

STORY FACTS (established continuity - do not contradict):
{story_facts}

STORY SO FAR (previous chapters):
{story_so_far}
{prev_block}
LORE BRIEF FOR THIS CHAPTER:
{lore}

CHAPTER {n}: {ch_title}
Chapter summary: {chapter.get('summary', '(none provided)')}

Requirements:
- Write {min_words}-{max_words} words of continuous narrative prose. Length
  is enforced after drafting: chapters below {min_words} words are sent back
  for substantive expansion (not padding), so write fully from the start.
- Keep character names, relationships, and world facts EXACTLY consistent
  with the story bible and the STORY FACTS above.
- Do NOT include a chapter heading, the chapter title, or any meta commentary.
- Begin directly with the narrative."""

        try:
            draft = _draft_chapter(n, prompt, system, min_words)
        except EndpointUnavailable:
            print(f"[WRITER] Endpoint never came back; aborting the writing "
                  f"phase at chapter {n}. Rerun the same command to resume "
                  "from the last saved chapter.")
            raise
        except Exception as e:
            # Later chapters build on this one's summary and facts; drafting
            # on past a gap would break continuity. Stop and let a rerun
            # resume from here.
            print(f"[WRITER] Error writing chapter {n}: {e}. Stopping the "
                  "writing phase; rerun to resume from this chapter.")
            break

        if draft is None:
            print(f"[WRITER] The model produced no usable chapter {n} after "
                  "two attempts. Stopping the writing phase; rerun to "
                  "resume from this chapter.")
            break

        word_count = len(draft.split())
        drafts[n] = draft
        save_interim(
            chapter_filename("draft", n),
            f"## Chapter {n}: {ch_title}\n\n{draft}",
        )
        save_chapter(n, ch_title, draft)
        print(f"[WRITER] Draft completed for chapter {n} ({word_count} words).")

        summaries[n], chronology[n] = _summarize_and_extract(n, ch_title,
                                                             draft)
        save_state()

    update_context("drafts", drafts)
    update_context("summaries", summaries)
    if only is None:
        print(f"[WRITER] Writing phase complete "
              f"({len(drafts)}/{len(chapters)} drafts).")
