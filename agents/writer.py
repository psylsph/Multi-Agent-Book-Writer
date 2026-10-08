"""
Writer Agent
Drafts each chapter with continuity: the writer sees the story bible, the
chapter's lore brief, established story FACTS (who is alive, who has met
whom, relationships, timeline), and a rolling summary of earlier chapters.
After each chapter one call produces both the summary and a structured
story state (chronology) used by every later continuity check.
"""

import re

from shared import extraction_checks, prompts
from shared.context import context, update_context
from shared.llm_utils import clean_llm_text, extract_json
from shared.llm_client import (AbortRun, EndpointUnavailable, generate_prose,
                               generate_with_wait, get_config)
from shared.output import chapter_filename, render_chapter, save_chapter, \
    save_interim
from shared.resume import save_state
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
  "present": ["characters who are physically in a scene of this chapter"],
  "events": [
    {{"type": "first_meeting", "who": ["A", "B"]}},
    {{"type": "relationship_change", "who": ["A", "B"], "detail": "e.g. became lovers / had a row / first flirtation"}},
    {{"type": "death", "who": ["X"], "detail": "how"}},
    {{"type": "injury", "who": ["X"], "detail": "what injury"}},
    {{"type": "secret_revealed", "who": ["X"], "detail": "what and to whom"}}
  ]
}}
Rules:
- "present": only characters who are in a scene of this chapter, speaking or acting. NOT people who are merely mentioned, remembered, expected, or dead.
- first_meeting: two characters meeting for the FIRST time in this chapter. If the text shows they already know each other, it is not a first meeting.
- relationship_change: a clear shift shown in this chapter (new intimacy, a break, a betrayal). Not ordinary conversation, teamwork or friendliness.
- death: a character dies in this chapter or is reported dead in it. injury: someone is physically hurt.
- secret_revealed: someone learns something that was hidden from them; "who" is the person who learns it.
- Only include events that actually happened in this chapter; use [] if none. When unsure, leave the event out: a missing event is better than an invented one.
- Use character names exactly as written.
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
            state = _check_state(number, draft, state, canon, aliases)
            return summary, state
        except AbortRun:
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


def _ask_names(prompt, key, candidates):
    """Send a focused question whose JSON answer is {key: [names]}; return the
    set of candidates named, or None when it could not be asked/read. A name
    like "Marcus" is accepted for the candidate "Marcus Reed"."""
    try:
        raw = generate_with_wait(prompt, system=prompts.VERIFIER,
                                 agent="extractor", json_mode=True)
        answer = extract_json(raw, expect="object").get(key)
    except AbortRun:
        raise
    except Exception:
        return None
    if not isinstance(answer, list):
        return None
    named = set()
    for item in answer:
        said = str(item).strip().lower()
        matches = [c for c in candidates if said == c.lower()
                   or said in extraction_checks.name_variants(c)]
        if len(matches) == 1:
            named.add(matches[0])
    return named


def _confirm_deaths(candidates, evidence):
    """Which of `candidates` die in this chapter? (a subset, or None)"""
    return _ask_names(f"""Text:
\"\"\"{evidence}\"\"\"

Candidates: {', '.join(candidates)}

Question: which of the candidates die in this chapter, or are reported dead as news of something that has just happened? A death long ago (backstory), a near-death, a threat, a nightmare or a figure of speech does not count. Judge only from the text; use [] if none.
Reply with ONLY JSON: {{"dead": ["name", ...]}}""", "dead", candidates)


def _confirm_alive(candidates, evidence):
    """Which of `candidates` are still alive at the end of the passage? A
    differently phrased question than _confirm_deaths: a model that wrongly
    says someone died will usually also list them here as alive, and the
    disagreement exposes the mistake."""
    return _ask_names(f"""Text:
\"\"\"{evidence}\"\"\"

Candidates: {', '.join(candidates)}

Question: which of the candidates are still alive at the end of this text, as far as the text shows? Someone who has died, or is reported dead, is not alive. Judge only from the text; use [] if none.
Reply with ONLY JSON: {{"alive": ["name", ...]}}""", "alive", candidates)


def _check_state(number, text, state, canon, aliases):
    """Cross-check the extractor's state against the chapter text.

    (book.extraction_checks, on by default.) Three things, all aimed at the
    errors that corrupt continuity:
      1. names the text never mentions are dropped (invented characters);
      2. a character who died in an EARLIER chapter is not killed again (that
         would move their death);
      3. deaths are verified in both directions with ONE focused question per
         chapter: the candidates are every death the model claimed plus every
         bible character whose name sits next to death language, and the
         model says which of them really die. A claimed death that is not
         confirmed is dropped; a hinted character who is confirmed gets the
         death event the extractor missed. An unanswerable question leaves
         the model's own decision alone.
    A chapter with no death language and no claimed death costs no extra call.
    """
    if not get_config()["book"]["extraction_checks"]:
        return state
    variants = {n: extraction_checks.name_variants(n, aliases.get(n, ()))
                for n in canon}

    state, dropped = extraction_checks.ground_state(state, text, variants)
    if dropped:
        print(f"[WRITER] Chapter {number}: ignored name(s) that never appear "
              f"in the text: {', '.join(dropped)}")

    chronology = context.get("chronology") or {}
    prior_dead = {n.lower() for n in
                  merge_states(chronology, upto=number)["dead"]}
    # someone who died in an earlier chapter is not "in a scene" (a funeral,
    # a memory): the extractor lists them anyway
    state = {**state, "present": [n for n in state["present"]
                                  if n.lower() not in prior_dead]}
    events = []
    for event in state["events"]:
        if event["type"] == "death":
            who = [n for n in event["who"] if n.lower() not in prior_dead]
            if not who:
                continue                      # already dead: not a new death
            event = {**event, "who": who}
        events.append(event)
    state = {**state, "events": events}

    hints = extraction_checks.death_hints(text, variants)
    claimed = list(dict.fromkeys(
        n for e in state["events"] if e["type"] == "death" for n in e["who"]))
    candidates = list(dict.fromkeys(
        claimed + [n for n in hints if n.lower() not in prior_dead]))
    if not candidates:
        return state

    # snippets around the death language; the whole chapter if a claimed
    # death has no death language near it (the model may know better)
    if hints and all(n in hints for n in claimed):
        evidence = "\n...\n".join(dict.fromkeys(
            s for n in candidates for s in hints.get(n, [])))
    else:
        evidence = text
    dead = _confirm_deaths(candidates, evidence)
    if dead is None:
        return state
    # A death the extractor did NOT report is added only when a second,
    # differently phrased question agrees (the one-question version invented
    # deaths for living characters standing near death language).
    proposed = [n for n in candidates if n in dead and n not in claimed]
    alive = _confirm_alive(proposed, evidence) if proposed else set()
    adding = [n for n in proposed if alive is not None and n not in alive]
    for name in proposed:
        if name not in adding:
            print(f"[WRITER] Chapter {number}: possible death of {name} "
                  "not recorded: the two checks disagreed or could not be "
                  "read.")
    for name in claimed:
        if name not in dead:
            print(f"[WRITER] Chapter {number}: the second check does not "
                  f"confirm that {name} dies; ignoring that death.")
    state["events"] = [
        {**e, "who": [n for n in e["who"] if n in dead]}
        if e["type"] == "death" else e for e in state["events"]]
    state["events"] = [e for e in state["events"] if e["who"]]
    for name in adding:
        print(f"[WRITER] Chapter {number}: the second check found a death "
              f"the extractor missed: {name}.")
        state["events"].append({
            "type": "death", "who": [name],
            "detail": "confirmed by a second check of the text"})
    return state


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
    save_state("drafts", drafts)
    save_state("summaries", summaries)
    save_state("chronology", chronology)
    save_interim("summaries.md", _summaries_markdown(chapters, summaries))
    save_interim("story_state.md", _states_markdown(chronology))


def _final_body(number):
    """Body of chapter `number` as edited, or None before it is edited."""
    return (context.get("final") or {}).get(number)


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
    words = int(cfg["book"]["words_per_chapter"])
    tolerance = float(cfg["book"]["word_count_tolerance"])
    min_words, max_words = int(words * tolerance), int(words * 1.2)

    chapters = context.get("chapters", [])
    research = context.get("research", {})
    bible = context.get("bible") or {}
    title = context.get("title", "")

    if not chapters:
        print("[WRITER] No chapters found in context. Skipping writing.")
        return

    window = int(cfg["book"]["summary_window"])
    # The bible rides in the system message: identical for every chapter, so
    # servers with prompt caching reuse it instead of re-reading it each time.
    system = prompts.with_bible(prompts.WRITER, bible)

    drafts, summaries, chronology = (context.get("drafts", {}),
                                    context.get("summaries", {}),
                                    context.get("chronology", {}))

    def persist():
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
                persist()
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
        except AbortRun:
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
        save_state("drafts", drafts)  # before extraction: never lose a draft
        save_interim(chapter_filename("draft", n),
                     render_chapter(n, ch_title, draft))
        save_chapter(n, ch_title, draft)
        print(f"[WRITER] Draft completed for chapter {n} ({word_count} words).")

        summaries[n], chronology[n] = _summarize_and_extract(n, ch_title,
                                                             draft)
        persist()

    update_context("drafts", drafts)
    update_context("summaries", summaries)
    if only is None:
        print(f"[WRITER] Writing phase complete "
              f"({len(drafts)}/{len(chapters)} drafts).")
