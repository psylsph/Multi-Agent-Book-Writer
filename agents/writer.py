"""
Writer Agent
Drafts each chapter with continuity: the writer sees the story bible, the
chapter's lore brief, established story FACTS (who is alive, who has met
whom, relationships, timeline), and a rolling summary of earlier chapters.
After each chapter one call produces both the summary and a structured
story state (chronology) used by every later continuity check.
"""

from shared.context import context, update_context
from shared.llm_utils import clean_llm_text, extract_json, render_bible
from shared.llm_client import generate, get_config
from shared.output import chapter_filename, save_interim
from shared.story_state import (merge_states, minimal_state, normalize_state,
                                render_story_facts)

SYSTEM_PROMPT = (
    "You are a published novelist. You write vivid, coherent prose, follow "
    "the outline faithfully, and keep characters, relationships, and world "
    "facts consistent with what came before. You never break the fourth "
    "wall or add meta commentary."
)


def _summarize_and_extract(number, title, draft):
    """One call: prose summary (for later chapters' prompts) + structured
    story state (timeline, present characters, meetings, deaths, ...)."""
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

Chapter {number}: {title}

\"\"\"{draft}\"\"\""""
    try:
        raw = generate(prompt, agent="writer")
        data = extract_json(raw, expect="object")
        summary = str(data.get("summary", "")).strip() or draft[:400]
        return summary, normalize_state(number, title, data)
    except Exception as e:
        print(f"[WRITER] Warning: state extraction failed for chapter "
              f"{number}: {e}")
        return draft[:400] + "...", minimal_state(number, title, "")


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


def run_writer():
    """
    Write drafts for every chapter, in order, feeding each chapter the
    established story facts, summaries, lore brief, and bible.
    """
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

    drafts, summaries, chronology = {}, {}, {}

    for chapter in chapters:
        n, ch_title = chapter["number"], chapter["title"]
        print(f"[WRITER] Writing chapter {n}/{len(chapters)}: {ch_title}")

        lore = research.get(n, "") or "(no lore brief available)"
        story_so_far = "\n\n".join(
            f"Chapter {k}: {summaries[k]}"
            for k in sorted(summaries) if summaries.get(k)
        ) or "(This is the first chapter.)"
        story_facts = render_story_facts(merge_states(chronology, upto=n))

        prompt = f"""Write Chapter {n} of "{title}".

{render_bible(bible)}

CHAPTER {n}: {ch_title}
Chapter summary: {chapter.get('summary', '(none provided)')}

LORE BRIEF FOR THIS CHAPTER:
{lore}

STORY FACTS (established continuity - do not contradict):
{story_facts}

STORY SO FAR (previous chapters):
{story_so_far}

Requirements:
- Write {min_words}-{max_words} words of continuous narrative prose. Length
  is enforced after drafting: chapters below {min_words} words are sent back
  for substantive expansion (not padding), so write fully from the start.
- Show, don't tell. Use dialogue where it brings characters to life.
- Keep character names, relationships, and world facts EXACTLY consistent
  with the bible and the STORY FACTS above.
- Do NOT include a chapter heading, the chapter title, or any meta commentary.
- Begin directly with the narrative."""

        try:
            draft = clean_llm_text(generate(prompt, system=SYSTEM_PROMPT,
                                            agent="writer"))
        except Exception as e:
            print(f"[WRITER] Error writing chapter {n}: {e}")
            continue

        word_count = len(draft.split())
        if word_count < 100:
            print(f"[WRITER] Warning: chapter {n} draft is only {word_count} "
                  "words; the model may have misbehaved.")
        drafts[n] = draft
        save_interim(
            chapter_filename("draft", n),
            f"## Chapter {n}: {ch_title}\n\n{draft}",
        )
        print(f"[WRITER] Draft completed for chapter {n} ({word_count} words).")

        summaries[n], chronology[n] = _summarize_and_extract(n, ch_title,
                                                             draft)
        update_context("chronology", chronology)
        save_interim("summaries.md", _summaries_markdown(chapters, summaries))
        save_interim("story_state.md", _states_markdown(chronology))

    update_context("drafts", drafts)
    update_context("summaries", summaries)
    print(f"[WRITER] Writing phase complete ({len(drafts)}/{len(chapters)} drafts).")
