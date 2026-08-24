"""
Writer Agent
Drafts each chapter with continuity: the writer sees the story bible, the
chapter's lore brief, and a rolling summary of everything written so far.
After each chapter a short plot summary is generated for later chapters.
"""

from shared.context import context, update_context
from shared.llm_utils import clean_llm_text, render_bible
from shared.ollama_client import generate, get_config
from shared.output import chapter_filename, save_interim

SYSTEM_PROMPT = (
    "You are a published novelist. You write vivid, coherent prose, follow "
    "the outline faithfully, and keep characters and world facts consistent. "
    "You never break the fourth wall or add meta commentary."
)


def _summarize_chapter(number, title, draft):
    """Condense a finished chapter into a few factual sentences."""
    prompt = f"""Summarize the following chapter in 3-4 factual sentences.
Cover the key events, any new characters or settings introduced, and how the
chapter ends. Write only the summary text.

Chapter {number}: {title}

\"\"\"{draft}\"\"\"
"""
    try:
        return generate(prompt, agent="writer")
    except Exception as e:
        print(f"[WRITER] Warning: could not summarize chapter {number}: {e}")
        return ""


def _summaries_markdown(chapters, summaries):
    """Rolling chapter summaries, rewritten as each chapter finishes."""
    lines = ["# Chapter Summaries (rolling)", ""]
    for ch in chapters:
        s = summaries.get(ch["number"])
        if s:
            lines += [f"## Chapter {ch['number']}: {ch['title']}", "", s, ""]
    return "\n".join(lines)


def run_writer():
    """
    Write drafts for every chapter, in order, feeding each chapter a
    summary of the story so far.
    """
    print("[WRITER] Starting writing phase...")
    cfg = get_config()
    words = int(cfg["book"].get("words_per_chapter", 800))
    min_words, max_words = int(words * 0.7), int(words * 1.3)

    chapters = context.get("chapters", [])
    research = context.get("research", {})
    bible = context.get("bible") or {}
    title = context.get("title", "")

    if not chapters:
        print("[WRITER] No chapters found in context. Skipping writing.")
        return

    drafts, summaries = {}, {}

    for chapter in chapters:
        n, ch_title = chapter["number"], chapter["title"]
        print(f"[WRITER] Writing chapter {n}/{len(chapters)}: {ch_title}")

        lore = research.get(n, "") or "(no lore brief available)"
        story_so_far = "\n\n".join(
            f"Chapter {k}: {summaries[k]}"
            for k in sorted(summaries) if summaries.get(k)
        ) or "(This is the first chapter.)"

        prompt = f"""Write Chapter {n} of "{title}".

{render_bible(bible)}

CHAPTER {n}: {ch_title}
Chapter summary: {chapter.get('summary', '(none provided)')}

LORE BRIEF FOR THIS CHAPTER:
{lore}

STORY SO FAR (previous chapters):
{story_so_far}

Requirements:
- Write {min_words}-{max_words} words of continuous narrative prose.
- Show, don't tell. Use dialogue where it brings characters to life.
- Keep character names and world facts EXACTLY consistent with the bible.
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

        summaries[n] = _summarize_chapter(n, ch_title, draft)
        save_interim("summaries.md", _summaries_markdown(chapters, summaries))

    update_context("drafts", drafts)
    update_context("summaries", summaries)
    print(f"[WRITER] Writing phase complete ({len(drafts)}/{len(chapters)} drafts).")
