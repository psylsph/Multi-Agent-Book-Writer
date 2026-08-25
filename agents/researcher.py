"""
Researcher Agent (Lore Keeper)
Builds a per-chapter writing brief from the story bible: which characters
are on page, setting details, plot beats to hit, and continuity notes.
Fiction-oriented replacement for the old "statistics and quotes" researcher.
"""

from shared.context import context, update_context
from shared.llm_utils import render_bible
from shared.llm_client import generate
from shared.output import chapter_filename, save_interim


def run_researcher():
    """
    Create a lore brief for every planned chapter.
    Briefs are keyed by chapter number (not title) so lookups can't drift.
    """
    print("[RESEARCHER] Building chapter lore briefs...")
    chapters = context.get("chapters", [])
    bible = context.get("bible") or {}

    if not chapters:
        print("[RESEARCHER] No chapters found in context. Skipping research.")
        return

    research_data = {}
    for chapter in chapters:
        n, title = chapter["number"], chapter["title"]
        print(f"[RESEARCHER] Briefing chapter {n}/{len(chapters)}: {title}")

        prompt = f"""You are a story lore keeper preparing a writing brief for ONE chapter of a novel.

STORY BIBLE
{render_bible(bible)}

CHAPTER TO BRIEF
Chapter {n}: {title}
{chapter.get('summary', '')}

Produce a concise brief (150-250 words) with exactly these sections:
- On-page characters: which bible characters appear and what each wants in this chapter
- Setting details: specific sensory details drawn from the world description
- Plot beats: 3-6 beats this chapter must hit to serve the outline
- Continuity: what must stay consistent with earlier chapters, and what to set up (or pay off) for later

Notes only - do not write prose."""

        try:
            brief = generate(prompt, agent="researcher")
            research_data[n] = brief
            save_interim(
                chapter_filename("lore", n),
                f"# Lore Brief - Chapter {n}: {title}\n\n{brief}",
            )
            print(f"[RESEARCHER] Brief completed for chapter {n}.")
        except Exception as e:
            print(f"[RESEARCHER] Error briefing chapter {n}: {e}")
            research_data[n] = ""

    update_context("research", research_data)
    done = sum(1 for v in research_data.values() if v)
    print(f"[RESEARCHER] Research phase complete ({done}/{len(chapters)} briefs).")
