"""
Reviewer Agent
Continuity review of each chapter draft against the accumulated story
state (who is alive, who has met whom, relationship progression, timeline)
plus the story bible. Returns a structured verdict + issues; the editor
agent feeds the findings into bounded revision rounds.
"""

from shared.context import context
from shared.llm_utils import extract_json, render_bible
from shared.ollama_client import generate
from shared.output import chapter_filename, save_interim
from shared.story_state import merge_states, render_story_facts

SYSTEM_PROMPT = (
    "You are a meticulous continuity editor for fiction. You catch timeline "
    "and relationship contradictions that ruin immersion. You judge only "
    "from the facts given and report JSON."
)


def run_reviewer(number, title, draft):
    """
    Review one chapter draft for continuity with everything before it.

    Returns (verdict, issues) where verdict is "pass" or "revise" and
    issues is a list of {type, description, fix} dicts.
    """
    bible = context.get("bible") or {}
    chronology = context.get("chronology") or {}
    prior = merge_states(chronology, upto=number)  # strictly earlier chapters

    prompt = f"""Review this chapter draft for CONTINUITY errors against the story so far.

STORY BIBLE
{render_bible(bible)}

STORY FACTS BEFORE THIS CHAPTER
{render_story_facts(prior)}

Chapter summaries so far:
{(chr(10).join(f'Ch{n}: ' + s.get('summary', '') for n, s in sorted(chronology.items()) if n < number)) or '(first chapter)'}

CHAPTER {number} DRAFT TO REVIEW ("{title}"):
\"\"\"{draft}\"\"\"

Check for these specific continuity failures:
1. DEAD RESURRECTION: a character who died earlier acting, speaking, or being present as alive (memories/dreams/dialogue ABOUT them are fine).
2. RELATIONSHIP REGRESSION: characters who have already met (see facts) acting like strangers, or intimate characters suddenly formal/distant without cause - e.g. reintroducing themselves, "pleased to meet you", surprise at knowing each other, or lovers who forget their history.
3. RELATIONSHIP LEAP: characters closer/more intimate than their recorded state allows (they cannot be lovers if they only met last chapter with no escalation shown).
4. KNOWLEDGE ERRORS: a character knowing a secret that was never revealed to them, or forgetting something they personally witnessed.
5. TIMELINE ERRORS: impossible ordering, time of day/date contradicting the timeline, injuries healed instantly, day/night mismatches.
6. SETTING ERRORS: characters in places they cannot have reached, or locations contradicting earlier chapters.

Return ONLY JSON (no fences):
{{"verdict": "pass" or "revise",
  "issues": [{{"type": "dead_resurrection|relationship_regression|relationship_leap|knowledge|timeline|setting", "description": "what exactly contradicts what, with quotes", "fix": "the minimal change that fixes it"}}]}}

Only report real contradictions with the facts above - not style opinions. Empty issues + "pass" if the chapter is consistent."""

    try:
        raw = generate(prompt, system=SYSTEM_PROMPT, agent="reviewer")
        data = extract_json(raw, expect="object")
        verdict = str(data.get("verdict", "pass")).strip().lower()
        if verdict not in ("pass", "revise"):
            verdict = "revise" if data.get("issues") else "pass"
        issues = []
        for issue in data.get("issues") or []:
            if isinstance(issue, dict) and issue.get("description"):
                issues.append({
                    "type": str(issue.get("type", "continuity")).strip(),
                    "description": str(issue["description"]).strip(),
                    "fix": str(issue.get("fix", "")).strip(),
                })
        if issues:
            verdict = "revise"
        if verdict == "pass" and not issues:
            save_interim(chapter_filename("review", number),
                         f"# Review - Chapter {number}: {title}\n\n"
                         f"**Verdict: PASS** - no continuity issues found.\n")
        else:
            body = "\n".join(
                f"- **[{i['type']}]** {i['description']}"
                + (f"\n  Fix: {i['fix']}" if i["fix"] else "")
                for i in issues)
            save_interim(chapter_filename("review", number),
                         f"# Review - Chapter {number}: {title}\n\n"
                         f"**Verdict: {verdict.upper()}** ({len(issues)} issues)\n\n{body}\n")
        return verdict, issues
    except Exception as e:
        print(f"[REVIEWER] Error reviewing chapter {number}: {e}")
        return "pass", []  # don't block the pipeline on a review failure
