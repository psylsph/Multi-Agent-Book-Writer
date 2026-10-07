"""
Reviewer Agent
Reviews each chapter draft on up to three axes (book.review_checks):

  continuity   against the accumulated story state (who is alive, who has met
               whom, relationship progression, timeline) plus the story bible
  outline      does the chapter actually deliver the beats its outline entry
               and lore brief promise?
  constraints  does it obey the author's rules that a word-counting linter
               cannot check (point of view, tense, content and style rules)?

Returns a structured verdict + issues; the editor agent feeds the findings
into bounded revision rounds.
"""

from shared import prompts
from shared.context import context
from shared.llm_client import (EndpointUnavailable, generate_with_wait,
                               get_config)
from shared.llm_utils import extract_json
from shared.output import chapter_filename, save_interim
from shared.story_state import (merge_states, render_story_facts,
                                render_story_so_far)

ALL_CHECKS = ("continuity", "outline", "constraints")

CONTINUITY_TYPES = ("dead_resurrection", "relationship_regression",
                    "relationship_leap", "knowledge", "timeline", "setting")
_TYPES_BY_CHECK = {"continuity": CONTINUITY_TYPES,
                   "outline": ("outline_gap",),
                   "constraints": ("constraint",)}

_CONTINUITY_CHECKS = """\
1. DEAD RESURRECTION: a character who died earlier acting, speaking, or being present as alive (memories/dreams/dialogue ABOUT them are fine).
2. RELATIONSHIP REGRESSION: characters who have already met (see facts) acting like strangers, or intimate characters suddenly formal/distant without cause - e.g. reintroducing themselves, "pleased to meet you", surprise at knowing each other, or lovers who forget their history.
3. RELATIONSHIP LEAP: characters closer/more intimate than their recorded state allows (they cannot be lovers if they only met last chapter with no escalation shown).
4. KNOWLEDGE ERRORS: a character knowing a secret that was never revealed to them, or forgetting something they personally witnessed.
5. TIMELINE ERRORS: impossible ordering, time of day/date contradicting the timeline, injuries healed instantly, day/night mismatches.
6. SETTING ERRORS: characters in places they cannot have reached, or locations contradicting earlier chapters."""

_OUTLINE_CHECK = """\
OUTLINE_GAP: a key beat or event that the CHAPTER OUTLINE or its plot beats require is ENTIRELY ABSENT from the draft, or the chapter does not end where the outline says it should. A beat that is present but brief or understated is NOT a gap. Name the missing beat and say where a short passage could deliver it."""

_CONSTRAINT_CHECK = """\
CONSTRAINT: the draft breaks one of the AUTHOR CONSTRAINTS below in a way a word counter cannot detect - point of view, tense, content rules, style rules. Skip mechanical rules (banned words, quotas); a linter checks those. Quote the offending passage and say how to fix it."""


def enabled_checks():
    """book.review_checks as an ordered tuple of known names (default all)."""
    wanted = get_config()["book"].get("review_checks")
    if isinstance(wanted, str):
        wanted = [wanted]
    names = [str(w).strip().lower() for w in wanted or ()]
    chosen = tuple(c for c in ALL_CHECKS if c in names)
    return chosen or ALL_CHECKS


def _build_prompt(number, title, draft, checks, bible):
    chronology = context.get("chronology") or {}
    parts = ["Review this chapter draft. Report only real, specific "
             "problems of the kinds listed below; style opinions are not "
             "problems."]
    types = []

    if "continuity" in checks:
        prior = merge_states(chronology, upto=number)  # earlier chapters only
        window = int(get_config()["book"].get("summary_window", 8))
        recap = render_story_so_far(
            {n: s.get("summary", "") for n, s in chronology.items()},
            before=number, window=window) or "(first chapter)"
        parts.append(f"STORY FACTS BEFORE THIS CHAPTER\n"
                     f"{render_story_facts(prior)}\n\n"
                     f"Chapter summaries so far:\n{recap}")
        types += CONTINUITY_TYPES

    if "outline" in checks:
        chapter = next((c for c in context.get("chapters") or []
                        if c["number"] == number), {})
        brief = (context.get("research") or {}).get(number, "")
        parts.append(f"CHAPTER OUTLINE (what this chapter must deliver)\n"
                     f"{chapter.get('summary') or '(none given)'}"
                     + (f"\n\nLORE BRIEF (plot beats to hit)\n{brief[:1800]}"
                        if brief else ""))
        types.append("outline_gap")

    if "constraints" in checks:
        rules = bible.get("constraints") or []
        parts.append("AUTHOR CONSTRAINTS\n" + (
            "\n".join(f"- {r}" for r in rules) if rules else
            "(none listed; follow the style notes in the story bible)"))
        types.append("constraint")

    parts.append(f'CHAPTER {number} DRAFT TO REVIEW ("{title}"):\n'
                 f'"""{draft}"""')

    checklist = []
    if "continuity" in checks:
        checklist.append("CONTINUITY failures:\n" + _CONTINUITY_CHECKS)
    if "outline" in checks:
        checklist.append(_OUTLINE_CHECK)
    if "constraints" in checks:
        checklist.append(_CONSTRAINT_CHECK)
    parts.append("Check for these specific failures:\n\n"
                 + "\n\n".join(checklist))
    parts.append(
        'Return ONLY JSON (no fences):\n'
        '{"verdict": "pass" or "revise",\n'
        ' "issues": [{"type": "' + "|".join(types) + '", "description": '
        '"what exactly is wrong, with quotes", "fix": "the minimal change '
        'that fixes it"}]}\n\n'
        "Only report real problems with the material above. Empty issues + "
        '"pass" if the chapter is sound.')
    return "\n\n".join(parts)


def review_chapter(number, title, draft):
    """Review one chapter draft (see the module docstring for the checks).

    Returns (verdict, issues) where verdict is "pass" or "revise" and issues
    is a list of {type, description, fix} dicts. Raises when the model's
    answer cannot be read (and EndpointUnavailable when the server is down);
    run_reviewer() turns the former into a pass so a bad reply never blocks a
    run, but an evaluation needs to tell the two apart.
    """
    bible = context.get("bible") or {}
    checks = enabled_checks()
    prompt = _build_prompt(number, title, draft, checks, bible)
    # issue types that belong to a check that is switched off are dropped
    disabled = {t for check, types in _TYPES_BY_CHECK.items()
                if check not in checks for t in types}

    raw = generate_with_wait(
        prompt, system=prompts.with_bible(prompts.REVIEWER, bible),
        agent="reviewer", json_mode=True)
    data = extract_json(raw, expect="object")
    issues = []
    for issue in data.get("issues") or []:
        if isinstance(issue, dict) and issue.get("description"):
            kind = str(issue.get("type", "continuity")).strip()
            if kind in disabled:
                continue
            issues.append({
                "type": kind,
                "description": str(issue["description"]).strip(),
                "fix": str(issue.get("fix", "")).strip(),
            })
    verdict = "revise" if issues else "pass"
    scope = ", ".join(checks)
    if not issues:
        save_interim(chapter_filename("review", number),
                     f"# Review - Chapter {number}: {title}\n\n"
                     f"**Verdict: PASS** - no issues found "
                     f"(checked: {scope}).\n")
    else:
        body = "\n".join(
            f"- **[{i['type']}]** {i['description']}"
            + (f"\n  Fix: {i['fix']}" if i["fix"] else "")
            for i in issues)
        save_interim(chapter_filename("review", number),
                     f"# Review - Chapter {number}: {title}\n\n"
                     f"**Verdict: REVISE** ({len(issues)} issues; "
                     f"checked: {scope})\n\n{body}\n")
    return verdict, issues


def run_reviewer(number, title, draft):
    """review_chapter(), but a reply that cannot be read counts as a pass: a
    review failure must never block the pipeline. A server outage still
    aborts (don't silently mark chapters reviewed while it is down)."""
    try:
        return review_chapter(number, title, draft)
    except EndpointUnavailable:
        raise
    except Exception as e:
        print(f"[REVIEWER] Error reviewing chapter {number}: {e}")
        return "pass", []
