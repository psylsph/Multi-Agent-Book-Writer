"""
Editor Agent
Polishes each chapter and checks it for consistency against the story bible
(character names, world facts, constraints). Saves the assembled book and a
human-readable copy of the story bible to the output directory.
"""

import difflib
import re
import time
from pathlib import Path

from shared import prompts
from shared.context import context, update_context
from shared.llm_utils import clean_llm_text
from shared.llm_client import AbortRun, agent_enabled, \
    generate_prose, get_config
from shared.output import (atomic_write_text, chapter_filename,
                           format_bible_markdown, render_chapter, save_chapter,
                           save_interim, strip_heading)
from shared.consistency import (book_pov, check_chronology, dropped_names,
                                format_findings,
                                garbled_words, introduced_names, lint_book,
                                lint_chapter, max_length, pov_slip_finding,
                                repair_fused_words,
                                repeated_phrase_finding, style_rules,
                                word_count,
                                word_count_finding)
from shared.resume import note_chapter, save_state
from shared.scene_checks import (cast_finding, chapter_cast, place_finding,
                                 seed_places)
from shared.story_state import merge_states
from agents.reviewer import UNREVIEWED, run_reviewer


# A revision/polish that returns less than this fraction of the text it was
# given has almost certainly summarised or truncated the chapter - unless it
# still has the length the book asks for: tightening a 3,189-word chapter to
# 2,200 keeps 69% of it and is not truncation (see shrink_floor).
MIN_REVISION_RATIO = 0.75
MIN_POLISH_RATIO = 0.85
# A revision that grows the chapter by more than this added new material (a
# length expansion, a missing beat), which may contradict the story: it is
# re-reviewed even when the draft passed review.
REREVIEW_GROWTH = 0.10

# What one finding weighs when a revision or polish is judged better or worse
# than the text it replaces. A count alone would reject a revision that fixes
# a dead character walking about but trips two word quotas.
SEVERITY = {
    # the story contradicts itself
    "dead_resurrection": 5, "relationship_regression": 5,
    "relationship_leap": 5, "knowledge": 5, "timeline": 5, "setting": 5,
    "dead_character": 5, "already_met": 5,
    # the chapter misses its brief, the author's rules or its length
    "outline_gap": 3, "constraint": 3,
    # Length is reported and can earn expansion rounds, but never makes a
    # version "worse": quality over quantity. A tighter revision that fixes a
    # real finding was rejected (severity 3 -> 5) for dropping under the
    # minimum. Truncation is caught separately (MIN_REVISION_RATIO).
    "word_count": 0, "too_long": 0,
    # surface problems
    "name_mismatch": 2, "banned_word": 1, "quota": 1, "repeated_phrase": 1,
    # style rules (book.style) - surface level, but each one is concrete
    "banned_phrase": 1, "hedging": 1, "flat_prose": 1, "pov": 2, "tense": 2,
    "spelling": 1,
    # scene checks (shared/scene_checks.py): someone the outline leaves out is
    # in the scene - the brief missed; a county the book is not set in
    "cast": 3, "place": 2,
}
DEFAULT_SEVERITY = 3            # an issue type the reviewer made up


def severity(lint, issues=()):
    """Weighted total of lint findings and reviewer issues (see SEVERITY)."""
    return (sum(SEVERITY.get(f["check"], DEFAULT_SEVERITY) for f in lint)
            + sum(SEVERITY.get(i["type"], DEFAULT_SEVERITY) for i in issues))


def _earlier_bodies(n, drafts, final):
    """Bodies of the chapters before `n`: the edited text where it exists,
    otherwise the draft."""
    return [final.get(k) or drafts[k]
            for k in sorted(set(drafts) | set(final)) if k < n
            and (final.get(k) or drafts.get(k))]


def _sentences(text):
    out = []
    for para in text.split("\n\n"):
        out += re.split(r"(?<=[.!?])\s+", para.strip()) + [""]  # '' = break
    return out


def diff_report(number, title, original, final, notes):
    """Markdown report of what the editor changed: stats, the decisions it
    made, and a sentence-level unified diff of draft -> final."""
    a, b = _sentences(original), _sentences(final)
    diff = list(difflib.unified_diff(a, b, "draft", "final", lineterm="", n=1))
    same = difflib.SequenceMatcher(None, a, b).ratio()
    lines = [f"# Edit diff - Chapter {number}: {title}", "",
             f"- words: {word_count(original)} -> {word_count(final)}",
             f"- sentences unchanged: {same:.0%}"]
    lines += [f"- {note}" for note in notes]
    lines += ["", "```diff"] + (diff or ["(no changes)"]) + ["```", ""]
    return "\n".join(lines)


# Lines of the edit notes worth surfacing beside the chapter: where the
# editor gave up on something, not the routine "revision accepted".
_GAVE_UP = ("rejected", "failed", "discarded", "no progress", "could not be read")


def _repair_copy(n, source, text, notes, label):
    """Split words the model fused while copying `source` ("againsther"),
    in place of rejecting a revision or polish over a missing space."""
    text, fixes = repair_fused_words(source, text)
    if fixes:
        shown = ", ".join(f"{a} -> {b}" for a, b in fixes[:5])
        print(f"[EDITOR] Chapter {n}: repaired {len(fixes)} fused word(s) in "
              f"the {label} ({shown}).")
        notes.append(f"{label}: repaired fused words ({shown})")
    return text


def short_note(notes, lint, issues, unreviewed=False):
    """One line on what the edit left undone ('' when it went cleanly), for
    the progress view; the full story is in interim/diff_chapter_NN.md."""
    parts = []
    for note in notes:
        # the first review's status is reported below, from the final verdict
        if note.startswith(("round", "polish")) and any(
                word in note for word in _GAVE_UP):
            text = re.sub(r"^round \d+: ", "", note).split("\n")[0]
            if text not in parts:
                parts.append(text)
    if unreviewed:
        parts.append("no readable review")
    left = len(lint) + len(issues)
    if left:
        kinds = ", ".join(dict.fromkeys(
            [f["check"] for f in lint] + [i["type"] for i in issues]))
        parts.append(f"{left} finding(s) left: {kinds}")
    return "; ".join(parts)


def _bible_reference(bible):
    """Compact bible summary for the editor's consistency check."""
    names = ", ".join(c["name"] for c in bible.get("characters") or [])
    parts = []
    if names:
        parts.append(f"Character names (must be spelled exactly): {names}")
    if bible.get("world"):
        parts.append(f"World: {bible['world'][:800]}")
    constraints = bible.get("constraints") or []
    if constraints:
        parts.append("Constraints:\n" + "\n".join(f"- {c}" for c in constraints))
    if bible.get("notes"):
        parts.append(f"Author notes: {bible['notes']}")
    return "\n".join(parts) or "(no bible)"


def shrink_floor(before_words, ratio, length_words):
    """Fewest words a revision (ratio MIN_REVISION_RATIO, length = the
    chapter minimum) or polish (MIN_POLISH_RATIO, length = the target) may
    return. Shrinking past `ratio` is truncation only while it also goes
    under the length the book asks for; an over-long chapter may be
    tightened down to that length. 1806 -> 438 is still refused."""
    return min(ratio * before_words, length_words or ratio * before_words)


def _revise(number, title, draft, lint, issues, system=None):
    """Ask the model to fix specific findings in a draft."""
    length_note = ""
    wc = next((f for f in lint if f["check"] == "word_count"), None)
    if wc:
        current = word_count(draft)
        minimum = int(wc.get("minimum", 0))
        delta = max(minimum - current, 200)
        length_note = f"""
LENGTH - HARD REQUIREMENT: the draft is {current} words; the minimum is
{minimum}. You MUST return at least {minimum} words (add at least {delta} new
words). Write every beat out
in full scene: dialogue exchanges with back-and-forth, actions in sequence,
senses on the page. If a paragraph summarises what happened, dramatise it
instead. Do NOT pad: no repetition in different words, no filler modifiers,
no restating what the reader knows, no throat-clearing. Every new sentence
must add information, character, or momentum."""
    long_find = next((f for f in lint if f["check"] == "too_long"), None)
    if long_find:
        current, target = word_count(draft), int(long_find.get("target", 0))
        length_note = f"""
LENGTH - TIGHTEN: the draft is {current} words; aim for about {target}. Cut
words, not story: repetition, feelings the reader has already been shown,
beats described twice, redundant tags and filler. Keep every scene, every
event, every character who appears and every fact the chapter establishes.
Never replace a scene with a summary of it."""    # repeated after the chapter: on a long chapter, a local model follows
    # what it read last more reliably than rules 3000 words back
    kinds = ", ".join(dict.fromkeys(
        [f["check"] for f in lint] + [i["type"] for i in issues]))
    length_reminder = (f"; the result must be at least {minimum} words"
                       if wc else "")
    if long_find:
        length_reminder = (f"; tighten to about {target} words without "
                           "losing a scene, event or character")
    prompt = f"""You are revising Chapter {number} ("{title}") to fix specific, verified findings.

FINDINGS TO FIX
{format_findings(lint, issues)}
{length_note}
Rules:
- Fix ONLY the listed findings. Do not rewrite style, add scenes, or change plot beyond the fixes.
  (Exceptions: a LENGTH finding, where you add real scenes and detail - or tighten - as
  directed, and an outline_gap finding, where you add the missing beat as a short scene or
  passage.)
- For repeated_phrase findings, reword each listed phrase with fresh imagery; do not just swap one word.
- For banned words, replace them naturally; for quotas, cut the least-needed instances down to the limit.
- For continuity findings, apply the suggested fix or an equivalent minimal change.
- Keep the chapter's voice and all unaffected content intact.

CHAPTER (current draft):
{prompts.fence(draft, "chapter")}

Reminder: fix only these findings ({kinds}){length_reminder}. Return ONLY the full
revised chapter prose - every paragraph, not just the changed ones - with no
headings and no commentary."""
    revised = clean_llm_text(
        generate_prose(prompt, system=system, agent="editor"))
    if len(revised.split()) < 100:
        print(f"[EDITOR] Revision for chapter {number} too short "
              f"({len(revised.split())} words); discarding.")
        return None
    return revised


def run_editor(only=None):
    """
    Review-and-revise loop per chapter, then a final polish pass:
      1. deterministic lint (banned words, quotas, name & chronology checks)
      2. reviewer agent (dead characters, relationship/knowledge/timeline)
      3. bounded revision rounds until findings are fixed
      4. style polish, then a final lint guard
    On any failure the best version so far is kept (nothing is ever lost).

    only: edit just this chapter number (used by the review-as-you-go loop
    in main.py); None edits every chapter not yet edited. Either way the
    book and lint report are rewritten at the end, so a partial book is
    always on disk.
    """
    if only is None:
        print("[EDITOR] Starting review & edit phase...")
    cfg = get_config()
    bible = context.get("bible") or {}
    chapters = context.get("chapters", [])
    drafts = context.get("drafts", {})
    chronology = context.get("chronology") or {}
    constraints = bible.get("constraints") or []
    max_rounds = int(cfg["book"]["revision_rounds"])
    target_words = int(cfg["book"]["words_per_chapter"])
    tolerance = float(cfg["book"]["word_count_tolerance"])
    min_words = int(target_words * tolerance)
    max_wc_rounds = int(cfg["book"]["extra_length_rounds"])
    max_ratio = float(cfg["book"].get("max_length_ratio") or 0)
    max_words = max_length(target_words, max_ratio)
    reviewer_on = agent_enabled("reviewer")
    reference = _bible_reference(bible)
    name_ignore = cfg["book"]["name_lint_ignore"] or []
    repetition_on = bool(cfg["book"]["repetition_lint"])
    style_cfg = cfg["book"].get("style") or {}
    character_names = [c.get("name", "") for c in bible.get("characters") or []]
    places = seed_places(context.get("seed", ""), bible.get("setting", ""))
    alias_map = {c["name"]: c["aliases"] for c in bible.get("characters") or []
                 if c.get("aliases")}
    # role + bible reference live in the system message (identical for every
    # call, so prompt caching applies); the chapter stays in the user message
    revise_system = f"{prompts.REVISER}\n\nSTORY BIBLE REFERENCE\n{reference}"
    polish_system = f"{prompts.POLISHER}\n\nSTORY BIBLE REFERENCE\n{reference}"
    rules = style_rules(style_cfg, constraints)
    polish_rules = ("- Do not break these rules (they are checked after you):\n"
                    + "\n".join(f"  - {r}" for r in rules) + "\n") if rules else ""

    final = dict(context.get("final") or {})
    unreviewed = set(context.get("unreviewed") or ())
    for chapter in chapters:
        n, title = chapter["number"], chapter["title"]
        if only is not None and n != only:
            continue
        draft = drafts.get(n)
        if not draft:
            print(f"[EDITOR] No draft for chapter {n}; skipping.")
            continue
        if n in final:
            print(f"[EDITOR] Chapter {n}: already edited; skipping.")
            continue

        print(f"[EDITOR] Chapter {n}/{len(chapters)}: review & edit")
        original = draft
        notes = []
        earlier_texts = _earlier_bodies(n, drafts, final)
        # book.style.pov is checked by the style lint; without it, the person
        # the seed names, or third person, is held instead
        pov, pov_why = (None, "") if str(style_cfg.get("pov") or "").strip() \
            else book_pov(None, [context.get("seed", "")] + constraints)

        cast = chapter_cast(chapter.get("summary", ""),
                            bible.get("characters") or [])

        def full_lint(text):
            findings = lint_chapter(n, text, bible, constraints,
                                    ignore_names=name_ignore, style=style_cfg)
            for scene in (cast_finding(text, cast, bible.get("characters") or []),
                          place_finding(text, places)):
                if scene:
                    findings.append(scene)
            findings += check_chronology(
                n, text, merge_states(chronology, upto=n), alias_map)
            if pov:
                slip = pov_slip_finding(text, pov, pov_why)
                if slip:
                    findings.append(slip)
            wc_find = word_count_finding(text, target_words, tolerance,
                                         max_ratio)
            if wc_find:
                findings.append(wc_find)
            if repetition_on:
                rep_find = repeated_phrase_finding(text, earlier_texts,
                                                   character_names)
                if rep_find:
                    findings.append(rep_find)
            return findings

        print(f"[EDITOR] Chapter {n}: {word_count(draft)} words "
              f"(target {target_words}, min {min_words}"
              + (f", max {max_words}" if max_words else "") + ")")

        # ---- review & revise rounds
        lint = full_lint(draft)
        notes.append(f"lint on the draft: {len(lint)} finding(s)"
                     + ("".join(f"\n  - {f['check']}" for f in lint)))
        last_wc_expansion = word_count(draft)
        tightened = False
        verdict, issues = ("pass", [])
        if reviewer_on:
            print(f"[REVIEWER] Reviewing chapter {n}...")
            verdict, issues = run_reviewer(n, title, draft)
            print(f"[REVIEWER] Chapter {n}: {verdict}"
                  + (f" ({len(issues)} issues)" if issues else ""))
            notes.append(f"reviewer: {verdict}"
                         + (f" ({len(issues)} issues)" if issues else "")
                         + (" - the reply could not be read"
                            if verdict == UNREVIEWED else ""))
        else:
            print(f"[REVIEWER] disabled; deterministic lint only "
                  f"({len(lint)} findings)")

        for rnd in range(1, max_rounds + max_wc_rounds + 1):
            if not lint and not issues:
                break
            needs_length = any(f["check"] == "word_count" for f in lint)
            needs_trim = any(f["check"] == "too_long" for f in lint)
            if rnd > max_rounds and not needs_length:
                break  # extra rounds are reserved for length expansion
            if tightened and not issues and all(
                    f["check"] == "too_long" for f in lint):
                # one tightening pass: cutting harder on a second round is
                # how scenes get summarised away (quality over quantity)
                print(f"[EDITOR] Chapter {n}: still {word_count(draft)} "
                      "words after tightening; accepting.")
                notes.append(f"round {rnd}: still long after one tightening "
                             "pass; accepted")
                break
            if rnd > max_rounds:
                # keep expanding only while the revision is still growing
                wc_now = word_count(draft)
                growth = (wc_now - last_wc_expansion) \
                    / max(last_wc_expansion, 1)
                if rnd > max_rounds + 1 and growth < 0.15:
                    print(f"[EDITOR] Chapter {n}: expansion stalled at "
                          f"{wc_now} words; accepting.")
                    break
                last_wc_expansion = wc_now
            print(f"[EDITOR] Revision round {rnd}/"
                  f"{max_rounds + max_wc_rounds} for chapter "
                  f"{n}: {len(lint)} lint, {len(issues)} review issues")
            try:
                revised = _revise(n, title, draft, lint, issues,
                                  system=revise_system)
            except AbortRun:
                raise  # abort rather than silently shipping unedited chapters
            except Exception as e:
                print(f"[EDITOR] Revision failed for chapter {n}: {e}")
                notes.append(f"round {rnd}: revision failed ({e})")
                break
            if revised is None:
                notes.append(f"round {rnd}: revision discarded (too short)")
                break
            revised = _repair_copy(n, draft, revised, notes, f"round {rnd} revision")
            garbled = garbled_words(draft, revised)
            if garbled:
                print(f"[EDITOR] Revision of chapter {n} garbled word(s) "
                      f"({', '.join(garbled[:5])}); keeping the previous "
                      "version.")
                notes.append(f"round {rnd}: revision rejected (garbled words: "
                             + ", ".join(garbled[:5]) + ")")
                break
            if word_count(revised) < shrink_floor(
                    word_count(draft), MIN_REVISION_RATIO, min_words):
                print(f"[EDITOR] Revision for chapter {n} shrank the chapter "
                      f"({word_count(draft)} -> {word_count(revised)} words); "
                      "discarding it.")
                notes.append(f"round {rnd}: revision rejected (shrank the "
                             "chapter)")
                break
            # only a tightening round must keep everyone: fixing a dead
            # character who walks about removes him, so names the findings
            # mention are theirs to change
            asked = " ".join([f.get("detail", "") for f in lint]
                             + [str(i.get("description", "")) + " "
                                + str(i.get("fix", "")) for i in issues])
            lost = [m for m in dropped_names(draft, revised, character_names)
                    if not dropped_names(asked, "", [m])] if needs_trim else []
            if lost:
                print(f"[EDITOR] Revision of chapter {n} dropped "
                      f"{', '.join(lost)}; keeping the previous version.")
                notes.append(f"round {rnd}: revision rejected (dropped "
                             + ", ".join(lost) + ")")
                break
            new_lint = full_lint(revised)
            new_verdict, new_issues = verdict, issues
            # re-read when the size moved either way: new text can carry new
            # mistakes, and cut text can be a missing outline beat
            resized = (abs(word_count(revised) - word_count(draft))
                       > REREVIEW_GROWTH * word_count(draft))
            if reviewer_on and (issues or verdict != "pass" or resized):
                new_verdict, new_issues = run_reviewer(n, title, revised)
                if new_verdict == UNREVIEWED:
                    # an unreadable re-review is not a clean bill of health:
                    # assume the issues it was meant to confirm are still there
                    new_issues = issues
                    notes.append(f"round {rnd}: re-review could not be read; "
                                 "earlier issues assumed unfixed")
            before, after = severity(lint, issues), \
                severity(new_lint, new_issues)
            if after > before:
                print(f"[EDITOR] Revision made chapter {n} worse "
                      f"(severity {before} -> {after}); "
                      "keeping the previous version.")
                notes.append(f"round {rnd}: revision rejected (severity "
                             f"{before} -> {after})")
                break
            if needs_trim:
                tightened = True
            if (after == before and not needs_trim
                    and word_count(revised) < word_count(draft)):
                # shorter is fine when it is better; shorter AND no better
                # only loses text (length no longer weighs in severity)
                print(f"[EDITOR] Revision of chapter {n} is shorter and fixed "
                      "nothing; keeping the previous version.")
                notes.append(f"round {rnd}: revision rejected (shorter, "
                             "fixed nothing)")
                break
            # Progress: fewer/lighter findings, or - while expanding for
            # length - more words (the word_count finding stays until the
            # minimum is reached; the stall check above bounds that).
            progress = after < before or (
                needs_length and word_count(revised) > word_count(draft)) or (
                needs_trim and word_count(revised) < word_count(draft))
            notes.append(f"round {rnd}: revision accepted (findings "
                         f"{len(lint) + len(issues)} -> "
                         f"{len(new_lint) + len(new_issues)}, severity "
                         f"{before} -> {after})")
            draft, lint = revised, new_lint
            verdict, issues = new_verdict, new_issues
            if not progress:
                # not worse, so kept - but another round on the same findings
                # would most likely do the same again: stop spending calls
                print(f"[EDITOR] Chapter {n}: the revision fixed nothing "
                      f"(severity {after}); no further rounds.")
                notes.append(f"round {rnd}: no progress; stopping")
                break

        if lint:
            print(f"[EDITOR] Chapter {n}: {len(lint)} findings remain after "
                  f"revision (see lint_report.md)")

        # ---- final polish pass
        # The body only: the heading is ours to render, and asking the model
        # to keep one only gave it something to get wrong. The rules the lint
        # guard enforces are listed, or a polish that brings a banned phrase
        # back is thrown away whole.
        print(f"[EDITOR] Polishing chapter {n}...")
        prompt = f"""Copy-edit Chapter {n} ("{title}") below.

- Fix grammar, spelling, punctuation and clumsy phrasing; smooth a sentence
  that trips the reader.
- This is a copy-edit, not a rewrite: keep every paragraph, every event, the
  dialogue and the author's voice. Do not summarise, shorten, add scenes or
  rename anyone.
{polish_rules}
{prompts.fence(draft, "chapter")}

Return ONLY the edited chapter prose: no heading, no commentary."""
        try:
            # the heading is ours to write: whatever the model did with it
            # ("## Chapter Three", no heading at all) only the body is kept
            body = strip_heading(clean_llm_text(
                generate_prose(prompt, system=polish_system, agent="editor")))
        except AbortRun:
            raise  # abort rather than silently shipping the unpolished draft
        except Exception as e:
            print(f"[EDITOR] Error editing chapter {n}: {e}; keeping draft.")
            body = draft

        if body is not draft:
            body = _repair_copy(n, draft, body, notes, "polish")
        # deterministic guard: keep whichever version has fewer lint findings
        if body is draft:
            notes.append("polish failed; pre-polish version kept")
        elif word_count(body) < shrink_floor(
                word_count(draft), MIN_POLISH_RATIO,
                target_words if max_words else 0):
            print(f"[EDITOR] Polish shortened chapter {n} "
                  f"({word_count(draft)} -> {word_count(body)} words); "
                  "keeping pre-polish version.")
            notes.append("polish rejected (shortened the chapter)")
            body = draft
        elif garbled_words(draft, body):
            garbled = garbled_words(draft, body)
            print(f"[EDITOR] Polish of chapter {n} garbled word(s) "
                  f"({', '.join(garbled[:5])}); keeping pre-polish version.")
            notes.append("polish rejected (garbled words: "
                         + ", ".join(garbled[:5]) + ")")
            body = draft
        elif dropped_names(draft, body, character_names):
            lost = dropped_names(draft, body, character_names)
            print(f"[EDITOR] Polish of chapter {n} dropped "
                  f"{', '.join(lost)}; keeping pre-polish version.")
            notes.append("polish rejected (dropped " + ", ".join(lost) + ")")
            body = draft
        elif introduced_names(draft, body):
            new = introduced_names(draft, body)
            print(f"[EDITOR] Polish of chapter {n} introduced new name(s) "
                  f"({', '.join(new)}); keeping pre-polish version.")
            notes.append("polish rejected (introduced new names: "
                         + ", ".join(new) + ")")
            body = draft
        elif severity(full_lint(body)) > severity(full_lint(draft)):
            print(f"[EDITOR] Polish introduced lint findings for chapter {n}; "
                  "keeping pre-polish version.")
            notes.append("polish rejected (introduced lint findings)")
            body = draft
        else:
            notes.append("polish accepted")

        if reviewer_on and verdict == UNREVIEWED:
            unreviewed.add(n)
            print(f"[EDITOR] Chapter {n}: finished WITHOUT a readable review "
                  "(see review_chapter_NN.md).")
        else:
            unreviewed.discard(n)
        update_context("unreviewed", unreviewed)
        final[n] = body
        update_context("final", final)
        save_state("final", final)
        save_state("unreviewed", sorted(unreviewed))
        note_chapter(n, edited_at=time.time(),
                     note=short_note(notes, lint, issues,
                                     reviewer_on and verdict == UNREVIEWED))
        save_interim(chapter_filename("diff", n),
                     diff_report(n, title, original, body, notes))
        save_interim(chapter_filename("edited", n),
                     render_chapter(n, title, body))
        # replace the draft in the durable per-chapter file with the
        # polished version
        save_chapter(n, title, body)
        print(f"[EDITOR] Chapter {n} complete.")

    update_context("final", final)
    # one chapter at a time (review as you go): keep rewriting the same book
    # file, or overwrite: false would leave one numbered copy per chapter
    return finalize_book(final, reuse_path=only is not None)


def _in_book_order(bodies):
    """[(number, title, body)] for the chapters in `bodies` ({number: body}),
    in outline order."""
    titles = {c["number"]: c["title"] for c in context.get("chapters") or []}
    order = [n for n in titles if n in bodies]
    order += sorted(n for n in bodies if n not in titles)
    return [(n, titles.get(n, ""), bodies[n]) for n in order if bodies[n]]


def finalize_book(bodies, reuse_path=False):
    """Re-emit the lint report and the assembled book. Idempotent.
    bodies: {chapter number: body text}. reuse_path: see save_book()."""
    _write_lint_report(bodies)
    return save_book(bodies, reuse_path=reuse_path)


def _write_lint_report(bodies):
    """Final deterministic lint report across the whole book."""
    cfg = get_config()
    bible = context.get("bible") or {}
    constraints = bible.get("constraints") or []
    target_words = int(cfg["book"]["words_per_chapter"])
    tolerance = float(cfg["book"]["word_count_tolerance"])
    min_words = int(target_words * tolerance)
    max_words = max_length(target_words, cfg["book"].get("max_length_ratio"))
    chapters = _in_book_order(bodies)
    full_text = "\n\n".join(render_chapter(n, t, b) for n, t, b in chapters)
    findings = lint_book(
        full_text, bible, constraints,
        cfg["book"]["name_lint_ignore"] or [],
        {n: b for n, _, b in chapters} if cfg["book"]["repetition_lint"]
        else None,
        style=cfg["book"].get("style") or {})

    # per-chapter word counts (wc -w semantics)
    counts = []
    for n, _, body in chapters:
        wc = word_count(body)
        flag = " **SHORT**" if wc < min_words else (
            " **LONG**" if max_words and wc > max_words else "")
        counts.append(f"- Chapter {n}: {wc} words{flag} (min {min_words}"
                      + (f", max {max_words}" if max_words else "") + ")")

    lines = ["# Lint Report (final book)", "",
             f"Chapters: {len(chapters)}  "
             f"Words: {word_count(full_text)}  "
             f"Target/chapter: {target_words} (min {min_words})", "",
             "## Chapter word counts", ""]
    lines += counts or ["(none)"]
    missed = sorted(n for n in context.get("unreviewed") or () if n in bodies)
    if missed:
        lines += ["", "## Not reviewed", "",
                  "The reviewer's reply could not be read for these chapters, "
                  "so they were not checked for continuity, outline or "
                  "constraint problems: "
                  + ", ".join(str(n) for n in missed)]
    lines += ["", "## Findings", ""]
    if not findings:
        lines.append("No deterministic findings. "
                     "(LLM continuity reviews are in review_chapter_NN.md)")
    else:
        lines += [f"- [{f['check']}] {f['detail']}" for f in findings]
    save_interim("lint_report.md", "\n".join(lines) + "\n")


def save_book(bodies, reuse_path=False):
    """Assemble and write the book + story bible to the output directory.
    bodies: {chapter number: body text}; headings come from the outline.

    With output.overwrite off each call normally writes a NEW numbered file.
    reuse_path=True rewrites the file this run already wrote instead, for
    callers that save the same growing book repeatedly."""
    cfg = get_config()
    out_dir = Path(cfg["output"]["directory"])
    out_dir.mkdir(parents=True, exist_ok=True)

    filename = cfg["output"]["filename"]
    out_path = out_dir / filename
    previous = context.get("output_path")
    if reuse_path and previous and not cfg["output"]["overwrite"]:
        out_path = Path(previous)
    elif not cfg["output"]["overwrite"] and out_path.exists():
        stem, suffix = out_path.stem, out_path.suffix or ".txt"
        i = 1
        while (out_dir / f"{stem}-{i}{suffix}").exists():
            i += 1
        out_path = out_dir / f"{stem}-{i}{suffix}"

    title = context.get("title") or "Untitled"
    book = (f"# {title}\n\n" + "\n\n".join(
        render_chapter(n, t, b) for n, t, b in _in_book_order(bodies)) + "\n")
    try:
        atomic_write_text(out_path, book)
        update_context("output_path", str(out_path))
        print(f"[EDITOR] Final book saved to {out_path}")
    except OSError as e:
        print(f"[EDITOR] Error saving book: {e}")
        return book

    # human-readable copy of the story bible for easy inspection
    bible = context.get("bible") or {}
    try:
        bible_path = out_dir / "story_bible.md"
        atomic_write_text(bible_path, format_bible_markdown(bible))
        print(f"[EDITOR] Story bible saved to {bible_path}")
    except OSError as e:
        print(f"[EDITOR] Warning: could not save story bible: {e}")

    return book
