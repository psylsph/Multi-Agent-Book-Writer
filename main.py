"""
Main Pipeline
Orchestrates the multi-agent book writing workflow:

    seed prompt -> seed review (size check, questions) -> architect (story bible)
                 -> planner (outline)
                 -> researcher (lore briefs) -> writer (drafts + summaries)
                 -> editor (polish + consistency) -> output
"""

import argparse
import contextlib
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from agents.architect import run_architect
from agents.planner import extract_seed_outline, run_planner
from agents.seed_review import run_seed_review
from agents.researcher import run_researcher
from agents.writer import refresh_state, run_writer
from agents.editor import finalize_book, run_editor, save_book
from shared import llm_client, runlog, web_search
from shared.context import get_context, reset_context, update_context
from shared.llm_client import EndpointUnavailable, agent_enabled, \
    get_config, load_config, preflight
from shared.output import (archive_previous_run, interim_dir,
                           interim_enabled, save_interim, save_interim_json)
from shared.resume import StateWriteError, has_resume, load_plan, \
    load_state, summarize_for_log

EXAMPLE_SEED = Path(__file__).resolve().parent / "seeds" / "example_seed.md"


def positive_int(value):
    try:
        i = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"'{value}' is not an integer")
    if i < 1:
        raise argparse.ArgumentTypeError("chapter count must be >= 1")
    return i


def resolve_seed_text(args):
    """Get the seed prompt from --seed FILE, --prompt TEXT, or --demo/bundled example."""
    if args.demo:
        if args.seed or args.prompt:
            sys.exit("Error: --demo cannot be combined with --seed or --prompt")
        if not EXAMPLE_SEED.exists():
            sys.exit(f"Error: example seed missing: {EXAMPLE_SEED}")
        print(f"[PIPELINE] Using the bundled example seed ({EXAMPLE_SEED.name}).")
        return EXAMPLE_SEED.read_text(encoding="utf-8")
    if args.seed:
        seed_path = Path(args.seed)
        if not seed_path.exists():
            sys.exit(f"Error: seed file not found: {seed_path}")
        return seed_path.read_text(encoding="utf-8")
    if args.prompt:
        return args.prompt
    if not EXAMPLE_SEED.exists():
        sys.exit(f"Error: no seed given and example seed missing: {EXAMPLE_SEED}")
    print("[PIPELINE] No --seed or --prompt given; "
          f"using the bundled example seed ({EXAMPLE_SEED.name}).")
    return EXAMPLE_SEED.read_text(encoding="utf-8")


def _norm_seed(text):
    """Seed text normalised for comparison (line endings, outer blanks)."""
    return (text or "").replace("\r\n", "\n").strip()


def check_resume(args, state, num_chapters):
    """Decide which seed a resumed run uses and refuse unsafe resumes.

    Interim files belong to ONE seed. Resuming them under a different seed
    would skip every saved chapter and ship the old book under the new
    bible, so that is an error unless --no-resume is given. With no seed
    argument at all, the seed saved with the run is reused.
    """
    if not state["bible"]:
        sys.exit("Error: the saved run in output/state/ is unreadable "
                 "(bible.json missing or corrupt). Rerun with --no-resume to "
                 "start over.")
    stored = state["bible"].get("seed", "")
    explicit = bool(args.demo or args.seed or args.prompt)
    if not explicit and stored:
        print("[RESUME] Reusing the seed saved with the run.")
        seed_text = stored
    else:
        seed_text = resolve_seed_text(args)
        if stored and _norm_seed(seed_text) != _norm_seed(stored):
            sys.exit(
                "Error: the saved run is for a DIFFERENT seed. "
                "Resuming would reuse its chapters for this seed. Rerun with "
                "--no-resume to start fresh, or pass the original seed to "
                "resume.")
    saved = state["chapters"] or []
    if num_chapters and saved and len(saved) != num_chapters:
        sys.exit(f"Error: the saved run has {len(saved)} chapters but "
                 f"-c {num_chapters} was given. Rerun with --no-resume to "
                 "start fresh, or drop -c to resume.")
    if saved and len(state["final"]) >= len(saved):
        print("[RESUME] The previous run already finished; the book will be "
              "reassembled from the saved chapters. Use --no-resume to "
              "write a new one.")
    return seed_text


def _research_complete(state):
    """True when a resumed run already has a brief for every chapter, so the
    researcher (and web search) has nothing left to do."""
    if not state or not state["chapters"]:
        return False
    return all(state["research"].get(c["number"]) for c in state["chapters"])


def apply_plan(plan):
    """Use the book size settled by the seed review (saved in plan.json, so a
    resumed run keeps it). Returns the chapter count to plan, or None."""
    if not plan:
        return None
    wpc = plan.get("words_per_chapter")
    if isinstance(wpc, int) and wpc > 0:
        get_config()["book"]["words_per_chapter"] = wpc
    chapters = plan.get("chapters")
    return chapters if isinstance(chapters, int) and chapters > 0 else None


def _hydrate_resume(state):
    """Push loaded resume state into the shared context."""
    if state["bible"]:
        update_context("bible", state["bible"])
        update_context("title", state["bible"].get("title", ""))
        update_context("seed", state["bible"].get("seed", ""))
    if state["chapters"]:
        update_context("chapters", state["chapters"])
    if state["research"]:
        update_context("research", state["research"])
    if state["drafts"]:
        update_context("drafts", state["drafts"])
    if state["summaries"] is not None:
        update_context("summaries", state["summaries"])
    if state["chronology"] is not None:
        update_context("chronology", state["chronology"])
    if state["final"]:
        update_context("final", dict(state["final"]))
    if state.get("unreviewed"):
        update_context("unreviewed", set(state["unreviewed"]))


@contextlib.contextmanager
def _phase(phases, name):
    """Record how long a pipeline phase took (even when it fails)."""
    started = time.time()
    try:
        yield
    finally:
        phases.append((name, time.time() - started))


def _report_stats(phases):
    """Print and save the run's LLM usage and phase timings. Never raises."""
    try:
        if not llm_client.STATS:
            return
        print("\n" + llm_client.stats_markdown(phases))
        save_interim("run_stats.md", llm_client.stats_markdown(phases))
        save_interim_json("run_stats.json", llm_client.stats_data(phases))
    except Exception as e:  # reporting must never mask the real outcome
        print(f"[PIPELINE] Could not write run statistics: {e}")


def review_as_you_go():
    """True when chapters are reviewed one at a time as they are written
    (book.review_as_you_go). Needs the editor; ignored when it is disabled."""
    wanted = bool(get_config()["book"]["review_as_you_go"])
    if wanted and not agent_enabled("editor"):
        print("[PIPELINE] book.review_as_you_go needs the editor, which is "
              "disabled; writing all chapters first instead.")
        return False
    return wanted


def run_interleaved():
    """Write, review and polish ONE chapter at a time.

    Each chapter is drafted, put through lint/review/revise/polish, and its
    summary and story facts are re-extracted from the final text before the
    next chapter is written. Later chapters therefore build on the corrected
    earlier ones instead of on unreviewed drafts. Stops at the first chapter
    that can't be drafted; a rerun resumes from there.
    """
    chapters = get_context("chapters")
    for chapter in chapters:
        n = chapter["number"]
        print(f"\n[PIPELINE] Chapter {n}/{len(chapters)}: {chapter['title']}")
        run_writer(only=n)
        if not (get_context("drafts") or {}).get(n):
            print(f"[PIPELINE] Chapter {n} could not be drafted; stopping. "
                  "Rerun the same command to resume from this chapter.")
            return
        if n in (get_context("final") or {}):
            continue  # edited (and refreshed) in an earlier run
        run_editor(only=n)
        refresh_state(n)
    finalize_book(get_context("final"))


def _incomplete_chapters(context, editor_on):
    """Numbers of planned chapters that never reached their final form
    (edited when the editor is on, drafted otherwise)."""
    done = (set(context.get("final") or ())
            if editor_on else set(context.get("drafts") or ()))
    return [c["number"] for c in context.get("chapters") or []
            if c["number"] not in done]


@dataclass
class Run:
    """One pipeline run: its settings, plus what one stage hands the next."""
    seed_text: str
    num_chapters: int | None = None   # None: the seed review/planner decide
    resuming: bool = False
    plan_only: bool = False
    interleave: bool = False          # book.review_as_you_go (and editor on)
    clarifications: list = field(default_factory=list)


@dataclass(frozen=True)
class Stage:
    """One step of the pipeline.

    run(run) does the work and returns an exit code to stop the pipeline
    there, or None to go on. wanted(run) False leaves the stage out entirely
    (an agent switched off, the other branch of review-as-you-go). done(run)
    returns a message when the saved run already has the stage's result: the
    message is printed instead of running it. Per-chapter resume (skipping
    finished chapters) is done inside the agents.
    """
    name: str                       # phase name in the run statistics
    title: str                      # the step banner
    run: Callable[[Run], int | None]
    wanted: Callable[[Run], bool] = lambda run: True
    done: Callable[[Run], str | None] = lambda run: None


def _seed_review(run):
    plan = run_seed_review(run.seed_text, run.num_chapters,
                           len(extract_seed_outline(run.seed_text)))
    if plan["stop"]:
        print("\n[PIPELINE] Stopped at the seed review; nothing was "
              "written. Edit the seed (see output/interim/"
              "seed_review.md) and rerun.")
        return 0
    run.clarifications = plan["clarifications"]
    run.num_chapters = apply_plan(plan) or run.num_chapters
    return None


def _architect(run):
    run_architect(run.seed_text, run.clarifications)


def _planner(run):
    if not run_planner(num_chapters=run.num_chapters):
        print("[PIPELINE] Planning failed. Exiting.")
        return 1
    if run.plan_only:
        print("\n[PIPELINE] Plan only: stopping after the outline "
              "(output/interim/outline.md). Rerun without --plan-only to "
              "write the book; it continues from this plan.")
        return 0
    return None


def _researcher(run):
    run_researcher()


def _write_and_review(run):
    run_interleaved()


def _writer(run):
    run_writer()


def _editor(run):
    run_editor()


def _save_drafts(run):
    save_book(get_context("drafts"))


STAGES = (
    Stage("seed review", "Step 0: Seed review (length and gaps)",
          _seed_review, wanted=lambda run: not run.resuming),
    Stage("architect", "Step 1: Architect (story bible)", _architect,
          done=lambda run: (run.resuming and get_context("bible")
                            and "Story bible loaded from the saved run; "
                                "skipping.") or None),
    Stage("planner", "Step 2: Planner (chapter outline)", _planner),
    Stage("researcher", "Step 3: Researcher (lore briefs)",
          _researcher,
          wanted=lambda run: agent_enabled("researcher")),
    # review as you go: each chapter is written, then reviewed and polished,
    # before the next one is started
    Stage("write + review", "Steps 4-5: Write & Review chapter by chapter",
          _write_and_review, wanted=lambda run: run.interleave),
    Stage("writer", "Step 4: Writer (chapter drafts)",
          _writer, wanted=lambda run: not run.interleave),
    Stage("editor", "Step 5: Review & Edit (continuity, lint, revise, polish)",
          _editor,
          wanted=lambda run: not run.interleave and agent_enabled("editor")),
    Stage("save drafts", "Step 5: Editor disabled; saving drafts.",
          _save_drafts,
          wanted=lambda run: not run.interleave
          and not agent_enabled("editor")),
)


def _run_stages(run, phases):
    """Run every wanted stage in order. Returns an exit code when a stage
    stopped the pipeline, else None."""
    for stage in STAGES:
        if not stage.wanted(run):
            continue
        print(f"\n[PIPELINE] {stage.title}")
        print("-" * 60)
        skip = stage.done(run)
        if skip:
            print(f"[PIPELINE] {skip}")
            continue
        with _phase(phases, stage.name):
            code = stage.run(run)
        if code is not None:
            return code
    return None


def _summary(started):
    """Print the end-of-run summary. Returns the exit code (1 when chapters
    are unfinished)."""
    context = get_context()
    incomplete = _incomplete_chapters(context, agent_enabled("editor"))
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE!" if not incomplete else "PIPELINE INCOMPLETE")
    print("=" * 60)
    print(f"Title: {context['title']}")
    print(f"Chapters: {len(context['chapters'])}")
    print(f"Drafted: {len(context['drafts'])} | "
          f"Edited: {len(context['final'])}")
    print(f"Total time: {(time.time() - started) / 60:.1f} minutes")
    unreviewed = sorted(context.get("unreviewed") or ())
    if unreviewed:
        print("Not reviewed (the reviewer's reply could not be read): "
              "chapters " + ", ".join(str(n) for n in unreviewed)
              + ". See interim/review_chapter_NN.md.")
    if incomplete:
        print("Chapters not finished: "
              + ", ".join(str(n) for n in incomplete))
        print("The book saved so far is partial. Fix the error above, "
              "then rerun the same command to resume.")
    print("=" * 60)
    return 1 if incomplete else 0


def run_pipeline(seed_text, num_chapters=None, resuming=False, state=None,
                 plan_only=False):
    """
    Execute the complete book writing pipeline (see STAGES).

    Args:
        seed_text: the creative seed (premise, characters, world, outline...)
        num_chapters: explicit chapter count override (None = seed/config)
        resuming: continue from the saved state instead of starting over
        state: already-loaded resume state (loaded here when omitted)
        plan_only: stop after the outline (seed review, bible, plan)
    """
    print("=" * 60)
    print("MULTI-AGENT BOOK WRITER")
    print("=" * 60)

    started = time.time()
    llm_client.reset_stats()
    phases = []
    reset_context()
    run = Run(seed_text, num_chapters, resuming, plan_only)
    if resuming:
        state = state or load_state()
        _hydrate_resume(state)
        print(f"[RESUME] Loaded {summarize_for_log(state)}.")
        # the size the seed review settled when the run began
        run.num_chapters = num_chapters or apply_plan(load_plan())
    else:
        archived = archive_previous_run()  # moved aside, never deleted
        if archived:
            print(f"[PIPELINE] Previous run archived to {archived}/")
    if interim_enabled():
        print(f"[PIPELINE] Interim artifacts: {interim_dir()}/")

    try:
        run.interleave = review_as_you_go()
        code = _run_stages(run, phases)
        return code if code is not None else _summary(started)
    except KeyboardInterrupt:
        print("\n[PIPELINE] Interrupted by user. Rerun the same command to "
              "resume.")
        return 130
    except EndpointUnavailable as e:
        print(f"\n[PIPELINE] Aborted: the LLM endpoint went down ({e})")
        print("[PIPELINE] Start/restart the server, then rerun the same "
              "command -- resume will pick up from the last saved chapter.")
        return 1
    except StateWriteError as e:
        print(f"\n[PIPELINE] Aborted: {e}")
        print("[PIPELINE] Free some disk space or fix the permissions, then "
              "rerun the same command to resume from the last saved step.")
        return 1
    except Exception as e:
        print(f"\n[PIPELINE] Error: {e}")
        raise
    finally:
        _report_stats(phases)


def main():
    parser = argparse.ArgumentParser(
        description="Multi-agent book writer: feed a seed prompt with "
                    "characters, world and outline; get a drafted book.")
    parser.add_argument("n", nargs="?", type=positive_int, default=None,
                        metavar="N",
                        help="number of chapters (legacy positional form)")
    parser.add_argument("-c", "--chapters", type=positive_int, default=None,
                        help="number of chapters (overrides the seed outline "
                             "and the config default)")
    parser.add_argument("--seed", metavar="FILE",
                        help="path to a seed prompt file (.md/.txt)")
    parser.add_argument("--prompt", metavar="TEXT",
                        help="inline seed prompt text")
    parser.add_argument("--demo", action="store_true",
                        help="use the bundled example seed")
    parser.add_argument("--config", default="config.yaml",
                        help="config file path (default: config.yaml)")
    parser.add_argument("--model", help="override llm.model (agents with their "
                             "own agents.<name>.model keep it)")
    parser.add_argument("--out", dest="out_file",
                        help="override the output filename "
                             "(written under the configured output dir)")
    parser.add_argument("--plan-only", action="store_true",
                        help="stop after the seed review, story bible and "
                             "outline; rerun without it to write the book")
    parser.add_argument("--seed-review", choices=("ask", "warn", "off"),
                        help="override book.seed_review: ask questions and "
                             "choose the size (ask), only report (warn), or "
                             "skip the check (off)")
    parser.add_argument("--no-resume", action="store_true",
                        help="ignore any saved run and restart "
                             "from the seed prompt")
    args = parser.parse_args()

    num_chapters = args.chapters if args.chapters is not None else args.n

    # Load config, apply CLI overrides
    try:
        cfg = load_config(args.config)
    except FileNotFoundError as e:
        sys.exit(f"Error: {e}")
    if args.model:
        cfg["llm"]["model"] = args.model
    if args.out_file:
        cfg["output"]["filename"] = args.out_file
    if args.seed_review:
        cfg["book"]["seed_review"] = args.seed_review

    if cfg["output"]["log"]:
        log_path = runlog.start_log(Path(cfg["output"]["directory"]) / "logs")
        if log_path:
            print(f"[PIPELINE] Logging to {log_path}")
    try:
        _run(args, cfg, num_chapters)
    finally:
        runlog.stop_log()


def _run(args, cfg, num_chapters):
    resuming = not args.no_resume and has_resume()
    state = load_state() if resuming else None
    seed_text = (check_resume(args, state, num_chapters) if resuming
                 else resolve_seed_text(args))

    # Make sure the LLM endpoint is up and the model exists before any work
    try:
        preflight()
    except (ConnectionError, RuntimeError) as e:
        sys.exit(f"Error: {e}")

    # Web fact-checking: make sure SearXNG is up now (offering to start it in
    # Docker) rather than discovering a missing server mid-run.
    if (web_search.enabled() and agent_enabled("researcher")
            and not _research_complete(state)):
        web_search.offer_docker_setup()

    exit_code = run_pipeline(seed_text, num_chapters=num_chapters,
                             resuming=resuming, state=state,
                             plan_only=args.plan_only)
    if exit_code == 0 and get_context("output_path"):
        out = get_context("output_path")
        print(f"\n\u2713 Book written! Check {out}")
    sys.exit(exit_code or 0)


if __name__ == "__main__":
    main()
