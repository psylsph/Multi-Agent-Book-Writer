"""
Main Pipeline
Orchestrates the multi-agent book writing workflow:

    seed prompt -> architect (story bible) -> planner (outline)
                 -> researcher (lore briefs) -> writer (drafts + summaries)
                 -> editor (polish + consistency) -> output
"""

import argparse
import sys
import time
from pathlib import Path

from agents.architect import run_architect
from agents.planner import run_planner
from agents.researcher import run_researcher
from agents.writer import run_writer
from agents.editor import run_editor, save_book
from shared.context import get_context, reset_context
from shared.llm_client import get_config, load_config, preflight
from shared.output import clear_interim, interim_dir, interim_enabled

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


def agent_enabled(name):
    """Check the enabled flag for an agent in config.yaml (default True)."""
    return get_config().get("agents", {}).get(name, {}).get("enabled", True)


def run_pipeline(seed_text, num_chapters=None):
    """
    Execute the complete book writing pipeline.

    Args:
        seed_text: the creative seed (premise, characters, world, outline...)
        num_chapters: explicit chapter count override (None = seed/config)
    """
    print("=" * 60)
    print("MULTI-AGENT BOOK WRITER")
    print("=" * 60)

    start_time = time.time()
    reset_context()

    if interim_enabled():
        clear_interim()  # drop artifacts from previous runs
        print(f"[PIPELINE] Interim artifacts will appear in {interim_dir()}/")

    try:
        # Step 1: Story bible from the seed prompt
        print("\n[PIPELINE] Step 1: Architect (story bible)")
        print("-" * 60)
        run_architect(seed_text)

        # Step 2: Chapter outline (honors the seed's own outline)
        print("\n[PIPELINE] Step 2: Planner (chapter outline)")
        print("-" * 60)
        chapters = run_planner(num_chapters=num_chapters)
        if not chapters:
            print("[PIPELINE] Planning failed. Exiting.")
            return 1

        # Step 3: Lore briefs per chapter
        if agent_enabled("researcher"):
            print("\n[PIPELINE] Step 3: Researcher (lore briefs)")
            print("-" * 60)
            run_researcher()

        # Step 4: Continuity-aware drafting
        print("\n[PIPELINE] Step 4: Writer (chapter drafts)")
        print("-" * 60)
        run_writer()

        # Step 5: Review, revise & polish (lint + reviewer + editor loop)
        if agent_enabled("editor"):
            print("\n[PIPELINE] Step 5: Review & Edit "
                  "(continuity, lint, revise, polish)")
            print("-" * 60)
            run_editor()
        else:
            print("\n[PIPELINE] Step 5: Editor disabled; saving drafts.")
            drafts = get_context("drafts")
            chapters = get_context("chapters")
            save_book([
                f"## Chapter {c['number']}: {c['title']}\n\n{drafts[c['number']]}"
                for c in chapters if drafts.get(c["number"])
            ])

        context = get_context()
        elapsed = time.time() - start_time
        print("\n" + "=" * 60)
        print("PIPELINE COMPLETE!")
        print("=" * 60)
        print(f"Title: {context['title']}")
        print(f"Chapters: {len(context['chapters'])}")
        print(f"Drafted: {len(context['drafts'])} | "
              f"Edited: {len(context['final'])}")
        print(f"Total time: {elapsed / 60:.1f} minutes")
        print("=" * 60)
        return 0

    except KeyboardInterrupt:
        print("\n[PIPELINE] Interrupted by user.")
        return 130
    except Exception as e:
        print(f"\n[PIPELINE] Error: {e}")
        raise


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
    parser.add_argument("--model", help="override the LLM model")
    parser.add_argument("--out", dest="out_file",
                        help="override the output filename "
                             "(written under the configured output dir)")
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

    # Make sure the LLM endpoint is up and the model exists before any work
    try:
        preflight()
    except (ConnectionError, RuntimeError) as e:
        sys.exit(f"Error: {e}")

    seed_text = resolve_seed_text(args)

    exit_code = run_pipeline(seed_text, num_chapters=num_chapters)
    if exit_code == 0:
        out = Path(cfg["output"]["directory"]) / cfg["output"]["filename"]
        print(f"\n\u2713 Book written! Check {out}")
    sys.exit(exit_code or 0)


if __name__ == "__main__":
    main()
