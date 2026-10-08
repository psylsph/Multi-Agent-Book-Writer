#!/usr/bin/env python3
"""Turn the finished chapters into a single EPUB.

    uv run python make_epub.py                      # output/chapters -> output/<title>.epub
    uv run python make_epub.py --author "A. Writer" --cover cover.jpg
    uv run python make_epub.py output/draft.md -o book.epub

Reads a directory of chapter_NN.md files (the default is the pipeline's
<output.directory>/chapters) or one assembled book file such as draft.md.
Nothing is sent anywhere and no LLM is used.
"""

import argparse
import re
import sys
from pathlib import Path

import yaml

from shared import epub


def default_source(config_path):
    """<output.directory>/chapters from the config file, else ./output/chapters."""
    directory = "output"
    try:
        cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8")) or {}
        directory = (cfg.get("output") or {}).get("directory") or directory
    except (OSError, yaml.YAMLError):
        pass
    return Path(directory) / "chapters"


def slug(title):
    return re.sub(r"[^A-Za-z0-9]+", "-", title).strip("-").lower() or "book"


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Build a single EPUB from the book's finished chapters.")
    p.add_argument("source", nargs="?",
                   help="a directory of chapter_NN.md files, or one book file "
                        "(default: <output.directory>/chapters from the config)")
    p.add_argument("-o", "--output", help="EPUB file to write "
                   "(default: <title>.epub next to the chapters directory)")
    p.add_argument("--title", help="book title (default: read from the "
                   "pipeline's files)")
    p.add_argument("--author", default="", help="author name for the title "
                   "page, cover and metadata")
    p.add_argument("--language", default="en-GB",
                   help="language code (default: en-GB)")
    p.add_argument("--cover", help="cover image (jpg, png, gif or svg); "
                   "default: a generated typographic cover")
    p.add_argument("--numbers", choices=("words", "digits", "none"),
                   default="words", help="chapter labels: 'Chapter One' "
                   "(default), 'Chapter 1', or none")
    p.add_argument("--keep-annotations", action="store_true",
                   help="keep planning notes such as '(heat 3)' in chapter titles")
    p.add_argument("--plain-quotes", action="store_true",
                   help="leave quotes, dashes and ellipses exactly as written")
    p.add_argument("--config", default="config.yaml",
                   help="config file used to find the output directory")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    source = Path(args.source) if args.source else default_source(args.config)
    try:
        book_title, chapters = epub.load_chapters(source, args.keep_annotations)
    except FileNotFoundError as e:
        print(f"[EPUB] {e}", file=sys.stderr)
        return 1
    if not chapters:
        print(f"[EPUB] No chapter files (chapter_NN.md) found in {source}",
              file=sys.stderr)
        return 1
    title = args.title or book_title or "Untitled"
    if title == "Untitled":
        print("[EPUB] No title found; using 'Untitled' (set one with --title).")
    if args.output:
        out = Path(args.output)
    else:
        out = source.resolve().parent / f"{slug(title)}.epub"
    try:
        epub.build_epub(chapters, title, out, author=args.author,
                        language=args.language, cover=args.cover,
                        number_style=args.numbers, smart=not args.plain_quotes)
    except (OSError, ValueError) as e:
        print(f"[EPUB] Could not build the book: {e}", file=sys.stderr)
        return 1
    print(f"[EPUB] {title}: {len(chapters)} chapters, "
          f"{epub.word_count(chapters):,} words -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
