"""Render a whole book as one self-contained, coloured HTML page (pure code).

The web GUI serves this live from a run's chapters/, so the link always shows
what exists right now - draft, edited, or in progress. No LLM, no JavaScript,
no external assets: chapter markdown is cleaned up with shared/epub.py's
typography (curly quotes, emphasis, scene-break ornaments) and wrapped in an
embedded stylesheet.

Reading ergonomics, on purpose:
  - a parchment / ink palette with an accent colour, plus a dark twin picked
    by the reader's own light/dark preference;
  - a title block with the running word count, a table of contents (collapsed
    behind a summary on phones, a fixed rail on wide screens) and drop caps on
    chapter openings;
  - justified, hyphenated paragraphs with first-line indents, like a printed
    book;
  - A-/A+ text-size buttons that resize the page live (root-relative rem) and
    remember the choice via localStorage - the only script on the page, kept
    inline so nothing is ever fetched.
"""

import re
from xml.sax.saxutils import escape

from shared import epub

_QUOTED = {'"': "&quot;", "'": "&apos;"}


def _esc(text):
    return escape(str(text), _QUOTED)

_CSS = """\
:root { color-scheme: light dark;
  --paper: #f6efe2; --card: #fdf9ec; --ink: #3a3128; --soft: #8c8272;
  --accent: #a2512e; --rule: #e4d8c2; }
@media (prefers-color-scheme: dark) { :root {
  --paper: #18150f; --card: #201d15; --ink: #d8d0bf; --soft: #948b7a;
  --accent: #d9a368; --rule: #312b20; } }
* { box-sizing: border-box }
html { scroll-behavior: smooth;
  font-size: calc(100% * var(--scale, 1)) }         /* A-/A+ resize, live */
body { margin: 0; background: var(--paper); color: var(--ink);
  font-family: Charter, "Iowan Old Style", "Palatino Linotype", Palatino,
               Georgia, "Times New Roman", serif;
  font-size: 1.13rem; line-height: 1.68;
  text-rendering: optimizeLegibility; }
.page { max-width: 38rem; margin: 0 auto; padding: 2.6rem 1.4rem 4rem }

header h1 { margin: 0 0 .5em; font-size: 2.35rem; line-height: 1.14 }
header .meta { margin: 0 0 2.2em; color: var(--soft); font-size: .8rem;
  letter-spacing: .16em; text-transform: uppercase }

aside { position: fixed; left: 1.6rem; top: 2.4rem; bottom: 1.6rem;
  width: 13.5rem; overflow-y: auto; scrollbar-width: thin; display: none;
  font-size: .82rem; line-height: 1.55 }
aside .heading, details summary { color: var(--accent); font-size: .78rem;
  letter-spacing: .2em; text-transform: uppercase }
aside .heading { margin: 0 0 .8em; color: var(--soft) }
details summary { cursor: pointer; padding: .3em .4em }
details.floating { border-top: 1px solid var(--rule);
  border-bottom: 1px solid var(--rule); margin-bottom: 2.6em }
details.floating a { display: block }
aside a { display: block }
details.floating a, aside a { padding: .3em .5em; border-radius: 6px;
  color: inherit; text-decoration: none; white-space: nowrap;
  overflow: hidden; text-overflow: ellipsis }
details.floating a:hover, :is(details, aside) a:hover { background: var(--card);
  color: var(--accent) }
.n { color: var(--soft); font-size: .74em; margin-right: .6em }

@media (min-width: 72rem) {
  .page { max-width: none; width: 38rem; margin-left: 17.5rem }
  aside { display: block }
  details.floating { display: none }
}

section.chapter { scroll-margin-top: 1.2rem }
section.chapter + section.chapter { margin-top: 3.4em; padding-top: 2.4em;
  border-top: 1px solid var(--rule) }
h2 { margin: 0 0 1.7em }
h2 .label { display: block; color: var(--accent); font-weight: 600;
  font-size: .74em; letter-spacing: .22em; text-transform: uppercase;
  margin-bottom: .4em }
h2 .name { display: block; font-size: 1.48em; font-weight: 600;
  line-height: 1.2 }
h2::after { content: "\\2766"; display: block; text-align: center;
  color: var(--accent); font-size: .95em; margin-top: 1.35em }

main p { margin: 0; text-indent: 1.7em; text-align: justify;
  hyphens: auto; -webkit-hyphens: auto; overflow-wrap: break-word }
p.first { text-indent: 0 }
p.first::first-letter { float: left; font-size: 3.1em; line-height: .86;
  padding: .02em .09em 0 0; color: var(--accent); font-weight: 700 }
p.scene-break { text-indent: 0; text-align: center; hyphens: none;
  color: var(--accent); letter-spacing: .7em; font-size: .78em;
  margin: 1.8em 0 1.6em }

.reader-controls { position: fixed; right: 1.1rem; bottom: 1.1rem;
  z-index: 5; display: flex; gap: .35rem }
.reader-controls button { font: inherit; font-size: .95rem; line-height: 1;
  min-width: 2.5em; padding: .55em .5em; border-radius: 8px; cursor: pointer;
  background: var(--card); color: var(--accent); border: 1px solid var(--rule);
  box-shadow: 0 1px 4px rgb(0 0 0 / .12) }
.reader-controls button:hover { color: var(--ink) }

footer { margin-top: 4.5em; color: var(--soft); font-size: .72rem;
  letter-spacing: .15em; text-transform: uppercase; text-align: center }

@media print {
  html, body { background: #fff; color: #000; font-size: 11.5pt }
  aside, details.floating, .reader-controls { display: none }
  .page { margin: 0; padding: 0 }
}
"""


def slug(title):
    """Same naming as make_epub.py, so the .html sits beside the .epub."""
    return re.sub(r"[^A-Za-z0-9]+", "-", str(title)).strip("-").lower() \
        or "book"


def _links(chapters, number_style):
    """One contents list; shared by the phone's collapsible TOC and the rail."""
    return "".join(
        f'<a href="#c{ch.number}"><span class="n">'
        f'{_esc(epub.chapter_label(ch.number, number_style) or "&sect;")}'
        f'</span>{_esc(ch.title)}</a>'
        for ch in chapters)


def build(chapters, title="Untitled", number_style="words"):
    """[epub.Chapter] + book title -> one complete HTML document."""
    sections = []
    for ch in chapters:
        number = epub.chapter_label(ch.number, number_style)
        label = (f'<span class="label">{_esc(number)}</span>'
                 if number else "")
        sections.append(
            f'<section class="chapter" id="c{ch.number}"><h2>{label}'
            f'<span class="name">{_esc(ch.title)}</span></h2>\n'
            f'{epub.body_html(ch.body)}</section>')
    height = (f"{len(chapters)} chapters &middot; "
              f"{epub.word_count(chapters):,} words")
    links = _links(chapters, number_style)
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{_esc(title)}</title>
<style>
{_CSS}</style>
</head>
<body>
<aside role="navigation" aria-label="Contents"
><p class="heading">Contents</p>{links}</aside>
<div class="page">
<header><h1>{_esc(title)}</h1>
<p class="meta">{height}</p></header>
<details class="floating"><summary>Contents</summary>{links}</details>
<main>
{chr(10).join(sections)}
</main>
<footer>rendered live from the chapters in progress &middot;
multi-agent book-writer</footer>
</div>
<div class="reader-controls">
<button type="button" data-size="&minus;" aria-label="Smaller text">A&minus;</button>
<button type="button" data-size="+" aria-label="Larger text">A+</button>
</div>
<script>
(function () {{
  var scale = 1;
  try {{ scale = parseFloat(localStorage.getItem('book-size')) || 1; }}
  catch (e) {{ }}
  var clamp = function (v) {{
    return Math.min(1.4, Math.max(.7, Math.round(v * 10) / 10)); }};
  var apply = function (value) {{
    scale = clamp(value);
    document.documentElement.style.setProperty('--scale', scale);
    try {{ localStorage.setItem('book-size', String(scale)); }} catch (e) {{ }} }};
  apply(scale);
  Array.prototype.forEach.call(
    document.querySelectorAll('button[data-size]'), function (button) {{
      button.addEventListener('click', function () {{
        apply(scale + (button.getAttribute('data-size') === '+' ? .1 : -.1)); }});
    }});
}})();
</script>
</body>
</html>
"""
