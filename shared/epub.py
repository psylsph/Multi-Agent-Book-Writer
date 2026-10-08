"""Turn finished chapter files into a single EPUB 3 book (standard library only).

Reads the chapters the pipeline leaves in <output>/chapters/ (or one assembled
book file), tidies the typography, and writes a reflowable EPUB with a cover,
title page, contents page, a navigable table of contents and a stylesheet.
make_epub.py is the command-line front end.

What it does to the text, deliberately and visibly:
  - straight quotes become curly quotes, "..." an ellipsis, "--" an em dash;
  - markdown emphasis (*italic*, **bold**, _italic_) becomes real markup;
  - blank-line separated paragraphs become <p>; a line of "***", "---" or
    "* * *" becomes a scene-break ornament;
  - planning notes the pipeline leaves in chapter headings, such as
    "(Stuart POV, heat 3)", are dropped from the displayed titles.
"""

import hashlib
import re
import uuid
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from xml.sax.saxutils import escape, quoteattr

# --------------------------------------------------------------- chapters

_HEADING_RE = re.compile(
    r"^#{1,2}[ \t]*Chapter[ \t]+(\d+)[ \t]*[:.\-–—]*[ \t]*(.*?)[ \t]*$",
    re.IGNORECASE)
_FILE_RE = re.compile(r"chapter[_\- ]?(\d+)", re.IGNORECASE)
_TRAILING_PAREN_RE = re.compile(r"\s*\(([^()]*)\)\s*$")
# The note is a planning annotation if it mentions one of these, or is just
# a time of day ("(day)").
_ANNOTATION_RE = re.compile(r"\b(?:heat|pov|coda)\b|^\s*(?:day|night)\s*$",
                            re.IGNORECASE)
_SCENE_BREAK_RE = re.compile(r"^(?:(?:\*\s*){3,}|(?:-\s*){3,}|(?:_\s*){3,}|#|~{3,})$")


@dataclass
class Chapter:
    number: int
    title: str          # display title, annotations already removed
    body: str           # markdown text without the heading line


def clean_title(title, keep_annotations=False):
    """Drop trailing planning notes: 'The Fete (day) (heat 3)' -> 'The Fete'."""
    title = " ".join(str(title or "").split())
    if keep_annotations:
        return title
    while True:
        m = _TRAILING_PAREN_RE.search(title)
        if not m or not _ANNOTATION_RE.search(m.group(1)):
            return title
        title = title[:m.start()].rstrip()


def parse_chapter(text, fallback_number=0, keep_annotations=False):
    """One chapter file -> Chapter. A leading '# Chapter N: Title' line gives
    the number and title; without one, the file's own number is used."""
    lines = text.lstrip("\ufeff").splitlines()
    while lines and not lines[0].strip():
        lines.pop(0)
    number, title = fallback_number, ""
    if lines:
        m = _HEADING_RE.match(lines[0].strip())
        if m:
            number, title = int(m.group(1)), m.group(2)
            lines.pop(0)
    return Chapter(number, clean_title(title, keep_annotations),
                   "\n".join(lines).strip())


def _split_book(text, keep_annotations):
    """An assembled book file -> (book title or '', [Chapter])."""
    book_title, chunks, current = "", [], None
    for line in text.lstrip("\ufeff").splitlines():
        if _HEADING_RE.match(line.strip()):
            current = [line]
            chunks.append(current)
        elif current is not None:
            current.append(line)
        elif not book_title and re.match(r"^#\s+\S", line):
            book_title = line.lstrip("# ").strip()
    return book_title, [parse_chapter("\n".join(c), 0, keep_annotations)
                        for c in chunks]


def load_chapters(source, keep_annotations=False):
    """Read chapters from a directory of chapter files or one book file.
    Returns (book_title_or_empty, [Chapter] in numeric order)."""
    source = Path(source)
    if source.is_dir():
        files = []
        for path in source.iterdir():
            m = _FILE_RE.search(path.stem)
            if path.suffix.lower() in (".md", ".txt") and m:
                files.append((int(m.group(1)), path))
        chapters = [parse_chapter(p.read_text(encoding="utf-8"), n,
                                  keep_annotations)
                    for n, p in sorted(files)]
        return find_book_title(source), chapters
    if source.is_file():
        title, chapters = _split_book(source.read_text(encoding="utf-8"),
                                      keep_annotations)
        if not chapters:           # a single unheaded text: one chapter
            chapters = [parse_chapter(source.read_text(encoding="utf-8"), 1,
                                      keep_annotations)]
        return title, chapters
    raise FileNotFoundError(f"no such chapter directory or file: {source}")


def find_book_title(chapters_dir):
    """The book's title, from the pipeline's files next to the chapters:
    the assembled draft's first line, else the saved story bible."""
    out = Path(chapters_dir).resolve().parent
    for name in sorted(out.glob("*.md")):
        if name.name == "story_bible.md":
            continue
        try:
            first = name.read_text(encoding="utf-8").lstrip("\ufeff").split(
                "\n", 1)[0]
        except OSError:
            continue
        if re.match(r"^#\s+(?!Chapter\b)\S", first):
            return first.lstrip("# ").strip()
    bible = out / "story_bible.md"
    if bible.exists():
        first = bible.read_text(encoding="utf-8").split("\n", 1)[0]
        m = re.match(r"^#\s+(.*?)\s+-\s+Story Bible\s*$", first)
        if m:
            return m.group(1)
    return ""


# -------------------------------------------------------------- typography

_OPENERS = " \t\n([{—–‘“-/"
_ELISIONS = ("tis", "twas", "til", "cause", "em", "n", "round", "bout",
             "cept", "neath", "twere", "twill")


def smarten(text):
    """Curly quotes, ellipses and em dashes. Existing curly quotes are kept."""
    text = text.replace("...", "…").replace("--", "—")
    out = []
    for i, ch in enumerate(text):
        if ch not in "\"'":
            out.append(ch)
            continue
        j = i - 1
        while j >= 0 and text[j] in "*_":          # emphasis marks are transparent
            j -= 1
        before = text[j] if j >= 0 else " "
        opening = before in _OPENERS
        if ch == '"':
            out.append("“" if opening else "”")
            continue
        after = text[i + 1:i + 8].lower()
        if opening and (after[:1].isdigit() or any(
                after.startswith(w) and not after[len(w):len(w) + 1].isalpha()
                for w in _ELISIONS)):
            out.append("’")               # 'tis, 'em, the '90s
        else:
            out.append("‘" if opening else "’")
    return "".join(out)


_STRONG_RE = re.compile(r"\*\*(?=\S)(.+?)(?<=\S)\*\*")
_EM_RE = re.compile(r"(?<![\w*])\*(?=[^\s*])(.+?)(?<=[^\s*])\*(?![\w*])")
_EM_US_RE = re.compile(r"(?<![\w_])_(?=[^\s_])(.+?)(?<=[^\s_])_(?![\w_])")


def inline_html(text, smart=True):
    """One paragraph of markdown text -> escaped XHTML with emphasis markup."""
    if smart:
        text = smarten(text)
    text = escape(text)
    text = _STRONG_RE.sub(r"<strong>\1</strong>", text)
    text = _EM_RE.sub(r"<em>\1</em>", text)
    text = _EM_US_RE.sub(r"<em>\1</em>", text)
    return text


def body_html(body, smart=True):
    """Chapter markdown -> XHTML paragraphs. The first paragraph of the
    chapter and the first after a scene break are marked class="first"
    (no indent); scene-break lines become an ornament."""
    html, fresh = [], True
    for block in re.split(r"\n\s*\n", body.strip()):
        block = " ".join(line.strip() for line in block.splitlines()).strip()
        if not block:
            continue
        if _SCENE_BREAK_RE.match(block):
            html.append('<p class="scene-break">* * *</p>')
            fresh = True
            continue
        css = ' class="first"' if fresh else ""
        html.append(f"<p{css}>{inline_html(block, smart)}</p>")
        fresh = False
    return "\n".join(html)


_ONES = ["", "One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight",
         "Nine", "Ten", "Eleven", "Twelve", "Thirteen", "Fourteen", "Fifteen",
         "Sixteen", "Seventeen", "Eighteen", "Nineteen"]
_TENS = ["", "", "Twenty", "Thirty", "Forty", "Fifty", "Sixty", "Seventy",
         "Eighty", "Ninety"]


def number_word(n):
    """1 -> 'One', 21 -> 'Twenty-One'; beyond 99 the digits."""
    if 0 < n < 20:
        return _ONES[n]
    if 20 <= n < 100:
        return _TENS[n // 10] + (f"-{_ONES[n % 10]}" if n % 10 else "")
    return str(n)


def chapter_label(number, style="words"):
    if style == "none":
        return ""
    return f"Chapter {number_word(number) if style == 'words' else number}"


# ------------------------------------------------------------------- cover

_PALETTES = [
    ("#14213d", "#f1ede4", "#c9a45c"),      # navy and gold
    ("#2b1d2e", "#f3ece6", "#d08c60"),      # aubergine and copper
    ("#1f3a33", "#f1efe6", "#c8b273"),      # forest and brass
    ("#3a1f24", "#f4ebe3", "#cf9b7a"),      # claret and rose
    ("#222831", "#eeeeee", "#9fb4c7"),      # slate and ice
    ("#f2ebdd", "#2a2420", "#9a5b3c"),      # parchment and rust
]


def _wrap(title, width=16):
    """Split a title into the fewest lines of about `width` characters, as
    evenly balanced as possible (so "The Picnic, and After" becomes "The
    Picnic," / "and After", not "The Picnic, and" / "After")."""
    words = title.split()
    if not words:
        return [""]
    n = len(words)
    lines_needed = min(n, max(1, -(-len(title) // width)))
    # best[k][i]: least cost of putting the first i words on k lines, where a
    # line's cost is the square of its slack against the longest line allowed
    limit = max(width, max(len(w) for w in words))
    inf = float("inf")
    best = [[inf] * (n + 1) for _ in range(n + 2)]
    cut = [[0] * (n + 1) for _ in range(n + 2)]
    best[0][0] = 0
    for k in range(1, n + 1):
        for i in range(1, n + 1):
            for j in range(i):
                text = " ".join(words[j:i])
                if len(text) > limit * 1.25 or best[k - 1][j] == inf:
                    continue
                cost = best[k - 1][j] + (limit - len(text)) ** 2
                if cost < best[k][i]:
                    best[k][i], cut[k][i] = cost, j
    ks = [k for k in range(lines_needed, n + 1) if best[k][n] < inf]
    if not ks:
        return [title]
    k, i, out = ks[0], n, []
    while k:
        j = cut[k][i]
        out.append(" ".join(words[j:i]))
        k, i = k - 1, j
    return out[::-1]


def cover_svg(title, author=""):
    """A typographic cover (SVG, 1600x2400). The colours come from the title,
    so a given book always looks the same."""
    pick = int(hashlib.sha256(title.encode("utf-8")).hexdigest(), 16)
    bg, fg, accent = _PALETTES[pick % len(_PALETTES)]
    lines = _wrap(title)
    longest = max(len(x) for x in lines)
    size = int(max(90, min(210, 1250 / (0.56 * max(longest, 6)))))
    lead = int(size * 1.22)
    top = 900 - (len(lines) - 1) * lead // 2
    tspans = "".join(
        f'<tspan x="800" y="{top + i * lead}">{escape(x)}</tspan>'
        for i, x in enumerate(lines))
    byline = (
        f'<text x="800" y="2120" text-anchor="middle" font-size="64" '
        f'letter-spacing="14" fill="{fg}">{escape(author.upper())}</text>'
        if author else "")
    rule_y = top + (len(lines) - 1) * lead + int(size * 0.62)
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1600 2400" width="1600" height="2400">
<rect width="1600" height="2400" fill="{bg}"/>
<rect x="90" y="90" width="1420" height="2220" fill="none" stroke="{accent}" stroke-width="6"/>
<rect x="120" y="120" width="1360" height="2160" fill="none" stroke="{accent}" stroke-width="2"/>
<g font-family="Georgia, 'Times New Roman', serif" fill="{fg}">
<text text-anchor="middle" font-size="{size}">{tspans}</text>
<path d="M 560 {rule_y} H 760 M 840 {rule_y} H 1040" stroke="{accent}" stroke-width="4"/>
<path d="M 800 {rule_y - 22} l 22 22 l -22 22 l -22 -22 z" fill="{accent}"/>
{byline}
</g>
</svg>
"""


# ------------------------------------------------------------------ styles

STYLESHEET = """\
@charset "utf-8";
html { font-size: 100%; }
body {
  font-family: Georgia, "Iowan Old Style", "Palatino Linotype", Palatino, "Book Antiqua", serif;
  line-height: 1.5;
  margin: 0;
  padding: 0 0.2em;
  text-align: justify;
  -webkit-hyphens: auto; -epub-hyphens: auto; hyphens: auto;
  widows: 2; orphans: 2;
}
p { margin: 0; text-indent: 1.4em; }
p.first { text-indent: 0; }
p.first::first-line { font-variant: small-caps; letter-spacing: 0.04em; }
p.scene-break {
  text-indent: 0; text-align: center; margin: 1.4em 0; letter-spacing: 0.6em;
  page-break-after: avoid; break-after: avoid;
}
header.chapter { margin: 18% 0 2.4em; text-align: center; }
p.chapter-number {
  text-indent: 0; text-align: center; font-size: 0.8em; margin: 0 0 0.7em;
  letter-spacing: 0.28em; text-transform: uppercase; color: #7a7a7a;
}
h1.chapter-title {
  font-size: 1.7em; font-weight: normal; line-height: 1.2; margin: 0;
  text-align: center; -webkit-hyphens: none; -epub-hyphens: none; hyphens: none;
  page-break-after: avoid; break-after: avoid;
}
header.chapter hr {
  width: 3em; border: 0; border-top: 1px solid #999; margin: 1.3em auto 0;
}
section.chapter { page-break-before: always; break-before: page; }
section.titlepage { text-align: center; margin-top: 28%; }
section.titlepage h1 {
  font-size: 2.3em; font-weight: normal; line-height: 1.2; margin: 0 0 0.6em;
  text-align: center; -webkit-hyphens: none; -epub-hyphens: none; hyphens: none;
}
section.titlepage p {
  text-indent: 0; text-align: center; letter-spacing: 0.2em;
  text-transform: uppercase; font-size: 0.95em; margin-top: 1.6em;
}
section.titlepage hr { width: 4em; border: 0; border-top: 1px solid #999; margin: 0 auto; }
nav h1 { font-size: 1.5em; font-weight: normal; text-align: center; margin: 14% 0 1.4em; }
nav ol { list-style: none; margin: 0; padding: 0; }
nav li { margin: 0.55em 0; text-align: left; text-indent: 0; }
nav a { color: inherit; text-decoration: none; }
section.cover, section.cover div { margin: 0; padding: 0; text-align: center; }
section.cover img { max-width: 100%; max-height: 100%; height: auto; }
"""

_NS = ('xmlns="http://www.w3.org/1999/xhtml" '
       'xmlns:epub="http://www.idpf.org/2007/ops"')


def _page(title, body, language, body_attrs=""):
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE html>
<html {_NS} lang="{language}" xml:lang="{language}">
<head>
<meta charset="utf-8"/>
<title>{escape(title)}</title>
<link rel="stylesheet" type="text/css" href="style.css"/>
</head>
<body{body_attrs}>
{body}
</body>
</html>
"""


# ------------------------------------------------------------------ builder

_IMAGE_TYPES = {".jpg": "image/jpeg", ".jpeg": "image/jpeg",
                ".png": "image/png", ".gif": "image/gif",
                ".svg": "image/svg+xml"}


def word_count(chapters):
    return sum(len(c.body.split()) for c in chapters)


def build_epub(chapters, title, out_path, author="", language="en-GB",
               cover=None, number_style="words", smart=True, identifier=None):
    """Write the EPUB. `cover` is an image path, or None for a generated
    typographic cover. Returns the output path."""
    if not chapters:
        raise ValueError("no chapters to put in the book")
    title = " ".join(str(title or "").split()) or "Untitled"
    author = " ".join(str(author or "").split())
    out_path = Path(out_path)

    if cover:
        cover = Path(cover)
        media = _IMAGE_TYPES.get(cover.suffix.lower())
        if not media:
            raise ValueError(f"unsupported cover image type: {cover.suffix}")
        cover_name, cover_bytes = f"cover{cover.suffix.lower()}", cover.read_bytes()
    else:
        media, cover_name = "image/svg+xml", "cover.svg"
        cover_bytes = cover_svg(title, author).encode("utf-8")

    book_id = identifier or "urn:uuid:" + str(
        uuid.uuid5(uuid.NAMESPACE_URL, f"book-writer:{title}:{author}"))

    pages = {}      # file name -> (xhtml text, nav label or None)
    pages["cover.xhtml"] = (_page(
        "Cover",
        '<section epub:type="cover" class="cover"><div>'
        f'<img src="images/{cover_name}" alt={quoteattr("Cover of " + title)}/>'
        '</div></section>', language), None)
    pages["titlepage.xhtml"] = (_page(
        title,
        f'<section epub:type="titlepage" class="titlepage"><h1>{escape(title)}</h1>'
        '<hr/>' + (f"<p>{escape(author)}</p>" if author else "")
        + "</section>", language), None)

    nav_items, chapter_files = [], []
    for ch in chapters:
        name = f"chapter_{ch.number:02d}.xhtml" if ch.number > 0 else (
            f"chapter_x{len(chapter_files) + 1:02d}.xhtml")
        label = chapter_label(ch.number, number_style)
        head = '<header class="chapter">'
        if label and ch.title:
            head += f'<p class="chapter-number">{escape(label)}</p>'
        head += f'<h1 class="chapter-title">{escape(ch.title or label)}</h1>'
        head += "<hr/></header>"
        nav_label = (f"{label}: {ch.title}" if label and ch.title
                     else (ch.title or label or f"Part {len(chapter_files) + 1}"))
        pages[name] = (_page(
            nav_label,
            f'<section epub:type="chapter" class="chapter">{head}\n'
            f"{body_html(ch.body, smart)}\n</section>", language), nav_label)
        nav_items.append((name, nav_label))
        chapter_files.append(name)

    toc_li = "\n".join(f'<li><a href="{n}">{escape(label)}</a></li>'
                       for n, label in nav_items)
    nav = f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE html>
<html {_NS} lang="{language}" xml:lang="{language}">
<head>
<meta charset="utf-8"/>
<title>Contents</title>
<link rel="stylesheet" type="text/css" href="style.css"/>
</head>
<body>
<nav epub:type="toc" id="toc">
<h1>Contents</h1>
<ol>
{toc_li}
</ol>
</nav>
<nav epub:type="landmarks" hidden="hidden">
<ol>
<li><a epub:type="cover" href="cover.xhtml">Cover</a></li>
<li><a epub:type="titlepage" href="titlepage.xhtml">Title page</a></li>
<li><a epub:type="toc" href="nav.xhtml">Contents</a></li>
<li><a epub:type="bodymatter" href="{chapter_files[0]}">Start of the book</a></li>
</ol>
</nav>
</body>
</html>
"""

    points = "\n".join(
        f'<navPoint id="np{i}" playOrder="{i}"><navLabel><text>{escape(label)}'
        f'</text></navLabel><content src="{n}"/></navPoint>'
        for i, (n, label) in enumerate(nav_items, 1))
    ncx = f"""<?xml version="1.0" encoding="UTF-8"?>
<ncx xmlns="http://www.daisy.org/z3986/2005/ncx/" version="2005-1">
<head>
<meta name="dtb:uid" content={quoteattr(book_id)}/>
<meta name="dtb:depth" content="1"/>
<meta name="dtb:totalPageCount" content="0"/>
<meta name="dtb:maxPageNumber" content="0"/>
</head>
<docTitle><text>{escape(title)}</text></docTitle>
<navMap>
{points}
</navMap>
</ncx>
"""

    manifest = [
        '<item id="nav" href="nav.xhtml" media-type="application/xhtml+xml" properties="nav"/>',
        '<item id="ncx" href="toc.ncx" media-type="application/x-dtbncx+xml"/>',
        '<item id="css" href="style.css" media-type="text/css"/>',
        f'<item id="cover-image" href="images/{cover_name}" media-type="{media}" '
        'properties="cover-image"/>',
        '<item id="cover" href="cover.xhtml" media-type="application/xhtml+xml"/>',
        '<item id="titlepage" href="titlepage.xhtml" media-type="application/xhtml+xml"/>',
    ]
    spine = ['<itemref idref="cover"/>', '<itemref idref="titlepage"/>',
             '<itemref idref="nav"/>']
    for i, name in enumerate(chapter_files, 1):
        manifest.append(f'<item id="ch{i}" href="{name}" '
                        'media-type="application/xhtml+xml"/>')
        spine.append(f'<itemref idref="ch{i}"/>')
    modified = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    opf = f"""<?xml version="1.0" encoding="UTF-8"?>
<package xmlns="http://www.idpf.org/2007/opf" version="3.0" unique-identifier="book-id" xml:lang="{language}">
<metadata xmlns:dc="http://purl.org/dc/elements/1.1/">
<dc:identifier id="book-id">{escape(book_id)}</dc:identifier>
<dc:title>{escape(title)}</dc:title>
{f"<dc:creator>{escape(author)}</dc:creator>" if author else ""}
<dc:language>{language}</dc:language>
<meta property="dcterms:modified">{modified}</meta>
<meta name="cover" content="cover-image"/>
</metadata>
<manifest>
{chr(10).join(manifest)}
</manifest>
<spine toc="ncx">
{chr(10).join(spine)}
</spine>
<guide>
<reference type="cover" title="Cover" href="cover.xhtml"/>
<reference type="text" title="Start of the book" href="{chapter_files[0]}"/>
</guide>
</package>
"""

    container = """<?xml version="1.0" encoding="UTF-8"?>
<container version="1.0" xmlns="urn:oasis:names:tc:opendocument:xmlns:container">
<rootfiles>
<rootfile full-path="OEBPS/content.opf" media-type="application/oebps-package+xml"/>
</rootfiles>
</container>
"""

    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_name(out_path.name + ".tmp")
    stamp = (1980, 1, 1, 0, 0, 0)

    def add(zf, name, data, compress=True):
        info = zipfile.ZipInfo(name, stamp)
        info.compress_type = zipfile.ZIP_DEFLATED if compress else zipfile.ZIP_STORED
        info.external_attr = 0o644 << 16
        zf.writestr(info, data.encode("utf-8") if isinstance(data, str) else data)

    with zipfile.ZipFile(tmp, "w") as zf:
        add(zf, "mimetype", "application/epub+zip", compress=False)
        add(zf, "META-INF/container.xml", container)
        add(zf, "OEBPS/content.opf", opf)
        add(zf, "OEBPS/nav.xhtml", nav)
        add(zf, "OEBPS/toc.ncx", ncx)
        add(zf, "OEBPS/style.css", STYLESHEET)
        add(zf, f"OEBPS/images/{cover_name}", cover_bytes)
        for name, (text, _) in pages.items():
            add(zf, f"OEBPS/{name}", text)
    tmp.replace(out_path)
    return out_path
