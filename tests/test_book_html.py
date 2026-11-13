"""Tests for the pure HTML-page builder (shared/book_html.py)."""

from html.parser import HTMLParser

from shared import book_html, epub


def _chapters():
    return [
        epub.Chapter(1, "The Fete", 'A ribbon was cut. "Mind the punt," '
                     "said Cade. *Very* carefully.\n\n"
                     "* * *\n\nThen the rain came."),
        epub.Chapter(2, "", "Silently, she counted."),
    ]


class _Balance(HTMLParser):
    """A generated page must at least be balanced, escaped html."""

    VOID = {"area", "base", "br", "col", "embed", "hr", "img", "input",
            "link", "meta", "source", "track", "wbr"}

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.stack = []

    def handle_starttag(self, tag, attrs):
        if tag not in self.VOID:
            self.stack.append(tag)

    def handle_endtag(self, tag):
        assert self.stack and self.stack[-1] == tag, \
            f"</{tag}> does not close <{'>'.join(self.stack[-2:])}>"
        self.stack.pop()


def test_the_page_is_complete_and_self_contained():
    html = book_html.build(_chapters(), title="The Marsh Light")
    assert "<!DOCTYPE html>" in html
    assert "prefers-color-scheme: dark" in html        # the dark twin theme
    assert "first-letter" in html                      # drop caps
    # nothing may be fetched from outside: the only script is the inline one
    assert "<script>" in html                          # the size controls
    assert "src=" not in html and "href=\"http" not in html
    assert "http://" not in html and "https://" not in html
    parser = _Balance()
    parser.feed(html)
    assert not parser.stack, f"unclosed tags {parser.stack}"


def test_the_text_is_justified_like_print():
    html = book_html.build(_chapters(), title="T")
    assert "text-align: justify" in html
    assert "hyphens: auto" in html                     # no rivers of white
    # the ornaments and the title stay out of the justification
    assert "text-align: center; hyphens: none" in html


def test_the_reader_resizes_the_text_live():
    html = book_html.build(_chapters(), title="T")
    assert 'data-size="+"' in html and 'data-size="&minus;"' in html
    assert "font-size: calc(100% * var(--scale, 1))" in html  # root-relative
    assert "localStorage" in html                      # the choice survives
    assert "clamp" in html                             # bounded: .7 - 1.4


def test_titles_and_text_are_escaped_and_marked_up():
    html = book_html.build(_chapters(), title='The "<Fete>" & Friends')
    assert "<title>The &quot;&lt;Fete&gt;&quot; &amp; Friends</title>" in html
    assert "<em>Very</em>" in html                     # markdown emphasis
    assert "A ribbon was cut" in html and "\u201c" in html   # curly quotes


def test_toc_links_match_chapters_and_count():
    import re

    html = book_html.build(_chapters(), title="T")
    anchors = [line for line in html.splitlines() if 'href="#c' in line]
    assert len(anchors) == 2
    assert 'id="c1"' in html and 'id="c2"' in html
    assert "Chapter One" in html and "Chapter Two" in html  # like the EPUB
    assert re.search(r"2 chapters &middot; [\d,]+ words", html)
    # a chapter without a title still gets its label as the heading
    assert '<span class="name"></span>' in html


def test_scene_breaks_and_first_paragraphs_are_styled():
    html = book_html.build(_chapters(), title="T")
    assert '<p class="scene-break">' in html
    assert '<p class="first">' in html


def test_a_titled_but_empty_chapter_renders_h2_text():
    html = book_html.build([epub.Chapter(1, "Only A Heading", "")], title="T")
    assert "Only A Heading" in html


def test_slug_matches_make_epub_s():
    assert book_html.slug("The Fete, and After!") == "the-fete-and-after"
    assert book_html.slug("???") == "book"
    assert book_html.slug("t") == "t"
