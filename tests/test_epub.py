"""The EPUB builder: typography, chapter parsing and the structure of the file."""

import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

import pytest

import make_epub
from shared import epub

NS = {"opf": "http://www.idpf.org/2007/opf",
      "dc": "http://purl.org/dc/elements/1.1/",
      "x": "http://www.w3.org/1999/xhtml",
      "ncx": "http://www.daisy.org/z3986/2005/ncx/",
      "c": "urn:oasis:names:tc:opendocument:xmlns:container"}


# ---------------------------------------------------------------- titles

@pytest.mark.parametrize("raw,clean", [
    ("The Fall (Stuart POV, heat 2)", "The Fall"),
    ("Week One (heat 5)", "Week One"),
    ("The Fete (day) (heat 3)", "The Fete"),
    ("The Morning After (heat 4 + coda)", "The Morning After"),
    ("Mary (Part One)", "Mary (Part One)"),          # a real subtitle stays
    ("The (Other) Side", "The (Other) Side"),
    ("  Spaced   out ", "Spaced out"),
    ("", ""),
])
def test_clean_title_drops_only_planning_notes(raw, clean):
    assert epub.clean_title(raw) == clean


def test_keep_annotations_leaves_the_title_alone():
    assert epub.clean_title("Week One (heat 5)", keep_annotations=True) == \
        "Week One (heat 5)"


def test_parse_chapter_reads_number_title_and_body():
    ch = epub.parse_chapter("# Chapter 4: Week One (heat 5)\n\nText here.\n")
    assert (ch.number, ch.title, ch.body) == (4, "Week One", "Text here.")
    ch = epub.parse_chapter("## Chapter 12 - The End\n\nBody")
    assert (ch.number, ch.title) == (12, "The End")


def test_parse_chapter_without_a_heading_uses_the_fallback_number():
    ch = epub.parse_chapter("Just prose.", fallback_number=7)
    assert (ch.number, ch.title, ch.body) == (7, "", "Just prose.")


# ------------------------------------------------------------ typography

@pytest.mark.parametrize("raw,smart", [
    ('"Hello," she said.', "“Hello,” she said."),
    ("He said \"it's fine\".", "He said “it’s fine”."),
    ("'Rough ride?' she asked.", "‘Rough ride?’ she asked."),
    ("Don't go.", "Don’t go."),
    ("It was 'tis the season, and 'em all.", "It was ’tis the season, and ’em all."),
    ("the '90s", "the ’90s"),
    ('*"Hello"*', "*“Hello”*"),                       # after an emphasis mark
    ("Wait... what -- really?", "Wait… what — really?"),
    ("“Already” curly", "“Already” curly"),
    ("(\"quoted\")", "(“quoted”)"),
])
def test_smarten(raw, smart):
    assert epub.smarten(raw) == smart


def test_inline_markup_and_escaping():
    html = epub.inline_html("*Will do after lunch.* and **bold** and _it_ & <b>")
    assert html == ("<em>Will do after lunch.</em> and <strong>bold</strong> "
                    "and <em>it</em> &amp; &lt;b&gt;")
    assert epub.inline_html("snake_case_name and 2*3*4") == \
        "snake_case_name and 2*3*4"


def test_paragraphs_scene_breaks_and_first_paragraph_class():
    html = epub.body_html("One,\nwrapped.\n\nTwo.\n\n* * *\n\nThree.\n\n---\n\nFour.")
    assert html.splitlines() == [
        '<p class="first">One, wrapped.</p>',
        "<p>Two.</p>",
        '<p class="scene-break">* * *</p>',
        '<p class="first">Three.</p>',
        '<p class="scene-break">* * *</p>',
        '<p class="first">Four.</p>']


@pytest.mark.parametrize("n,word", [(1, "One"), (13, "Thirteen"), (20, "Twenty"),
                                    (21, "Twenty-One"), (99, "Ninety-Nine"),
                                    (100, "100")])
def test_number_words(n, word):
    assert epub.number_word(n) == word


def test_cover_wrapping_is_balanced():
    assert epub._wrap("The Picnic, and After") == ["The Picnic,", "and After"]
    assert epub._wrap("Dune") == ["Dune"]
    assert epub._wrap("Supercalifragilisticexpialidocious Day") == [
        "Supercalifragilisticexpialidocious", "Day"]
    assert epub._wrap("") == [""]


# ------------------------------------------------------------- the file

def _chapters(n=3):
    return [epub.Chapter(i, f"Title {i}", f'"Hi," said {i}.\n\nSecond *para*.')
            for i in range(1, n + 1)]


def _build(tmp_path, chapters=None, title="A Book", **kw):
    out = tmp_path / "book.epub"
    epub.build_epub(chapters or _chapters(), title, out, **kw)
    return out


def _read(path):
    with zipfile.ZipFile(path) as zf:
        return zf, {n: zf.read(n) for n in zf.namelist()}


def test_mimetype_is_first_and_stored_uncompressed(tmp_path):
    zf, files = _read(_build(tmp_path))
    first = zf.infolist()[0]
    assert first.filename == "mimetype"
    assert first.compress_type == zipfile.ZIP_STORED
    assert files["mimetype"] == b"application/epub+zip"


def test_every_xml_part_is_well_formed(tmp_path):
    _, files = _read(_build(tmp_path, author="A & B <C>", title="Q&A: \"x\""))
    for name, data in files.items():
        if name.endswith((".xhtml", ".opf", ".ncx", ".xml", ".svg")):
            ET.fromstring(data)                       # raises if malformed


def test_manifest_spine_and_navigation_agree(tmp_path):
    _, files = _read(_build(tmp_path))
    container = ET.fromstring(files["META-INF/container.xml"])
    opf_path = container.find(".//c:rootfile", NS).get("full-path")
    opf = ET.fromstring(files[opf_path])
    manifest = {i.get("id"): i for i in opf.findall(".//opf:item", NS)}
    for item in manifest.values():
        assert f"OEBPS/{item.get('href')}" in files
    spine = [r.get("idref") for r in opf.findall(".//opf:itemref", NS)]
    assert set(spine) <= set(manifest)
    assert spine[:3] == ["cover", "titlepage", "nav"]
    assert spine[3:] == ["ch1", "ch2", "ch3"]

    nav = ET.fromstring(files["OEBPS/nav.xhtml"])
    links = [a.get("href") for a in nav.findall(
        ".//x:nav[@id='toc']//x:a", NS)]
    assert links == ["chapter_01.xhtml", "chapter_02.xhtml", "chapter_03.xhtml"]
    ncx = ET.fromstring(files["OEBPS/toc.ncx"])
    assert [c.get("src") for c in ncx.findall(".//ncx:content", NS)] == links
    assert manifest["nav"].get("properties") == "nav"
    assert manifest["cover-image"].get("properties") == "cover-image"


def test_metadata(tmp_path):
    _, files = _read(_build(tmp_path, title="My Book", author="Jo Writer",
                            language="en-US"))
    opf = ET.fromstring(files["OEBPS/content.opf"])
    assert opf.find(".//dc:title", NS).text == "My Book"
    assert opf.find(".//dc:creator", NS).text == "Jo Writer"
    assert opf.find(".//dc:language", NS).text == "en-US"
    assert opf.find(".//dc:identifier", NS).text.startswith("urn:uuid:")
    assert any(m.get("property") == "dcterms:modified"
               for m in opf.findall(".//opf:meta", NS))


def test_identifier_is_stable_for_the_same_book(tmp_path):
    a = _read(_build(tmp_path, title="Same", author="Me"))[1]["OEBPS/content.opf"]
    b = _read(_build(tmp_path, title="Same", author="Me"))[1]["OEBPS/content.opf"]
    ida = ET.fromstring(a).find(".//dc:identifier", NS).text
    idb = ET.fromstring(b).find(".//dc:identifier", NS).text
    assert ida == idb


def test_no_author_means_no_creator_element(tmp_path):
    _, files = _read(_build(tmp_path))
    opf = ET.fromstring(files["OEBPS/content.opf"])
    assert opf.find(".//dc:creator", NS) is None


def test_chapter_pages_have_smart_text_heading_and_labels(tmp_path):
    _, files = _read(_build(tmp_path))
    page = files["OEBPS/chapter_02.xhtml"].decode()
    assert "Chapter Two" in page and "<h1 class=\"chapter-title\">Title 2</h1>" in page
    assert "“Hi,” said 2." in page and "<em>para</em>" in page
    _, files = _read(_build(tmp_path, number_style="digits"))
    assert "Chapter 2" in files["OEBPS/chapter_02.xhtml"].decode()
    _, files = _read(_build(tmp_path, number_style="none"))
    assert "Chapter" not in files["OEBPS/chapter_02.xhtml"].decode()


def test_untitled_chapter_uses_its_label_as_the_heading(tmp_path):
    chapters = [epub.Chapter(1, "", "Prose.")]
    _, files = _read(_build(tmp_path, chapters=chapters))
    page = files["OEBPS/chapter_01.xhtml"].decode()
    assert '<h1 class="chapter-title">Chapter One</h1>' in page
    assert "chapter-number" not in page


def test_plain_quotes_leaves_text_alone(tmp_path):
    _, files = _read(_build(tmp_path, smart=False))
    page = files["OEBPS/chapter_01.xhtml"].decode()
    assert '"Hi," said 1.' in page and "“" not in page


def test_generated_cover_carries_title_and_author(tmp_path):
    _, files = _read(_build(tmp_path, title="Nine & Ten", author="Jo"))
    svg = files["OEBPS/images/cover.svg"].decode()
    assert "Nine &amp;" in svg and "JO" in svg


def test_custom_cover_image_is_used(tmp_path):
    img = tmp_path / "mine.PNG"
    img.write_bytes(b"\x89PNG fake")
    _, files = _read(_build(tmp_path, cover=img))
    assert files["OEBPS/images/cover.png"] == b"\x89PNG fake"
    assert b"image/png" in files["OEBPS/content.opf"]
    assert "images/cover.png" in files["OEBPS/cover.xhtml"].decode()


def test_unsupported_cover_type_is_refused(tmp_path):
    bad = tmp_path / "cover.bmp"
    bad.write_bytes(b"x")
    with pytest.raises(ValueError, match="cover image type"):
        _build(tmp_path, cover=bad)


def test_empty_book_is_refused(tmp_path):
    with pytest.raises(ValueError):
        epub.build_epub([], "T", tmp_path / "x.epub")
    assert not (tmp_path / "x.epub").exists()


# ------------------------------------------------- reading the pipeline's files

def _write_run(tmp_path):
    out = tmp_path / "out"
    (out / "chapters").mkdir(parents=True)
    for n in (10, 2, 1):
        (out / "chapters" / f"chapter_{n:02d}.md").write_text(
            f"# Chapter {n}: Name {n} (heat {n})\n\nBody {n}.\n", encoding="utf-8")
    (out / "chapters" / "notes.md").write_text("not a chapter")
    (out / "draft.md").write_text("# The Real Title\n\n## Chapter 1: x\n\ny\n",
                                  encoding="utf-8")
    return out


def test_directory_is_read_in_numeric_order_with_the_book_title(tmp_path):
    out = _write_run(tmp_path)
    title, chapters = epub.load_chapters(out / "chapters")
    assert title == "The Real Title"
    assert [(c.number, c.title) for c in chapters] == [
        (1, "Name 1"), (2, "Name 2"), (10, "Name 10")]


def test_title_falls_back_to_the_story_bible(tmp_path):
    out = _write_run(tmp_path)
    (out / "draft.md").unlink()
    (out / "story_bible.md").write_text("# Bible Title - Story Bible\n\n## x\n",
                                        encoding="utf-8")
    assert epub.load_chapters(out / "chapters")[0] == "Bible Title"


def test_assembled_book_file_is_split_into_chapters(tmp_path):
    book = tmp_path / "draft.md"
    book.write_text("# T\n\n## Chapter 1: A (heat 1)\n\nOne.\n\n"
                    "## Chapter 2: B\n\nTwo.\n", encoding="utf-8")
    title, chapters = epub.load_chapters(book)
    assert title == "T"
    assert [(c.number, c.title, c.body) for c in chapters] == [
        (1, "A", "One."), (2, "B", "Two.")]


def test_missing_source_is_reported(tmp_path):
    with pytest.raises(FileNotFoundError):
        epub.load_chapters(tmp_path / "nope")


# --------------------------------------------------------------------- CLI

def test_cli_builds_next_to_the_chapters_by_default(tmp_path, capsys):
    out = _write_run(tmp_path)
    assert make_epub.main([str(out / "chapters"), "--author", "Jo"]) == 0
    book = out / "the-real-title.epub"
    assert book.exists()
    assert "3 chapters" in capsys.readouterr().out
    assert zipfile.is_zipfile(book)


def test_cli_explicit_output_and_title(tmp_path):
    out = _write_run(tmp_path)
    target = tmp_path / "x" / "mine.epub"
    assert make_epub.main([str(out / "chapters"), "-o", str(target),
                           "--title", "Override"]) == 0
    opf = ET.fromstring(_read(target)[1]["OEBPS/content.opf"])
    assert opf.find(".//dc:title", NS).text == "Override"


def test_cli_default_source_comes_from_the_config(tmp_path):
    cfg = tmp_path / "c.yaml"
    cfg.write_text("output:\n  directory: books/mine\n")
    assert make_epub.default_source(cfg) == Path("books/mine/chapters")
    assert make_epub.default_source(tmp_path / "missing.yaml") == \
        Path("output/chapters")


def test_cli_reports_a_missing_or_empty_source(tmp_path, capsys):
    assert make_epub.main([str(tmp_path / "nope")]) == 1
    (tmp_path / "empty").mkdir()
    assert make_epub.main([str(tmp_path / "empty")]) == 1
    err = capsys.readouterr().err
    assert "no such" in err and "No chapter files" in err


def test_slug():
    assert make_epub.slug("The Picnic, and After!") == "the-picnic-and-after"
    assert make_epub.slug("???") == "book"
