from __future__ import annotations

import zipfile
from pathlib import Path

import pytest

from audiobooks.errors import UnsupportedFormatError
from audiobooks.parsers import get_parser
from audiobooks.parsers.base import is_heading, split_into_chapters
from audiobooks.parsers.txt import text_to_paragraphs

FB2 = """<?xml version="1.0" encoding="utf-8"?>
<FictionBook xmlns="http://www.gribuser.ru/xml/fictionbook/2.0">
  <description>
    <title-info>
      <author><first-name>Лев</first-name><last-name>Толстой</last-name></author>
      <book-title>Тестовая книга</book-title>
      <lang>ru</lang>
    </title-info>
  </description>
  <body>
    <title><p>Лев Толстой</p><p>Тестовая книга</p></title>
    <section>
      <title><p>Часть первая</p></title>
      <section>
        <title><p>Глава 1</p></title>
        <p>Анна подошла к окну.</p>
        <p>— Ты пришёл? — спросила она.</p>
        <empty-line/>
      </section>
      <section>
        <title><p>Глава 2</p></title>
        <p>Утро было тихим.</p>
        <poem><stanza><v>Строка стиха</v></stanza></poem>
      </section>
    </section>
  </body>
  <body name="notes">
    <section><p>Примечание, которое не нужно читать.</p></section>
  </body>
</FictionBook>
"""


def test_txt_utf8_with_headings(sample_txt: Path) -> None:
    book = get_parser(sample_txt).parse(sample_txt)
    assert book.language == "ru"
    assert [c.title for c in book.chapters] == ["Глава 1", "Глава 2"]
    assert all(c.has_title for c in book.chapters)
    assert book.chapters[0].paragraphs[0].startswith("Анна подошла")
    assert book.chapters[1].paragraphs[-1] == "— Пойдём, — сказал Иван."


def test_txt_cp1251_decoding(tmp_path: Path) -> None:
    path = tmp_path / "legacy.txt"
    path.write_bytes(("Глава 1\n\n" + "Старая кодировка текста. " * 20).encode("cp1251"))
    book = get_parser(path).parse(path)
    assert "Старая кодировка" in book.chapters[0].paragraphs[0]


def test_txt_without_headings_gets_generated_titles(tmp_path: Path) -> None:
    path = tmp_path / "plain.txt"
    path.write_text("Первый абзац.\n\nВторой абзац.\n", encoding="utf-8")
    book = get_parser(path).parse(path)
    assert len(book.chapters) == 1
    assert not book.chapters[0].has_title


def test_hard_wrapped_lines_are_joined_but_dialogue_lines_split() -> None:
    text = "Это длинная строка,\nперенесённая вручную.\n\n— Реплика один.\n— Реплика два.\n"
    assert text_to_paragraphs(text) == [
        "Это длинная строка, перенесённая вручную.",
        "— Реплика один.",
        "— Реплика два.",
    ]


def test_heading_detection() -> None:
    assert is_heading("Глава 12")
    assert is_heading("ГЛАВА ПЕРВАЯ")
    assert is_heading("Chapter IV. The Storm")
    assert is_heading("XII")
    assert is_heading("Пролог")
    assert not is_heading("Глава семьи сказала, что ужин готов, и все пошли к столу в большой зал.")
    assert not is_heading("— Глава? — переспросил он.")


def test_consecutive_headings_are_combined() -> None:
    chapters = split_into_chapters(["Часть 1", "Глава 1", "Текст главы."])
    assert chapters[0].title == "Часть 1. Глава 1"


def test_fb2_sections_metadata_and_notes(tmp_path: Path) -> None:
    path = tmp_path / "book.fb2"
    path.write_text(FB2, encoding="utf-8")
    book = get_parser(path).parse(path)
    assert book.title == "Тестовая книга"
    assert book.author == "Лев Толстой"
    assert book.language == "ru"
    assert [c.title for c in book.chapters] == ["Часть первая. Глава 1", "Часть первая. Глава 2"]
    assert book.chapters[0].paragraphs == ["Анна подошла к окну.", "— Ты пришёл? — спросила она."]
    assert "Строка стиха" in book.chapters[1].paragraphs
    assert not any("Примечание" in p for c in book.chapters for p in c.paragraphs)


def test_fb2_zip(tmp_path: Path) -> None:
    path = tmp_path / "book.fb2.zip"
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("book.fb2", FB2)
    book = get_parser(path).parse(path)
    assert len(book.chapters) == 2


def test_fb2_entities_are_not_expanded(tmp_path: Path) -> None:
    xxe = FB2.replace(
        '<?xml version="1.0" encoding="utf-8"?>',
        '<?xml version="1.0" encoding="utf-8"?>\n<!DOCTYPE x [<!ENTITY secret SYSTEM "file:///etc/hosts">]>',
    ).replace("Утро было тихим.", "Утро &secret; было тихим.")
    path = tmp_path / "xxe.fb2"
    path.write_text(xxe, encoding="utf-8")
    book = get_parser(path).parse(path)
    text = " ".join(p for c in book.chapters for p in c.paragraphs)
    assert "localhost" not in text


def test_epub(tmp_path: Path) -> None:
    from ebooklib import epub

    book = epub.EpubBook()
    book.set_identifier("test-id")
    book.set_title("EPUB Test")
    book.set_language("en")
    book.add_author("Jane Doe")
    chapters = []
    for i in (1, 2):
        ch = epub.EpubHtml(title=f"Chapter {i}", file_name=f"ch{i}.xhtml", lang="en")
        body = "".join(
            f"<p>Paragraph {j} of chapter {i} with <em>emphasis</em>.</p>" for j in range(20)
        )
        ch.content = f"<html><body><h1>Chapter {i}</h1>{body}</body></html>"
        book.add_item(ch)
        chapters.append(ch)
    book.toc = chapters
    book.spine = ["nav", *chapters]
    book.add_item(epub.EpubNcx())
    book.add_item(epub.EpubNav())
    path = tmp_path / "test.epub"
    epub.write_epub(str(path), book)

    parsed = get_parser(path).parse(path)
    assert parsed.title == "EPUB Test"
    assert parsed.author == "Jane Doe"
    titles = [c.title for c in parsed.chapters]
    assert titles[-2:] == ["Chapter 1", "Chapter 2"]
    assert parsed.chapters[-1].paragraphs[0] == "Paragraph 0 of chapter 2 with emphasis."


def test_pdf_text_layer(tmp_path: Path) -> None:
    import fitz

    path = tmp_path / "test.pdf"
    doc = fitz.open()
    for text in ("Chapter 1\n\nThe first page of the story.", "Chapter 2\n\nThe second page."):
        page = doc.new_page()
        page.insert_text((72, 72), text, fontsize=12)
    doc.save(path)
    doc.close()

    parsed = get_parser(path).parse(path)
    assert [c.title for c in parsed.chapters] == ["Chapter 1", "Chapter 2"]
    assert parsed.chapters[1].paragraphs == ["The second page."]


def test_unsupported_extension(tmp_path: Path) -> None:
    with pytest.raises(UnsupportedFormatError):
        get_parser(tmp_path / "book.docx")


CYRILLIC_FONTS = [
    Path("C:/Windows/Fonts/arial.ttf"),
    Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
    Path("/Library/Fonts/Arial Unicode.ttf"),
]


def _book_pdf(path: Path, font: Path) -> None:
    """3 pages with a running header/footer, page numbers, wrapped lines and dialogue."""
    import pymupdf

    doc = pymupdf.open()
    body = [
        [
            ("ЧАСТЬ 1", 14),
            ("1", 13),
            ("В начале июля молодой человек вышел из своей каморки и", 12),
            ("медленно, как бы в нерешимости, отправился к мосту. Кого-", 12),
            ("нибудь он встретить не хотел.", 12),
            ("– Ты пришёл? – спросила она.", 12),
        ],
        [
            ("– Пришёл, – ответил он и сел у окна, глядя на падающий снег за стек-", 12),
            ("лом.", 12),
            ("2", 13),
            ("Утром снег перестал, и город стал тихим.", 12),
        ],
        [("ЧАСТЬ 2", 14), ("1", 13), ("Прошла неделя, и всё повторилось снова.", 12)],
    ]
    for n, lines in enumerate(body, start=1):
        page = doc.new_page()
        kw = {"fontname": "cyr", "fontfile": str(font)}
        page.insert_text((65, 40), "Автор: «Книга»", fontsize=10, **kw)
        page.insert_text((540, 40), str(n), fontsize=10, **kw)
        page.insert_text((65, 800), "100 лучших книг: www.example.ru", fontsize=10, **kw)
        y = 100
        for text, size in lines:
            page.insert_text((65, y), text, fontsize=size, **kw)
            y += size + 8
    doc.save(path)
    doc.close()


def test_pdf_rebuilds_paragraphs_and_strips_running_headers(tmp_path: Path) -> None:
    font = next((f for f in CYRILLIC_FONTS if f.is_file()), None)
    if font is None:
        pytest.skip("no Cyrillic TTF font available to build the test PDF")
    path = tmp_path / "book.pdf"
    _book_pdf(path, font)
    book = get_parser(path).parse(path)
    titles = [c.title for c in book.chapters]
    assert titles == ["ЧАСТЬ 1. Глава 1", "ЧАСТЬ 1. Глава 2", "ЧАСТЬ 2. Глава 1"]
    text = [p for c in book.chapters for p in c.paragraphs]
    assert not any("example.ru" in p or "«Книга»" in p for p in text)
    first = book.chapters[0].paragraphs
    # wrapped lines joined, line-end hyphen of "Кого-нибудь" kept
    assert first[0].startswith("В начале июля молодой человек вышел из своей каморки и медленно")
    assert "Кого-нибудь он встретить не хотел." in first[0]
    # each dialogue line is its own paragraph; a word split across pages is re-joined
    assert first[1] == "– Ты пришёл? – спросила она."
    assert first[2].endswith("падающий снег за стеклом.")


def test_outdated_parse_is_refreshed(ctx, sample_txt: Path) -> None:
    from audiobooks.services.books import BookService
    from audiobooks.utils import read_json, write_json

    service = BookService(ctx)
    book = service.import_book(sample_txt)
    cached = ctx.layout(book.id).root / "book.json"
    stale = read_json(cached)
    stale["parser_version"] = 0
    stale["chapters"] = stale["chapters"][:1]
    write_json(cached, stale)
    parsed = service.load_parsed(book.id)
    assert len(parsed.chapters) == 2
    assert service.import_book(sample_txt).id == book.id
