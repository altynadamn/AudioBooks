from __future__ import annotations

from audiobooks.models.script import SegmentType
from audiobooks.text.chunker import chunk_spans
from audiobooks.text.segmenter import segment_paragraphs, split_long, split_paragraph

D, N = SegmentType.DIALOGUE, SegmentType.NARRATION


def test_dash_dialogue_with_author_words() -> None:
    assert split_paragraph("— Ты всё-таки пришёл? — тихо спросила она.") == [
        ("Ты всё-таки пришёл?", D),
        ("тихо спросила она.", N),
    ]


def test_dash_dialogue_resumes_after_author_words() -> None:
    parts = split_paragraph("— Я обещал, — ответил он. — Разве я мог иначе?")
    assert parts == [("Я обещал,", D), ("ответил он.", N), ("Разве я мог иначе?", D)]


def test_dash_inside_sentence_is_not_a_switch() -> None:
    assert split_paragraph("— Москва — большой город.") == [("Москва — большой город.", D)]


def test_quotes() -> None:
    parts = split_paragraph('He looked up. "I will be there tomorrow," he said.')
    assert parts == [("He looked up.", N), ("I will be there tomorrow,", D), ("he said.", N)]


def test_short_quoted_title_is_narration() -> None:
    parts = split_paragraph("Он читал «Войну и мир» весь вечер.")
    assert parts == [("Он читал «Войну и мир» весь вечер.", N)]


def test_text_is_preserved() -> None:
    paragraph = "Обычный абзац повествования без диалога."
    assert split_paragraph(paragraph) == [(paragraph, N)]


def test_split_long_respects_sentences() -> None:
    text = "Первое предложение. Второе предложение. Третье предложение."
    pieces = split_long(text, 45)
    assert pieces == ["Первое предложение. Второе предложение.", "Третье предложение."]
    assert all(len(p) <= 45 for p in pieces)


def test_split_long_falls_back_to_words() -> None:
    text = "слово " * 50
    assert all(len(p) <= 30 for p in split_long(text.strip(), 30))


def test_segment_indices_and_paragraphs() -> None:
    spans = segment_paragraphs(["Нарратив.", "— Реплика, — сказал он."])
    assert [(s.index, s.paragraph, s.hint) for s in spans] == [(0, 0, N), (1, 1, D), (2, 1, N)]


def test_chunking_keeps_paragraphs_and_dialogue_runs() -> None:
    paragraphs = []
    for i in range(30):
        paragraphs.append(f"Повествование номер {i}. " * 5)
        paragraphs.append(f"— Реплика {i}, — сказал он.")
        paragraphs.append(f"— Ответ {i}, — ответила она.")
    spans = segment_paragraphs(paragraphs)
    chunks = chunk_spans(spans, paragraphs, max_chars=1000, context_paragraphs=2)
    assert len(chunks) > 1
    # every span appears exactly once, in order
    assert [s.index for c in chunks for s in c.spans] == [s.index for s in spans]
    for chunk in chunks:
        assert chunk.char_count <= 1000
        # a paragraph is never split between chunks
        paras = {s.paragraph for s in chunk.spans}
        assert all(s.paragraph in paras for s in spans if s.paragraph in paras)
        # chunks end after narration when possible, so no dialogue reply is orphaned
        last = chunk.spans[-1]
        if chunk is not chunks[-1]:
            assert paragraphs[last.paragraph].startswith(("Повествование", "— Ответ"))
    assert (
        chunks[1].context
        == paragraphs[chunks[1].spans[0].paragraph - 2 : chunks[1].spans[0].paragraph]
    )


def test_author_words_inside_quotes_are_narration() -> None:
    parts = split_paragraph(
        "«Все это вздор, – сказал он с надеждой, – и нечем тут было смущаться!»"
    )
    assert parts == [
        ("Все это вздор,", D),
        ("сказал он с надеждой,", N),
        ("и нечем тут было смущаться!", D),
    ]
