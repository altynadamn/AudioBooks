"""Group chapter spans into LLM-sized chunks without cutting dialogue exchanges."""

from __future__ import annotations

from dataclasses import dataclass, field

from audiobooks.models.script import SegmentType
from audiobooks.text.segmenter import Span


@dataclass(frozen=True)
class Chunk:
    index: int  # 1-based within the chapter
    spans: list[Span]
    context: list[str] = field(default_factory=list)  # preceding paragraphs, read-only

    @property
    def char_count(self) -> int:
        return sum(len(s.text) for s in self.spans)

    @property
    def span_ids(self) -> list[int]:
        return [s.index for s in self.spans]


def _paragraph_groups(spans: list[Span]) -> list[list[Span]]:
    groups: list[list[Span]] = []
    for span in spans:
        if groups and groups[-1][0].paragraph == span.paragraph:
            groups[-1].append(span)
        else:
            groups.append([span])
    return groups


def _is_narration_only(group: list[Span]) -> bool:
    return all(s.hint is SegmentType.NARRATION for s in group)


def chunk_spans(
    spans: list[Span],
    paragraphs: list[str],
    *,
    max_chars: int = 4000,
    context_paragraphs: int = 3,
) -> list[Chunk]:
    """Pack paragraphs into chunks of at most ~``max_chars``.

    Paragraphs are never split across chunks. Once a chunk is 70% full, the next
    narration-only paragraph ends it, so a back-and-forth dialogue stays together; a
    chunk is only cut inside a dialogue run when it would exceed ``max_chars``.
    Each chunk carries the previous paragraphs as context for speaker attribution.
    """
    soft_limit = int(max_chars * 0.7)
    groups = _paragraph_groups(spans)
    chunks: list[list[list[Span]]] = []
    current: list[list[Span]] = []
    size = 0
    for group in groups:
        group_size = sum(len(s.text) for s in group)
        if current and size + group_size > max_chars:
            chunks.append(current)
            current, size = [], 0
        current.append(group)
        size += group_size
        if size >= soft_limit and _is_narration_only(group):
            chunks.append(current)
            current, size = [], 0
    if current:
        chunks.append(current)

    result = []
    for i, groups_in_chunk in enumerate(chunks, start=1):
        first_paragraph = groups_in_chunk[0][0].paragraph
        ctx_start = max(0, first_paragraph - context_paragraphs)
        result.append(
            Chunk(
                index=i,
                spans=[s for g in groups_in_chunk for s in g],
                context=paragraphs[ctx_start:first_paragraph],
            )
        )
    return result
