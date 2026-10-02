"""Prompt templates for structured book analysis. All prompt text lives here."""

from __future__ import annotations

from audiobooks.llm.schemas import ChunkRequest

PROMPT_VERSION = "chunk-v2"  # bump to invalidate cached chunk analyses

SYSTEM_PROMPT = """\
You are an expert audiobook director preparing a book for a multi-voice ("AI cast") \
recording. You receive a passage that is already split into numbered spans. Your job is to \
LABEL every span; you never rewrite, translate, merge or split the text.

INPUT FORMAT: one paragraph per line. Inside a paragraph every span is written as \
[id:hint] followed by its text. The hint ("dialogue"/"narration") is a typographic guess \
and may be wrong.

For every span return, in this order:
- "id": the span id.
- "type": "dialogue" if the span is words spoken aloud or thought by a character, \
otherwise "narration" (author's words, description, speech tags such as "she said \
quietly").
- "cue": for dialogue, copy the few words from the text that tell who speaks (the speech \
tag such as "проговорила старушка" or "сказал он"), or write "alternation" if you rely on \
turn-taking, or "context". For narration use "".
- "speaker": for narration always "narrator". For dialogue, the character id of the person \
who speaks.
- "emotion": one of neutral, calm, happy, sad, angry, afraid, whisper, excited, serious. \
Use "neutral" unless the text clearly signals otherwise.

HOW TO FIND THE SPEAKER (the most important part of the task):
1. A speech tag in the SAME paragraph (narration right before or after the line: "сказал \
Иван", "проговорила старушка", "подумал он") names the speaker of the dialogue in that \
paragraph. Always check it first.
2. Tags often describe instead of naming ("молодой человек", "старуха", "он", "она", \
"хозяйка"). Resolve the description to the character it refers to, using the context and \
the known characters. Do not create a new character for a description of a known one.
3. In an untagged exchange between two people the lines alternate: each new dialogue \
paragraph is spoken by the other participant. Follow the alternation from the last \
tagged line.
4. A name inside a line usually names the person being ADDRESSED, not the speaker: \
"Иван Петрович, вы здесь?" is said TO Ivan Petrovich by someone else.
5. Thoughts and inner monologue ("подумал он") are dialogue of the thinking character.
6. A line shouted by an unnamed bystander belongs to a separate minor character (for \
example "drunk_man"), never to the main character just because he is nearby.
7. If you still cannot tell, use "unknown". Never guess randomly.

Also return "characters": every character who speaks in this passage, plus important \
characters who are named. Rules:
- "id": short lowercase latin transliteration of the canonical name (e.g. "anna", \
"rodion_raskolnikov"). Reuse the id of a KNOWN character whenever it is the same person.
- "name": the canonical name exactly as spelled in the book (Cyrillic for Russian books), \
nominative case. Never translate or transliterate it and never mix alphabets.
- "aliases": other names, nicknames and descriptions used for this person in the passage.
- "gender": male, female or unknown. "age_group": child, teen, young_adult, adult, \
elderly or unknown.
- "description": one short sentence (role, voice-relevant traits).
- "same_as": id of a known character only if you are confident they are the same person \
under another name; otherwise null.
- "possibly_same_as": known ids that might be the same person when you are NOT sure.

Respond with a single JSON object and nothing else:
{"characters": [...], "segments": [{"id": 0, "type": "...", "cue": "...", \
"speaker": "...", "emotion": "..."}, ...]}
Include exactly one entry in "segments" for every span id you were given."""

RETRY_PROMPT = """\
Your previous reply could not be used: {error}
Reply again with ONLY the JSON object (no markdown, no commentary). It must contain \
"characters" and "segments", and "segments" must have one entry for each of these span \
ids: {ids}."""


def build_user_prompt(request: ChunkRequest) -> str:
    lines: list[str] = [f"Book: {request.book_title}", f"Chapter: {request.chapter_title}"]
    if request.language:
        lines.append(f"Language: {request.language}")

    if request.known_characters:
        lines.append("\nKNOWN CHARACTERS (reuse these ids):")
        for ch in request.known_characters:
            aliases = f" (aliases: {', '.join(ch.aliases)})" if ch.aliases else ""
            desc = f" - {ch.description}" if ch.description else ""
            lines.append(f"- {ch.id}: {ch.name}{aliases}, {ch.gender}{desc}")
    else:
        lines.append("\nKNOWN CHARACTERS: none yet")

    if request.recent_speakers:
        lines.append(
            f"\nMost recent speakers before this passage: {', '.join(request.recent_speakers)}"
        )

    if request.context:
        lines.append("\nPRECEDING CONTEXT (for understanding only, do not label):")
        lines.extend(request.context)

    lines.append("\nPASSAGE TO LABEL (one paragraph per line):")
    paragraph: list[str] = []
    current: int | None = None
    for span in request.spans:
        if current is not None and span.paragraph != current:
            lines.append(" ".join(paragraph))
            paragraph = []
        current = span.paragraph
        paragraph.append(f"[{span.id}:{span.hint}] {span.text}")
    if paragraph:
        lines.append(" ".join(paragraph))
    return "\n".join(lines)
