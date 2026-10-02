"""Prompt templates for structured book analysis. All prompt text lives here."""

from __future__ import annotations

import json

from audiobooks.llm.schemas import ChunkRequest

PROMPT_VERSION = "chunk-v1"  # bump to invalidate cached chunk analyses

SYSTEM_PROMPT = """\
You are an expert audiobook director preparing a book for a multi-voice ("AI cast") \
recording. You receive a passage that is already split into numbered spans. Your job is to \
LABEL every span; you never rewrite, translate, merge or split the text.

For every span return:
- "type": "dialogue" if the span is words spoken aloud by a character, otherwise \
"narration" (author's words, description, speech tags such as "she said quietly"). The \
"hint" is a typographic guess and may be wrong.
- "speaker": for narration always "narrator". For dialogue, the character id of who \
speaks. Use context, speech tags, turn-taking and the known characters. If you genuinely \
cannot tell, use "unknown" - never guess randomly.
- "emotion": one of neutral, calm, happy, sad, angry, afraid, whisper, excited, serious. \
Use "neutral" unless the text clearly signals otherwise.

Also return "characters": every character who speaks in this passage, plus important \
characters who are named. Rules:
- "id": short lowercase latin slug of the canonical name (e.g. "anna", "ivan_petrovich"). \
Reuse the id of a KNOWN character whenever it is the same person.
- "name": canonical name in the book's language and nominative case.
- "aliases": other names/nicknames/forms used for this person in the passage.
- "gender": male, female or unknown. "age_group": child, teen, young_adult, adult, \
elderly or unknown.
- "description": one short sentence (role, voice-relevant traits).
- "same_as": id of a known character only if you are confident they are the same \
person under another name; otherwise null.
- "possibly_same_as": known ids that might be the same person when you are NOT sure.

Respond with a single JSON object and nothing else:
{"characters": [...], "segments": [{"id": 0, "type": "...", "speaker": "...", \
"emotion": "..."}, ...]}
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

    lines.append("\nSPANS TO LABEL (JSON lines: id, hint, text):")
    for span in request.spans:
        lines.append(
            json.dumps({"id": span.id, "hint": span.hint, "text": span.text}, ensure_ascii=False)
        )
    return "\n".join(lines)
