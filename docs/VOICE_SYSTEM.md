# Voice system

## Two kinds of voices

| Source | Model | Needs | Quality / notes |
|---|---|---|---|
| `reference` (cloned) | `Qwen/Qwen3-TTS-12Hz-0.6B-Base`, `generate_voice_clone` | a 10–20 s clean recording + its exact transcript | best for the narrator; speaks the book's language natively if the reference does |
| `builtin` | `Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice`, `generate_custom_voice` | nothing (model download ~2.5 GB) | 9 fixed speakers; most are native Chinese/English/Japanese/Korean speakers and have an accent in Russian |

Both are described by JSON files in `voices/metadata/`. A file may contain one voice or a
list of voices.

```json
{
  "id": "male_01",
  "name": "Male 01",
  "source": "reference",
  "gender": "male",
  "age_group": "adult",
  "style": ["calm", "deep"],
  "languages": ["ru"],
  "roles": ["character"],
  "reference_audio": "references/male_01.wav",
  "reference_text": "Exact transcript of the recording.",
  "license": "Recorded by the project author, CC0",
  "notes": ""
}
```

* `roles` – `narrator` and/or `character`. Automatic casting only picks voices with the
  `character` role and avoids the narrator voice while others exist.
* `reference_audio` is relative to `VOICE_LIBRARY_DIR` and must stay inside it.
* For built-in voices set `"source": "builtin"` and `"builtin_speaker"` (one of
  `aiden, dylan, eric, ono_anna, ryan, serena, sohee, uncle_fu, vivian` for the 0.6B
  CustomVoice model; check with `model.get_supported_speakers()` for other models).
* `TTS_BUILTIN_VOICES=false` hides built-in voices from casting (no CustomVoice download).

`python -m audiobooks voices list` shows every voice and whether it is usable
(reference file present).

## Adding cloned voices

```powershell
python -m audiobooks voices add female_01 C:\clips\female_01.wav `
    --text "Точная расшифровка этой записи." --gender female --age young_adult `
    --style warm --license "Own recording"
```

Good references: mono, 16–48 kHz, 10–20 s, no music or reverb, one speaker, natural
reading pace, and a transcript that matches the audio word for word (the Base model uses
it for in-context cloning). Without a transcript, only the speaker embedding is used,
which sounds less similar.

## Character → voice assignment

1. The LLM reports characters with gender and age group; the Character Registry stores
   them per book (see `docs/ARCHITECTURE.md`).
2. After analysis, `VoiceCaster` assigns voices to characters that have none, main
   characters (most lines) first:
   * gender match +4, mismatch −6 (unknown is neutral),
   * −0.75 per age-group step,
   * −2.5 per character already using that voice (spreads the cast),
   * ties broken by voice id → reproducible.
3. The assignment is saved in `character_registry.json` and **never changes afterwards**;
   new chapters only cast new characters.
4. Narration and `narrator` lines use `NARRATOR_VOICE`. Dialogue whose speaker is
   `unknown` uses `UNKNOWN_SPEAKER_VOICE` (default: narrator).

Override a choice:

```powershell
python -m audiobooks characters <book-id> --assign anna=female_01 --assign ivan=qwen_ryan
```

Pinned voices (`voice_locked`) are respected by casting. Changing a voice invalidates only
the affected segments' cache keys, so the next `generate` re-renders just those lines.

## Emotion

The LLM labels each segment with one of `neutral, calm, happy, sad, angry, afraid,
whisper, excited, serious` (free-form answers are normalized). The label is always stored
in the script. It is sent to TTS as a style instruction only when `TTS_USE_INSTRUCT=true`
**and** the voice uses a model that accepts `instruct` (CustomVoice). Voice cloning in
Qwen3-TTS Base has no instruction input, so emotion currently affects cloned voices only
through the model's own reading of the text.

## Legal and licensing considerations

Voice cloning reproduces a real person's voice. Before adding a reference recording:

* Use your own voice, or recordings from people who explicitly agreed to voice cloning.
* Public-domain/CC0 audiobook recordings (e.g. LibriVox) are public domain in the US,
  but check your jurisdiction, and remember that a *voice* can be protected by
  personality rights even when a recording is not.
* Do not use commercial audiobooks, films, games or celebrity voices.
* Record the source and license in the voice's `license` field.

Reference recordings are deliberately excluded from Git (`*.wav` is ignored). This
repository ships only metadata; the built-in Qwen speakers are part of the Qwen3-TTS model
(Apache-2.0, see the model card).

Generated audiobooks of copyrighted books are for personal use only unless you hold the
rights to the text.
