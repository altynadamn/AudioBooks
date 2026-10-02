# Voice library

```
voices/
    metadata/      one JSON file per voice (committed)
    references/    reference recordings for cloned voices (NOT committed, *.wav is ignored)
```

* `metadata/narrator_01.json` – default narrator, cloned from a local recording.
  Put a clean 10–20 s mono recording at `references/narrator_01.wav` whose transcript
  matches `reference_text` exactly.
* `metadata/qwen_builtin.json` – built-in speakers of
  `Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice` (no recordings needed).

Add your own cloned voice:

```powershell
python -m audiobooks voices add female_01 path\to\clip.wav --text "Exact transcript of the clip." --gender female --age adult --style warm
```

Only use recordings you have the right to use (your own voice, voice actors who agreed,
or clearly licensed public-domain/CC0 material). See `docs/VOICE_SYSTEM.md`.
