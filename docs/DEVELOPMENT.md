# Development

## Setup

```powershell
cd C:\AI\reader
python -m venv .venv                      # skip if .venv already exists
.\.venv\Scripts\Activate.ps1
# PyTorch with CUDA first (see README), then:
pip install -e ".[dev,api]"
pip install -e ".[tts]"                   # qwen-tts; does not replace an existing torch
```

## Everyday commands

```powershell
ruff check .                 # lint
ruff format .                # format
pytest                       # unit + integration tests (fake LLM/TTS, real FFmpeg if found)
pytest -k registry -v        # a subset
python -m audiobooks doctor  # environment check
```

Optional real-model smoke tests (GPU must be free; uses your `.env`):

```powershell
$env:AUDIOBOOKS_REAL_MODELS = "1"
pytest tests/test_real_models.py -s
Remove-Item Env:AUDIOBOOKS_REAL_MODELS
```

Dry run of the full pipeline without any model (tones instead of speech):

```powershell
$env:TTS_BACKEND = "fake"
python -m audiobooks generate input\book.txt --mode simple
Remove-Item Env:TTS_BACKEND
```

## Test layout

| File | Covers |
|---|---|
| `tests/test_parsers.py` | TXT (encodings, headings), FB2 (sections, notes, XXE), FB2.ZIP, EPUB, PDF |
| `tests/test_text.py` | dialogue segmentation, sentence splitting, chunking |
| `tests/test_registry_casting.py` | registry persistence, alias merging, uncertainty, stable voices |
| `tests/test_llm.py` | JSON repair, schema validation, retries, HTTP errors (mock transport) |
| `tests/test_ffmpeg.py` | FFmpeg command construction, concat/metadata escaping |
| `tests/test_storage.py` | SQLite repository, serialization, filename sanitizing |
| `tests/test_pipeline.py` | end-to-end with fake LLM/TTS: resume, caching, failures |
| `tests/test_api.py` | FastAPI endpoints |
| `tests/test_real_models.py` | optional Ornith + Qwen3-TTS smoke tests |

Tests never read `.env` (`Settings(_env_file=None)`), never download models and never need
a GPU.

## Conventions

* Python ≥ 3.11, type hints everywhere, Pydantic for anything crossing a boundary
  (files, LLM, API).
* `pathlib` for paths; subprocesses always get argument lists, never shell strings.
* Expected failures raise subclasses of `audiobooks.errors.AudioBooksError`; the CLI and
  API turn them into messages, everything else is a bug.
* Prompts live only in `llm/prompts.py`. Changing prompt semantics → bump
  `PROMPT_VERSION` so cached chunk analyses are invalidated.
* Changing what affects segment audio → change the cache key in
  `services/synthesis.py` (`seg-v1`) deliberately.
* New heavy components are added behind a protocol (`LLMProvider`, `TTSBackend`,
  `PageTextExtractor`) and injected through `AppContext`.

## Adding a book format

1. Create `parsers/<fmt>.py` with a `BookParser` subclass that returns
   `self.finalize(...)`.
2. Register the extension in `parsers/__init__.py` (`SUPPORTED_EXTENSIONS`, `get_parser`).
3. Add a test that builds a tiny file in `tmp_path`.

## Adding a TTS engine

Implement `TTSBackend` (`model_tag`, `synthesize`, `close`) in `tts/<engine>.py`, return
it from `tts.create_backend` for a new `TTS_BACKEND` value, and make `close()` release all
GPU memory.
