# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

"Comentario" — a Spanish-vocab learning tool. It listens to a live Spanish-language broadcast (typically soccer), transcribes it on the Apple Silicon GPU, lemmatizes and counts words, translates them offline, and serves a browser dashboard where you drag words between **New / Learning / Known** columns.

There are **two execution paths** that share the analysis code:
- **Live mode** (`spanish_vocab.app`) — the primary path. Captures system audio in real time.
- **File/batch mode** (`spanish_vocab.pipeline`) — transcribes an existing recording in one shot.

## Commands

```bash
# Setup (one-time)
brew install blackhole-2ch          # virtual audio loopback (macOS can't capture system audio directly)
pip install -r requirements.txt
python -m spacy download es_core_news_md   # Spanish NLP model, required, installed separately

# Live mode
python -m spanish_vocab.app --list-devices   # confirm BlackHole appears as an input device
python -m spanish_vocab.app                   # auto-detects BlackHole, serves dashboard at http://localhost:8000

# File/batch mode
python -m spanish_vocab.transcribe match.mp3 -o segments.json
python -m spanish_vocab.analyze segments.json -o session/vocab.json
# or the combined pipeline:
python -m spanish_vocab.pipeline match.mp3 --device cpu
```

There is **no test suite, linter, or build step** configured. Each module is runnable standalone via its `__main__` block.

### Audio routing (required for live mode)
macOS hides system audio from apps. In **Audio MIDI Setup**, create a **Multi-Output Device** containing both **BlackHole 2ch** and your speakers, then set system/app output to it. You hear audio normally; BlackHole gets a copy that `live_capture.py` reads.

## Architecture

The pipeline is the same conceptually in both modes — **audio → transcript → lemma counts → translation → `vocab.json` → dashboard** — but the backends differ:

| Concern | Live mode | File/batch mode |
|---|---|---|
| Transcription | `live_transcribe.Transcriber`, **mlx-whisper** (Metal GPU), default `whisper-small-mlx` | `transcribe.transcribe`, **faster-whisper** (CTranslate2), default `large-v3` |
| Counting/state | `live_store.VocabStore` (incremental, mutable) | `analyze.build_vocabulary` (one batch over all segments) |
| Translation | argos-translate, per-lemma, cached as words arrive | argos-translate, all entries at the end |

Shared analysis primitives live in **`analyze.py`** and are imported by `live_store.py`: `KEEP_POS`, `FUNCTION_POS`, `POS_LABELS`, `_load_nlp` (loads spaCy `es_core_news_md` with NER/parser disabled), and `_ensure_argos` (lazily downloads the es→en argos package on first run). When changing POS handling, lemma keying, or the translation setup, edit `analyze.py` and both paths inherit it.

### Live-mode data flow (`app.py` orchestrates)
1. `live_capture.AudioWindower` reads BlackHole via `sounddevice`, resamples to 16 kHz mono (`scipy.resample_poly`), and yields **overlapping windows** (`--window` 10s, `--overlap` 1s) tagged `is_first`.
2. `Transcriber.transcribe_words` returns `(word, start_seconds)` pairs. `app.py` drops words whose timestamp falls inside the overlap region (except the first window) to **de-duplicate** the shared audio between consecutive windows.
3. `VocabStore.add_text` lemmatizes, increments `(lemma, pos)` counts, and translates each new lemma once.
4. After each window, `VocabStore.write` does an **atomic write** (tmpfile + `os.replace`) of `session/vocab.json` so the dashboard never reads a half-written file.

### The `vocab.json` contract
Both modes emit the same shape, and the dashboard depends on it:
```
{ total_words, unique_words, words: [ { lemma, pos, pos_label, count,
    is_function_word, surface_forms, translation, example_es } ] }
```
(File mode also adds `example_en`.) If you change these field names, update `web/index.html` too.

### Dashboard (`web/index.html`, ~300 lines, no build/deps)
- Single static page. `app.py` copies it into the output dir (default `session/`) next to `vocab.json` and serves both via `http.server`. It **polls `vocab.json`** on an interval (`POLL_MS`) — there is no websocket/push.
- **New/Learning/Known categorization lives entirely in the browser** (`localStorage` key `vocabCats`), keyed by word, **not** in `vocab.json`. The Python side only ever produces counts and translations; it has no notion of a word's learning category. Newly heard words default to "New".
- Function words (`is_function_word`, e.g. el/de/que) are hidden by default; a header toggle reveals them.

## Conventions
- Each word is counted by **lemma** (so *marcó/marcando/marca* → *marcar*) keyed on `(lemma, pos)`; `surface_forms` keeps the top-5 observed inflections. Only POS in `KEEP_POS` and `tok.is_alpha` tokens are counted.
- Optional heavy deps (`faster_whisper`, `mlx_whisper`, `argostranslate`) are **imported lazily** inside functions/constructors so the unused backend isn't required. Preserve this when adding code.
- Live mode targets **Apple Silicon / Metal**; file mode's `--device cuda` path targets a GPU node. `--backend faster` is the CPU fallback for live mode.
