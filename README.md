# Comentario — live Spanish-commentary vocab tracker

Watch a Spanish-language broadcast on your Mac; the app listens to the audio,
transcribes it on the Apple Silicon GPU, counts how often each word is used,
translates it, and shows everything on a live dashboard with **New / Learning /
Known** columns you drag words between.

```
audio (BlackHole) ──▶ Whisper (small, on-device) ──▶ lemmatize + count ──▶ translate ──▶ live dashboard
```

## 1. One-time setup (~5 min)

macOS won't let an app capture system audio directly, so route playback through
a free virtual device (BlackHole) that the app can listen to.

```bash
# audio loopback
brew install blackhole-2ch

# python deps
pip install -r requirements.txt
python -m spacy download es_core_news_md
```

**Hear the match *and* capture it:** open **Audio MIDI Setup** → "+" →
**Create Multi-Output Device** → check both **BlackHole 2ch** and your
**speakers/headphones**. Then set your browser/video app's sound output (or the
system output) to that Multi-Output Device. You hear audio normally; BlackHole
gets a copy.

## 2. Run

```bash
python -m spanish_vocab.app --list-devices   # confirm BlackHole shows up
python -m spanish_vocab.app                   # auto-detects BlackHole, opens dashboard
```

Play the broadcast. The dashboard at <http://localhost:8000> fills in live.
`Ctrl-C` stops; your session is saved in `session/vocab.json`.

### Options
| flag | default | notes |
|---|---|---|
| `--model` | `mlx-community/whisper-medium-mlx` | accurate but light enough for an 8 GB Apple-Silicon laptop. Lighter/faster: `...whisper-small-mlx`, `...whisper-base-mlx`. More accurate (heavier): `...whisper-large-v3-turbo`, `...whisper-large-v3-mlx`. |
| `--window` | `10` | seconds per transcription window |
| `--overlap` | `1` | seconds shared between windows (de-duplicated by word timestamps) |
| `--device` | auto | input device index or name |
| `--backend` | `mlx` | `faster` = CPU faster-whisper fallback |
| `--no-translate` | off | skip translation for speed |
| `--no-prompt` | off | disable the soccer-vocabulary bias prompt fed to Whisper (correct spellings like *córner*, *penalti*, *fuera de juego*) |

## How counting works
Each word is reduced to its **lemma** before counting (so *marcó, marcando,
marca* → **marcar**), and part of speech is tracked so the board can group
nouns/verbs/adjectives. Function words (el, de, que…) are hidden by default —
toggle "show function words" in the header to include them.

## File-based mode (no live capture)
Already have a recording? The original batch path still works:
```bash
python -m spanish_vocab.transcribe match.mp3 -o segments.json
python -m spanish_vocab.analyze segments.json -o session/vocab.json
```

## Layout
```
spanish_vocab/      transcribe · analyze · live_capture · live_transcribe · live_store · app
web/index.html      the dashboard
session/vocab.json  written live; the dashboard polls it
```
