# Comentario — learn Spanish from live soccer commentary

Watch a Spanish-language broadcast on your Mac. Comentario listens to the audio, transcribes it
on the Apple Silicon GPU, counts how often each word is used, translates it offline, and serves
a browser app where you **track** vocabulary on a live board and **drill** it with flashcards —
earning each word's way from **New → Learning → Known**.

```
audio (BlackHole) ─▶ Whisper (on-device) ─▶ lemmatize + count (spaCy) ─▶ translate (argos) ─▶ vocab.json ─▶ browser
                                                                                                          ├─ Board (live frequency + columns)
                                                                                                          └─ Flashcards (study + promote)
```

Everything runs **on-device and offline** — no API keys, no audio leaves your machine.

---

## Contents
- [How it works](#how-it-works)
- [Requirements](#requirements)
- [One-time setup](#one-time-setup-5-min)
- [Running it](#running-it)
- [Command-line options](#command-line-options)
- [The Board](#the-board)
- [The Flashcards](#the-flashcards)
- [How counting & promotion work](#how-counting--promotion-work)
- [Accuracy & tuning](#accuracy--tuning)
- [File-based mode (no live capture)](#file-based-mode-no-live-capture)
- [Project layout](#project-layout)
- [Troubleshooting](#troubleshooting)

---

## How it works

macOS hides system audio from apps, so playback is routed through a free virtual loopback device
(**BlackHole**) that Comentario reads. Audio is captured in overlapping windows, each transcribed
independently by Whisper; overlapping words are de-duplicated by their timestamps. Each word is
reduced to its dictionary form (**lemma**) and tagged with a part of speech, counted, and
translated once. The running tally is written to `session/vocab.json`, which both browser pages
poll a few times a second.

Your learning progress (which words are New/Learning/Known, and how close each is to its next
promotion) lives entirely in the **browser's `localStorage`** — the Python side only ever produces
counts and translations. That means the Board and Flashcards always stay in sync, and your
progress survives restarts.

## Requirements

- **Apple Silicon Mac** (M1/M2/M3/M4). The live transcription path uses MLX/Metal and will not run
  on Intel Macs — see [File-based mode](#file-based-mode-no-live-capture) for the CPU fallback.
- **Python 3.10+**
- **Homebrew** (to install the BlackHole audio loopback)
- ~2 GB free disk for the Whisper model weights (downloaded once on first run)

## One-time setup (~5 min)

```bash
# 1. Audio loopback so the app can hear system audio
brew install blackhole-2ch

# 2. Python dependencies
pip install -r requirements.txt

# 3. Spanish NLP model (installed separately by spaCy)
python -m spacy download es_core_news_md
```

**Hear the match *and* capture it.** Open **Audio MIDI Setup** → click **“+”** →
**Create Multi-Output Device** → tick both **BlackHole 2ch** and your **speakers/headphones**.
Then set your browser/video app's sound output (or the macOS system output) to that
Multi-Output Device. You hear the audio normally; BlackHole receives a copy for the app.

## Running it

```bash
python -m spanish_vocab.app --list-devices   # confirm BlackHole appears as an input device
python -m spanish_vocab.app                  # auto-detects BlackHole, opens the dashboard
```

Play the broadcast (routed through BlackHole). The app opens **http://localhost:8000** and the
Board fills in live. Switch to the **Flashcards** tab any time to study what you've collected.
`Ctrl-C` stops the capture; your session is saved in `session/vocab.json` and your learning
progress is kept in the browser.

> **First run** downloads the Whisper model (~1.5 GB for the default `medium`) and the offline
> `es → en` translation package. Both are cached afterward.

### Command-line options

| flag | default | notes |
|---|---|---|
| `--list-devices` | — | print available audio devices and exit |
| `--device` | auto | input device index or name (defaults to auto-detected BlackHole) |
| `--model` | `mlx-community/whisper-medium-mlx` | accurate but light enough for an 8 GB laptop. Lighter/faster: `…/whisper-small-mlx`, `…/whisper-base-mlx`. More accurate (heavier): `…/whisper-large-v3-turbo`, `…/whisper-large-v3-mlx`. |
| `--backend` | `mlx` | `mlx` = Apple-Silicon GPU; `faster` = CPU faster-whisper fallback |
| `--window` | `10` | seconds of audio per transcription window |
| `--overlap` | `1` | seconds shared between consecutive windows (de-duplicated by word timestamps) |
| `--port` | `8000` | port for the local dashboard |
| `--out-dir` | `session` | where `vocab.json` and the served pages are written |
| `--no-translate` | off | skip translation for extra speed |
| `--no-prompt` | off | disable the soccer-vocabulary bias prompt fed to Whisper |
| `--no-browser` | off | don't auto-open the browser |

---

## The Board

The live tracker (the **Board** tab). As words are heard they appear in the **New** column; you
drag cards between **New / Learning / Known**, or use the per-card buttons. A **Most frequent**
ticker shows the top words by count, and each card shows its translation, an example sentence
(click to expand), part of speech, and how often it's been heard.

Header toggles:
- **show names** — names of players, teams, and places (tagged as proper nouns) are **hidden by
  default**; tick to include them.
- **show function words** — grammar glue (*el, de, que, …*) is hidden by default; tick to include.
- **Search** filters by Spanish word or English translation.

## The Flashcards

A Quizlet-style study tab built from the same words. Three **progress bars** in the top-right show
how your vocabulary is distributed across New / Learning / Known and shift as you study.

**Pick a deck:** **All cards**, **New**, **Learning**, or **Brush up** (Known). Each deck is
shuffled.

**Study a card:**
- The front shows the **Spanish** word; press **`Space`** (or click) to flip and reveal the
  **English** translation and an example sentence.
- After flipping, grade yourself with the arrow keys:
  - **`→` "Know"** — a correct answer (advances the word).
  - **`←` "Don't know"** — demotes the word one level.
- The arrows are ignored until you flip, so you always self-test first.

**Deck filters** (top bar): **show names** and **show untranslated** — by default, proper nouns
and words the translator couldn't translate (where the English equals the Spanish, e.g.
*córner → córner*) are excluded from decks so you only drill real, useful pairs. Tick either to
include them.

## How counting & promotion work

**Counting (lemma + POS).** Every word is reduced to its lemma before counting, so *marcó*,
*marcando*, and *marca* all roll up into **marcar**. Part of speech is tracked so the UI can group
and filter words (nouns, verbs, names, function words…).

**Promotion (earned through flashcards).** Each word advances by getting it right:

| From | To | Correct answers needed |
|---|---|---|
| New | Learning | **1** |
| Learning | Known | **2** |

A wrong answer (**`←`**) demotes a word one level (Known → Learning → New). Correct-answer counts
**accumulate and persist** across sessions, so a word advances over time. Promotions earned in the
Flashcards tab are reflected on the Board, and vice-versa — both read the same browser store.

## Accuracy & tuning

Transcription quality gates everything downstream, so it's the biggest lever:

- **Model.** The default `whisper-medium-mlx` balances accuracy against an 8 GB laptop. For more
  accuracy on a roomier machine, try `--model mlx-community/whisper-large-v3-turbo`. To stay
  lightest, `--model mlx-community/whisper-small-mlx`.
- **Soccer prompt.** By default Whisper is biased with a soccer-vocabulary prompt so it spells
  domain terms correctly (*córner*, *penalti*, *fuera de juego*, *fútbol*). Disable with
  `--no-prompt` if you ever find it over-eager.
- **Windowing.** `--window` / `--overlap` trade latency against context at window boundaries.

## File-based mode (no live capture)

Already have a recording (or on an Intel Mac)? Transcribe and analyze a file directly:

```bash
# Stage 1: audio file → transcript
python -m spanish_vocab.transcribe match.mp3 -o segments.json

# Stage 2+3: lemmatize, count, translate → vocab.json the pages read
python -m spanish_vocab.analyze segments.json -o session/vocab.json

# …or run both stages at once:
python -m spanish_vocab.pipeline match.mp3 --device cpu
```

This path uses **faster-whisper** (CPU or CUDA) instead of MLX. Drop the resulting `vocab.json`
into the `session/` folder and open the pages there to study.

## Project layout

```
spanish_vocab/
  app.py             live orchestrator: capture → transcribe → count → serve dashboard
  live_capture.py    reads BlackHole, yields overlapping 16 kHz audio windows
  live_transcribe.py Whisper transcription (mlx default, faster-whisper fallback) + soccer prompt
  live_store.py      running lemma counts + translations; atomic-writes vocab.json
  transcribe.py      file-based Stage 1 (faster-whisper)
  analyze.py         file-based Stage 2+3 (spaCy lemmatize/count + argos translate)
  pipeline.py        runs the file-based stages end to end
web/
  index.html         the Board (live frequency + New/Learning/Known columns)
  flashcards.html    the Flashcards study page
session/
  vocab.json         written live; the pages poll it
  index.html         copy served alongside vocab.json (also flashcards.html)
```

> Learning state is stored in the browser, not in `vocab.json`: `localStorage["vocabCats"]` holds
> each word's New/Learning/Known category and `localStorage["vocabProgress"]` holds its
> correct-answer count toward the next promotion.

## Troubleshooting

- **“No BlackHole input device found.”** Confirm `brew install blackhole-2ch` succeeded and that
  it shows up under `python -m spanish_vocab.app --list-devices`. You can also force a device with
  `--device <index|name>`.
- **The Board stays empty.** Make sure the broadcast's audio output is routed to the
  **Multi-Output Device** (or directly to BlackHole). Nothing is captured otherwise.
- **`spaCy model 'es_core_news_md' not found`.** Run `python -m spacy download es_core_news_md`.
- **It's lagging / fans spinning.** Use a lighter model (`--model mlx-community/whisper-small-mlx`)
  or a larger `--window`.
- **Lots of `córner → córner` style cards.** That's the offline translator leaving a word
  unchanged; the Flashcards tab hides these by default (untick **show untranslated** to confirm,
  or tick it to study them anyway).
