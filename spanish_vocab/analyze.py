"""Stage 2+3 — Lemmatize, count word frequencies, and translate.

Takes the transcript Segments and produces a ranked vocabulary list:
each entry is a lemma with its count, part of speech, the surface forms
seen, an example sentence, and English translations.
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import dataclass, field, asdict
from pathlib import Path

import spacy

from .transcribe import Segment, load_segments

# Coarse POS buckets we care about for a learner. Everything else (PUNCT,
# SYM, NUM, X, SPACE) is dropped. DET/PRON/ADP/CCONJ are "function words"
# kept but tagged so the UI can filter them out by default.
KEEP_POS = {
    "NOUN", "PROPN", "VERB", "ADJ", "ADV",
    "DET", "PRON", "ADP", "CCONJ", "SCONJ", "AUX", "INTJ",
}
FUNCTION_POS = {"DET", "PRON", "ADP", "CCONJ", "SCONJ", "AUX"}

POS_LABELS = {
    "NOUN": "noun", "PROPN": "proper noun", "VERB": "verb", "ADJ": "adjective",
    "ADV": "adverb", "DET": "determiner", "PRON": "pronoun", "ADP": "preposition",
    "CCONJ": "conjunction", "SCONJ": "conjunction", "AUX": "auxiliary",
    "INTJ": "interjection",
}


@dataclass
class WordEntry:
    lemma: str
    pos: str                       # coarse POS, e.g. "VERB"
    pos_label: str                 # human label, e.g. "verb"
    count: int
    is_function_word: bool
    surface_forms: list[str] = field(default_factory=list)
    example_es: str = ""
    example_en: str = ""
    translation: str = ""


def _load_nlp(model: str = "es_core_news_md"):
    try:
        return spacy.load(model, disable=["ner", "parser"])
    except OSError as e:
        raise SystemExit(
            f"spaCy model '{model}' not found. Install it with:\n"
            f"    python -m spacy download {model}"
        ) from e


def build_vocabulary(
    segments: list[Segment],
    spacy_model: str = "es_core_news_md",
    min_count: int = 1,
) -> list[WordEntry]:
    nlp = _load_nlp(spacy_model)

    counts: Counter[tuple[str, str]] = Counter()      # (lemma, pos) -> count
    forms: dict[tuple[str, str], Counter] = defaultdict(Counter)
    example: dict[tuple[str, str], str] = {}

    # Process all segments; nlp.pipe is faster for many short texts.
    texts = [s.text for s in segments]
    for seg_text, doc in zip(texts, nlp.pipe(texts, batch_size=64)):
        for tok in doc:
            if not tok.is_alpha or tok.pos_ not in KEEP_POS:
                continue
            lemma = tok.lemma_.lower().strip()
            if not lemma:
                continue
            key = (lemma, tok.pos_)
            counts[key] += 1
            forms[key][tok.text.lower()] += 1
            example.setdefault(key, seg_text)  # first sentence it appeared in

    entries: list[WordEntry] = []
    for (lemma, pos), count in counts.most_common():
        if count < min_count:
            continue
        entries.append(
            WordEntry(
                lemma=lemma,
                pos=pos,
                pos_label=POS_LABELS.get(pos, pos.lower()),
                count=count,
                is_function_word=pos in FUNCTION_POS,
                surface_forms=[w for w, _ in forms[(lemma, pos)].most_common(5)],
                example_es=example.get((lemma, pos), ""),
            )
        )
    return entries


# --------------------------------------------------------------------------- #
# Translation (offline via argos-translate)
# --------------------------------------------------------------------------- #
def _ensure_argos(from_code: str = "es", to_code: str = "en"):
    """Install the es->en package once, then return the translate function."""
    import argostranslate.package
    import argostranslate.translate

    installed = {
        (p.from_code, p.to_code)
        for p in argostranslate.package.get_installed_packages()
    }
    if (from_code, to_code) not in installed:
        print("Downloading argos es->en package (first run only)...")
        argostranslate.package.update_package_index()
        available = argostranslate.package.get_available_packages()
        pkg = next(
            p for p in available
            if p.from_code == from_code and p.to_code == to_code
        )
        argostranslate.package.install_from_path(pkg.download())

    return lambda text: argostranslate.translate.translate(text, from_code, to_code)


def translate_entries(
    entries: list[WordEntry],
    translate_examples: bool = True,
) -> list[WordEntry]:
    translate = _ensure_argos()
    for e in entries:
        e.translation = translate(e.lemma).strip()
        if translate_examples and e.example_es:
            e.example_en = translate(e.example_es).strip()
    return entries


def save_vocabulary(entries: list[WordEntry], path: str | Path) -> None:
    path = Path(path)
    payload = {
        "total_words": sum(e.count for e in entries),
        "unique_words": len(entries),
        "words": [asdict(e) for e in entries],
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                    encoding="utf-8")
    print(f"Wrote {len(entries)} unique words -> {path}")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Build a vocabulary from a transcript.")
    ap.add_argument("segments", help="segments.json from transcribe.py")
    ap.add_argument("-o", "--out", default="vocab.json")
    ap.add_argument("--min-count", type=int, default=1)
    ap.add_argument("--no-translate", action="store_true")
    args = ap.parse_args()

    segs = load_segments(args.segments)
    vocab = build_vocabulary(segs, min_count=args.min_count)
    if not args.no_translate:
        vocab = translate_entries(vocab)
    save_vocabulary(vocab, args.out)
