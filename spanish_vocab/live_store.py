"""Running vocabulary store, updated window-by-window.

Lemmatizes incoming text, increments per-lemma counts, translates each new
lemma once (cached), and writes vocab.json atomically for the dashboard to poll.
Category state (new/learning/known) lives in the browser, not here.
"""
from __future__ import annotations

import json
import os
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

from .analyze import (
    KEEP_POS, FUNCTION_POS, POS_LABELS, _load_nlp, _ensure_argos,
)


class VocabStore:
    def __init__(self, spacy_model: str = "es_core_news_md", translate: bool = True):
        self.nlp = _load_nlp(spacy_model)
        self._translate = _ensure_argos() if translate else None

        self.counts: Counter = Counter()                 # (lemma, pos) -> count
        self.forms: dict = defaultdict(Counter)
        self.example: dict = {}
        self.translation: dict = {}                       # lemma -> en
        self.total_tokens = 0

    def add_text(self, text: str) -> int:
        """Process one window's text; return number of countable tokens added."""
        text = text.strip()
        if not text:
            return 0
        doc = self.nlp(text)
        added = 0
        for tok in doc:
            if not tok.is_alpha or tok.pos_ not in KEEP_POS:
                continue
            lemma = tok.lemma_.lower().strip()
            if not lemma:
                continue
            key = (lemma, tok.pos_)
            self.counts[key] += 1
            self.forms[key][tok.text.lower()] += 1
            self.example.setdefault(key, text[:140])
            if self._translate and lemma not in self.translation:
                try:
                    self.translation[lemma] = self._translate(lemma).strip()
                except Exception:
                    self.translation[lemma] = ""
            self.total_tokens += 1
            added += 1
        return added

    def to_payload(self) -> dict:
        words = []
        for (lemma, pos), count in self.counts.most_common():
            words.append({
                "lemma": lemma,
                "pos": pos,
                "pos_label": POS_LABELS.get(pos, pos.lower()),
                "count": count,
                "is_function_word": pos in FUNCTION_POS,
                "surface_forms": [w for w, _ in self.forms[(lemma, pos)].most_common(5)],
                "translation": self.translation.get(lemma, ""),
                "example_es": self.example.get((lemma, pos), ""),
            })
        return {
            "total_words": self.total_tokens,
            "unique_words": len(self.counts),
            "words": words,
        }

    def write(self, path: str | Path) -> None:
        path = Path(path)
        data = json.dumps(self.to_payload(), ensure_ascii=False)
        # Atomic write so the dashboard never reads a half-written file.
        fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(data)
            os.replace(tmp, path)
        finally:
            if os.path.exists(tmp):
                os.remove(tmp)
