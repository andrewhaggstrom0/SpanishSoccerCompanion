"""Transcribe a 16 kHz window into timestamped words.

Primary backend is mlx-whisper (runs on the Apple Silicon GPU via Metal).
Returns word-level (text, start_seconds) pairs so the caller can drop words
that fall in the overlap region and were already counted.
"""
from __future__ import annotations

import numpy as np

# Default tuned for an 8 GB Apple-Silicon laptop: clearly more accurate than
# `small` on fast, noisy commentary while staying light on RAM. Push accuracy
# on a roomier machine with `…/whisper-large-v3-turbo`; drop to
# `…/whisper-small-mlx` or `…/whisper-base-mlx` if even this feels heavy.
DEFAULT_MLX_MODEL = "mlx-community/whisper-medium-mlx"

# Fed to Whisper as `initial_prompt` on every window. It biases decoding toward
# soccer terminology and correct Spanish spelling/accents (e.g. "córner" not
# "corner", "fútbol", "penalti") — a near-free accuracy boost on commentary.
# Keep it a natural-sounding snippet; Whisper only uses the trailing ~224 tokens.
SOCCER_PROMPT = (
    "Comentario de fútbol en español. El delantero marca un gol de cabeza tras "
    "un córner. Hay un penalti, una falta y un fuera de juego. El portero, la "
    "defensa, el centrocampista, el árbitro y el entrenador. Tiro libre, tarjeta "
    "amarilla, contraataque, regate, balón, área, banda, travesaño y prórroga."
)


class Transcriber:
    def __init__(
        self,
        model: str = DEFAULT_MLX_MODEL,
        language: str = "es",
        backend: str = "mlx",
        initial_prompt: str | None = SOCCER_PROMPT,
    ):
        self.model = model
        self.language = language
        self.backend = backend
        self.initial_prompt = initial_prompt
        self._fw = None  # lazy faster-whisper model

        if backend == "mlx":
            import mlx_whisper  # noqa: F401  (validate availability early)
            self._mlx = mlx_whisper
        elif backend == "faster":
            from faster_whisper import WhisperModel
            self._fw = WhisperModel(
                model if "/" not in model else "small",
                device="cpu",
                compute_type="int8",
            )
        else:
            raise ValueError(f"Unknown backend: {backend}")

    def transcribe_words(self, audio_16k: np.ndarray) -> list[tuple[str, float]]:
        if self.backend == "mlx":
            return self._mlx_words(audio_16k)
        return self._faster_words(audio_16k)

    def _mlx_words(self, audio: np.ndarray) -> list[tuple[str, float]]:
        result = self._mlx.transcribe(
            audio,
            path_or_hf_repo=self.model,
            language=self.language,
            word_timestamps=True,
            condition_on_previous_text=False,  # windows are independent
            initial_prompt=self.initial_prompt,
        )
        words: list[tuple[str, float]] = []
        for seg in result.get("segments", []):
            for w in seg.get("words", []):
                text = w.get("word", "").strip()
                if text:
                    words.append((text, float(w.get("start", seg.get("start", 0.0)))))
        return words

    def _faster_words(self, audio: np.ndarray) -> list[tuple[str, float]]:
        segments, _ = self._fw.transcribe(
            audio, language=self.language, word_timestamps=True,
            condition_on_previous_text=False,
            initial_prompt=self.initial_prompt,
        )
        words: list[tuple[str, float]] = []
        for seg in segments:
            for w in (seg.words or []):
                text = w.word.strip()
                if text:
                    words.append((text, float(w.start)))
        return words
