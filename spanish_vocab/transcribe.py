"""Stage 1 — Transcribe Spanish audio to text with word-level timestamps.

Uses faster-whisper (CTranslate2). On a GPU node set device="cuda" for a big
speedup; on CPU use compute_type="int8".
"""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path


@dataclass
class Segment:
    start: float
    end: float
    text: str


def transcribe(
    audio_path: str | Path,
    model_size: str = "large-v3",
    device: str = "cuda",          # use "cpu" if no GPU
    compute_type: str = "float16",  # use "int8" on CPU
    language: str = "es",
) -> list[Segment]:
    """Transcribe an audio file and return a list of Segments."""
    from faster_whisper import WhisperModel  # imported lazily (optional dep)

    audio_path = Path(audio_path)
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio not found: {audio_path}")

    model = WhisperModel(model_size, device=device, compute_type=compute_type)

    # vad_filter trims long silences, which helps with stadium ambience.
    segments, info = model.transcribe(
        str(audio_path),
        language=language,
        vad_filter=True,
        beam_size=5,
    )

    print(f"Detected language '{info.language}' (p={info.language_probability:.2f})")

    out: list[Segment] = []
    for seg in segments:
        text = seg.text.strip()
        if text:
            out.append(Segment(start=seg.start, end=seg.end, text=text))
    return out


def save_segments(segments: list[Segment], path: str | Path) -> None:
    path = Path(path)
    path.write_text(
        json.dumps([asdict(s) for s in segments], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Wrote {len(segments)} segments -> {path}")


def load_segments(path: str | Path) -> list[Segment]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return [Segment(**d) for d in data]


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Transcribe Spanish audio.")
    ap.add_argument("audio", help="Path to audio file (mp3/wav/m4a/...)")
    ap.add_argument("-o", "--out", default="segments.json")
    ap.add_argument("--model", default="large-v3")
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    args = ap.parse_args()

    compute = "float16" if args.device == "cuda" else "int8"
    segs = transcribe(args.audio, model_size=args.model,
                      device=args.device, compute_type=compute)
    save_segments(segs, args.out)
