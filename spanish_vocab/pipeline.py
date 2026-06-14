"""End-to-end pipeline: audio file -> vocab.json -> ready for the web page.

Usage:
    python -m spanish_vocab.pipeline match.mp3 --device cuda

Produces:
    segments.json   (raw transcript)
    vocab.json      (ranked vocabulary the web page reads)
"""
from __future__ import annotations

import argparse
from pathlib import Path

from .transcribe import transcribe, save_segments
from .analyze import build_vocabulary, translate_entries, save_vocabulary


def run(
    audio: str,
    out_dir: str = ".",
    model_size: str = "large-v3",
    device: str = "cuda",
    min_count: int = 2,
    translate: bool = True,
) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    compute = "float16" if device == "cuda" else "int8"

    print("== Stage 1: transcribe ==")
    segments = transcribe(audio, model_size=model_size,
                          device=device, compute_type=compute)
    save_segments(segments, out / "segments.json")

    print("== Stage 2: lemmatize + count ==")
    vocab = build_vocabulary(segments, min_count=min_count)
    print(f"  {len(vocab)} unique lemmas (min_count={min_count})")

    if translate:
        print("== Stage 3: translate ==")
        vocab = translate_entries(vocab)

    save_vocabulary(vocab, out / "vocab.json")
    print("\nDone. Open vocab.html next to vocab.json to study.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Spanish commentary -> vocab pipeline")
    ap.add_argument("audio", help="Path to audio file")
    ap.add_argument("--out-dir", default=".")
    ap.add_argument("--model", default="large-v3")
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--min-count", type=int, default=2,
                    help="Drop words appearing fewer than this many times")
    ap.add_argument("--no-translate", action="store_true")
    args = ap.parse_args()

    run(
        audio=args.audio,
        out_dir=args.out_dir,
        model_size=args.model,
        device=args.device,
        min_count=args.min_count,
        translate=not args.no_translate,
    )
