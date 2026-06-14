"""Live Spanish-commentary vocab tracker.

Captures system audio (BlackHole), transcribes it on the Apple Silicon GPU in
overlapping windows, counts lemmas, translates them, and serves a live
dashboard at http://localhost:PORT.

    python -m spanish_vocab.app --list-devices
    python -m spanish_vocab.app                 # auto-detects BlackHole
    python -m spanish_vocab.app --device 2 --model mlx-community/whisper-base-mlx
"""
from __future__ import annotations

import argparse
import shutil
import threading
import webbrowser
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from .live_capture import AudioWindower, list_devices
from .live_transcribe import Transcriber, DEFAULT_MLX_MODEL, SOCCER_PROMPT
from .live_store import VocabStore

WEB_DIR = Path(__file__).resolve().parent.parent / "web"


def _serve(directory: Path, port: int) -> ThreadingHTTPServer:
    handler = partial(SimpleHTTPRequestHandler, directory=str(directory))
    httpd = ThreadingHTTPServer(("127.0.0.1", port), handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd


def main() -> None:
    ap = argparse.ArgumentParser(description="Live Spanish commentary vocab tracker")
    ap.add_argument("--list-devices", action="store_true", help="List audio devices and exit")
    ap.add_argument("--device", default=None, help="Input device index or name (default: auto BlackHole)")
    ap.add_argument("--model", default=DEFAULT_MLX_MODEL)
    ap.add_argument("--backend", default="mlx", choices=["mlx", "faster"])
    ap.add_argument("--window", type=float, default=10.0)
    ap.add_argument("--overlap", type=float, default=1.0)
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--out-dir", default="session")
    ap.add_argument("--no-translate", action="store_true")
    ap.add_argument("--no-prompt", action="store_true",
                    help="Disable the soccer-vocabulary bias prompt fed to Whisper")
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args()

    if args.list_devices:
        print(list_devices())
        return

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(WEB_DIR / "index.html", out_dir / "index.html")
    shutil.copy(WEB_DIR / "flashcards.html", out_dir / "flashcards.html")
    vocab_path = out_dir / "vocab.json"
    VocabStore(translate=False).write(vocab_path)  # seed empty file

    device = args.device
    if device is not None and str(device).isdigit():
        device = int(device)

    print("Loading models (first run downloads weights)...")
    transcriber = Transcriber(
        model=args.model, backend=args.backend,
        initial_prompt=None if args.no_prompt else SOCCER_PROMPT,
    )
    store = VocabStore(translate=not args.no_translate)
    windower = AudioWindower(device=device, window_sec=args.window, overlap_sec=args.overlap)

    _serve(out_dir, args.port)
    url = f"http://localhost:{args.port}/index.html"
    print(f"\nDashboard: {url}\nPlay the match (output routed through BlackHole). Ctrl-C to stop.\n")
    if not args.no_browser:
        webbrowser.open(url)

    try:
        for audio, is_first in windower.windows():
            words = transcriber.transcribe_words(audio)
            if not is_first:
                words = [(w, t) for (w, t) in words if t >= args.overlap]
            text = " ".join(w for w, _ in words)
            if store.add_text(text):
                store.write(vocab_path)
                print(f"  heard {store.total_tokens} words · {len(store.counts)} unique", end="\r")
    except KeyboardInterrupt:
        print("\nStopping. Your session is saved in", vocab_path)
    finally:
        windower.stop()


if __name__ == "__main__":
    main()
