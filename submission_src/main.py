"""Competition entrypoint: `uv run src/main.py` inside the offline runtime.

Reads /code_execution/data/test_metadata.csv (+ clips/), transcribes every clip independently with
the bundled fine-tuned Whisper (CTranslate2), writes /code_execution/submission/submission.csv with
exactly the columns `audio_filename,transcript`.

Logging is deliberately sparse and never prints clip names or transcripts (competition rule: no
information about the test data in the logs; logs are capped at 500 lines x 300 chars).
"""

import csv
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from lit.casing import load_lexicon  # noqa: E402
from lit.infer import Transcriber, load_cfg, transcribe_many  # noqa: E402

DATA_DIR = Path(os.environ.get("LIT_DATA_DIR", "/code_execution/data"))
OUT_PATH = Path(os.environ.get("LIT_SUBMISSION_PATH", "/code_execution/submission/submission.csv"))
MODEL_DIR = HERE / "model"
COLUMNS = ["audio_filename", "transcript"]


def read_clip_names() -> list[str]:
    """One entry per clip of the test set: test_metadata.csv is authoritative (submission_format.csv
    is accepted as a fallback, then a directory listing)."""
    for name in ("test_metadata.csv", "submission_format.csv"):
        p = DATA_DIR / name
        if p.exists():
            with open(p, newline="", encoding="utf-8") as f:
                names = [r["audio_filename"] for r in csv.DictReader(f)]
            if names:
                return names
    return sorted(p.name for p in (DATA_DIR / "clips").iterdir() if p.is_file())


def main() -> None:
    t0 = time.time()
    names = read_clip_names()
    print(f"[main] {len(names)} clips", flush=True)

    lex_path = MODEL_DIR / "casing_lexicon.json"
    lexicon = load_lexicon(lex_path) if lex_path.exists() and os.environ.get("LIT_NO_CASING") != "1" else None
    t = Transcriber(MODEL_DIR / "ct2", load_cfg(MODEL_DIR), lexicon)
    print(f"[main] model loaded on {t.device} in {time.time()-t0:.0f}s", flush=True)

    texts = transcribe_many(t, [DATA_DIR / "clips" / n for n in names])

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = OUT_PATH.with_suffix(".tmp")
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)  # standard CSV quoting: commas / quotes / newlines are quoted for us
        w.writerow(COLUMNS)
        for n, txt in zip(names, texts):
            w.writerow([n, " ".join(txt.split())])
    tmp.replace(OUT_PATH)
    print(f"[main] wrote {len(names)} rows in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
