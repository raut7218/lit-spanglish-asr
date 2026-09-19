"""Competition entrypoint: `uv run src/main.py` inside the offline runtime.

Reads /code_execution/data/submission_format.csv + clips/, transcribes every clip with the bundled
fine-tuned Whisper (CTranslate2), writes /code_execution/submission/submission.csv.
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


def main() -> None:
    t0 = time.time()
    with open(DATA_DIR / "submission_format.csv", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames
        rows = list(reader)
    text_col = next(c for c in fields if c != "audio_filename")
    print(f"[main] {len(rows)} clips; output column '{text_col}'", flush=True)

    lex_path = MODEL_DIR / "casing_lexicon.json"
    lexicon = load_lexicon(lex_path) if lex_path.exists() and os.environ.get("LIT_NO_CASING") != "1" else None
    t = Transcriber(MODEL_DIR / "ct2", load_cfg(MODEL_DIR), lexicon)
    print(f"[main] model loaded on {t.device} in {time.time()-t0:.0f}s", flush=True)

    paths = [DATA_DIR / "clips" / r["audio_filename"] for r in rows]
    texts = transcribe_many(t, paths)
    for r, txt in zip(rows, texts):
        r[text_col] = txt

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = OUT_PATH.with_suffix(".tmp")
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    tmp.replace(OUT_PATH)
    print(f"[main] wrote {len(rows)} rows to {OUT_PATH} in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
