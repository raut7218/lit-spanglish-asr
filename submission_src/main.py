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
from lit.mbr import mbr_pick  # noqa: E402
from lit.postprocess import PLACEHOLDER, finalize_transcript  # noqa: E402

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


def write_csv(names, texts) -> None:
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = OUT_PATH.with_suffix(".tmp")
    try:
        import pandas as pd

        pd.DataFrame({"audio_filename": names, "transcript": texts}).to_csv(tmp, index=False, encoding="utf-8")
    except ImportError:  # same standard quoting, stdlib only
        with open(tmp, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f, lineterminator="\n")
            w.writerow(COLUMNS)
            w.writerows(zip(names, texts))
    tmp.replace(OUT_PATH)


def self_check(names) -> int:
    """Re-read the file the way a validator would (pandas defaults: '' / 'NA' / 'None' -> NaN) and repair
    anything a strict check could reject. Returns the number of repaired rows."""
    import pandas as pd

    df = pd.read_csv(OUT_PATH)
    assert list(df.columns) == COLUMNS, f"bad columns {list(df.columns)}"
    assert len(df) == len(names) and df["audio_filename"].tolist() == list(names), "row/order mismatch"
    bad = df["transcript"].isna()
    n_bad = int(bad.sum())
    if n_bad:
        df.loc[bad, "transcript"] = PLACEHOLDER
        df.to_csv(OUT_PATH, index=False, encoding="utf-8")
        assert not pd.read_csv(OUT_PATH)["transcript"].isna().any(), "could not repair NaN transcripts"
    return n_bad


def run_qwen(model_dir: Path, paths, cfg: dict) -> list[str]:
    """Qwen3-ASR via vLLM in a child process (the GPU is fully released when it exits). Its verbose output goes to a
    file, not our log (500-line cap); only the tail is shown, and only if it fails."""
    import json
    import subprocess
    import tempfile

    tmp = Path(tempfile.mkdtemp())
    (tmp / "clips.txt").write_text("\n".join(str(p) for p in paths), encoding="utf-8")
    qcfg = {**cfg.get("qwen", {}), "rules": cfg.get("rules", [])}
    with open(tmp / "qwen.log", "w") as log:
        rc = subprocess.run([sys.executable, "-m", "lit.qwen", "--model", str(model_dir), "--paths_file", str(tmp / "clips.txt"),
                             "--out", str(tmp / "hyps.json"), "--cfg", json.dumps(qcfg)], cwd=HERE, stdout=log,
                            stderr=subprocess.STDOUT, env=dict(os.environ, PYTHONPATH=str(HERE))).returncode
    if rc:
        tail = (tmp / "qwen.log").read_text(errors="replace").replace(str(DATA_DIR), "<data>").splitlines()[-15:]
        raise RuntimeError(f"qwen exit {rc}: " + " | ".join(l[:200] for l in tail))
    return json.loads((tmp / "hyps.json").read_text(encoding="utf-8"))


def main() -> None:
    t0 = time.time()
    names = read_clip_names()
    print(f"[main] {len(names)} clips", flush=True)

    lex_path = MODEL_DIR / "casing_lexicon.json"
    lexicon = load_lexicon(lex_path) if lex_path.exists() and os.environ.get("LIT_NO_CASING") != "1" else None
    cfg = load_cfg(MODEL_DIR)
    outs, budget = [], cfg.get("ensemble_budget_s", 4800)
    for s in cfg.get("systems", ["ct2"]):  # several fine-tuned models: per-clip MBR pick (lit.mbr); strongest first
        el = time.time() - t0
        if outs and el * (len(outs) + 1) / len(outs) > budget:  # the next model would not fit: ship what we have
            print(f"[main] time guard: stopping after {len(outs)} models ({el:.0f}s elapsed)", flush=True)
            break
        paths = [DATA_DIR / "clips" / n for n in names]
        if s.startswith("qwen"):
            try:
                outs.append(run_qwen(MODEL_DIR / s, paths, cfg))
                print(f"[main] {s} done at {time.time()-t0:.0f}s", flush=True)
            except Exception as e:  # never lose the submission to the new system: the Whisper systems still run
                print(f"[main] !!! {s} failed, skipping it: {str(e)[:250]}", flush=True)
            continue
        t = Transcriber(MODEL_DIR / s, cfg, lexicon)
        print(f"[main] model {len(outs) + 1} loaded on {t.device} in {time.time()-t0:.0f}s", flush=True)
        outs.append(transcribe_many(t, paths))
        del t
    texts = outs[0] if len(outs) == 1 else [mbr_pick(list(h)) for h in zip(*outs)]

    n_empty = sum(1 for x in texts if not " ".join(str(x or "").split()))
    texts = [finalize_transcript(x) for x in texts]
    write_csv(names, texts)
    fixed = self_check(names)
    print(f"[main] output check ok ({n_empty} empty predictions replaced by a placeholder, {fixed} rows repaired)", flush=True)
    print(f"[main] wrote {len(names)} rows in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
