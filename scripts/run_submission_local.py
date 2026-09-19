"""Run a submission.zip the way the platform does (unzip -> main.py) against a local data dir, then score.

    python scripts/run_submission_local.py --zip submission.zip --prepared PREPARED_DIR [--n 155]

Builds /tmp-style data dir from the dev set: clips/<orig>.mp3 + submission_format.csv, runs
`python src/main.py` with LIT_DATA_DIR / LIT_SUBMISSION_PATH pointed at it, scores WER with the
organisers' normaliser (scripts/official_score.py logic) and validates the CSV shape.
"""

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from lit.normalize import wer  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", required=True)
    ap.add_argument("--prepared", required=True)
    ap.add_argument("--dev_raw", required=True, help="original enspa_dev dir (mp3 clips as the platform provides them)")
    ap.add_argument("--n", type=int, default=0)
    ap.add_argument("--python", default=sys.executable)
    a = ap.parse_args()

    rows = [json.loads(l) for l in open(Path(a.prepared) / "dev.jsonl", encoding="utf-8")]
    if a.n:
        rows = rows[: a.n]
    work = Path(tempfile.mkdtemp())
    data, sub, src = work / "data", work / "submission", work / "src"
    (data / "clips").mkdir(parents=True)
    sub.mkdir()
    for r in rows:
        shutil.copy(Path(a.dev_raw) / "clips" / r["orig"], data / "clips" / r["orig"])
    with open(data / "submission_format.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["audio_filename", "transcript"])
        for r in rows:
            w.writerow([r["orig"], "hello world"])
    with zipfile.ZipFile(a.zip) as z:
        assert "main.py" in z.namelist(), "main.py not at zip root"
        z.extractall(src)

    env = dict(os.environ, LIT_DATA_DIR=str(data), LIT_SUBMISSION_PATH=str(sub / "submission.csv"))
    t0 = time.time()
    subprocess.run([a.python, str(src / "main.py")], check=True, env=env, cwd=work)
    el = time.time() - t0

    out = list(csv.DictReader(open(sub / "submission.csv", encoding="utf-8")))
    assert list(out[0].keys()) == ["audio_filename", "transcript"], out[0].keys()
    assert [o["audio_filename"] for o in out] == [r["orig"] for r in rows], "row order / count mismatch"
    hyp = {o["audio_filename"]: o["transcript"] for o in out}
    score = wer([r["ref"] for r in rows], [hyp[r["orig"]] for r in rows])
    empty = sum(1 for o in out if not o["transcript"].strip())
    print(f"[run_submission_local] rows={len(out)} empty={empty} time={el:.0f}s ({el/len(rows):.2f}s/clip) dev WER={score:.4f}")
    for r in rows[:3]:
        print("  REF:", r["ref"][:120], "\n  HYP:", hyp[r["orig"]][:120])
    shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    main()
