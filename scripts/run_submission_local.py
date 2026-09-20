"""Validate a submission.zip against the competition's submission guidelines, then score it on dev.

    python scripts/run_submission_local.py --zip submission.zip --prepared PREPARED --dev_raw ENSPA_DEV [--n 0]

Simulates the platform layout in a temp dir (data/clips + data/test_metadata.csv only, src/ = unzipped
zip, submission/ output), runs `python src/main.py`, and checks every rule from the "Code submission
format" page: main.py at the zip root, weights bundled, exact output columns / one row per clip /
standard CSV quoting, <=500 log lines of <=300 chars, no clip names in the logs, runtime.
"""

import argparse
import csv
import json
import os
import re
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

FAILS = []


def check(ok, msg):
    print(("  ok   " if ok else "  FAIL ") + msg)
    if not ok:
        FAILS.append(msg)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", required=True)
    ap.add_argument("--prepared", required=True)
    ap.add_argument("--dev_raw", required=True, help="original enspa_dev dir (mp3 clips as the platform provides them)")
    ap.add_argument("--n", type=int, default=0)
    ap.add_argument("--python", default=sys.executable)
    a = ap.parse_args()

    print("== zip structure")
    with zipfile.ZipFile(a.zip) as z:
        names = z.namelist()
        check("main.py" in names, "main.py is at the archive root (not nested in a folder)")
        check(not any(n.startswith("submission/") for n in names), "zip does not contain a submission/ dir")
        weights = [n for n in names if n.endswith("model.bin")]
        check(bool(weights), f"model weights bundled ({', '.join(weights)})")
        src_main = z.read("main.py").decode()
        check("/code_execution/data" in src_main and "/code_execution/submission/submission.csv" in src_main,
              "main.py defaults point to /code_execution/data and /code_execution/submission/submission.csv")
        check("test_metadata.csv" in src_main, "main.py reads test_metadata.csv")
        check(not re.search(r"https?://|from_pretrained\(\s*['\"][\w-]+/", src_main), "main.py does not need the network")
    size = Path(a.zip).stat().st_size / 1e9
    print(f"  info zip size {size:.2f} GB, {len(names)} files")

    rows = [json.loads(l) for l in open(Path(a.prepared) / "dev.jsonl", encoding="utf-8")]
    if a.n:
        rows = rows[: a.n]
    work = Path(tempfile.mkdtemp())
    data, sub, src = work / "data", work / "submission", work / "src"
    (data / "clips").mkdir(parents=True)
    sub.mkdir()
    for r in rows:
        shutil.copy(Path(a.dev_raw) / "clips" / r["orig"], data / "clips" / r["orig"])
    with open(data / "test_metadata.csv", "w", newline="", encoding="utf-8") as f:  # the ONLY manifest, as on the platform
        w = csv.writer(f)
        w.writerow(["audio_filename", "file_duration_seconds", "language"])
        for r in rows:
            w.writerow([r["orig"], int(round(r["duration"])), "enspa"])
    with zipfile.ZipFile(a.zip) as z:
        z.extractall(src)

    print("== run main.py")
    env = dict(os.environ, LIT_DATA_DIR=str(data), LIT_SUBMISSION_PATH=str(sub / "submission.csv"))
    t0 = time.time()
    p = subprocess.run([a.python, str(src / "main.py")], env=env, cwd=work, capture_output=True, text=True)
    el = time.time() - t0
    log = (p.stdout + p.stderr).splitlines()
    check(p.returncode == 0, f"main.py exited with code {p.returncode}")
    if p.returncode != 0:
        print("\n".join(log[-25:]))
        sys.exit(1)

    print("== output")
    with open(sub / "submission.csv", newline="", encoding="utf-8") as f:
        rd = list(csv.reader(f))
    check(rd[0] == ["audio_filename", "transcript"], f"header is exactly audio_filename,transcript (got {rd[0]})")
    check(all(len(r) == 2 for r in rd[1:]), "every row has exactly 2 fields (quoting is standard)")
    out = {r[0]: r[1] for r in rd[1:]}
    check(len(rd) - 1 == len(rows) and set(out) == {r["orig"] for r in rows}, f"one row per clip in test_metadata.csv ({len(rows)})")
    empty = sum(1 for v in out.values() if not v.strip())
    check(empty <= 0.5 * len(out), f"non-empty transcripts ({empty}/{len(out)} empty)")
    try:
        import pandas as pd

        df = pd.read_csv(sub / "submission.csv", keep_default_na=False)
        check(list(df.columns) == ["audio_filename", "transcript"] and len(df) == len(rows), "pandas.read_csv round-trips the file")
    except ImportError:
        pass
    tricky = sum(1 for v in out.values() if any(c in v for c in ',"\n'))
    print(f"  info {tricky} transcripts contain comma/quote/newline (must be quoted; csv module handles it)")

    print("== logs")
    check(len(log) <= 500, f"log has {len(log)} lines (limit 500)")
    check(max((len(l) for l in log), default=0) <= 300, f"longest log line {max((len(l) for l in log), default=0)} chars (limit 300)")
    leak = [r["orig"] for r in rows if Path(r["orig"]).stem in "\n".join(log)]
    check(not leak, "no clip file names appear in the logs")
    check(not any(v and v in "\n".join(log) for v in list(out.values())[:20] if len(v) > 12), "no transcripts appear in the logs")

    score = wer([r["ref"] for r in rows], [out[r["orig"]] for r in rows])
    print(f"== result: rows={len(rows)} time={el:.0f}s ({el/len(rows):.2f}s/clip) dev WER={score:.4f}")
    for r in rows[:2]:
        print("  REF:", r["ref"][:110], "\n  HYP:", out[r["orig"]][:110])
    shutil.rmtree(work, ignore_errors=True)
    if FAILS:
        print("\nVALIDATION FAILED:", *FAILS, sep="\n  - ")
        sys.exit(1)
    print("\nVALIDATION PASSED")


if __name__ == "__main__":
    main()
