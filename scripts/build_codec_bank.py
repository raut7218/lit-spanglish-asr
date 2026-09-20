"""Pre-render codec-degraded copies of the training clips so training needs no ffmpeg in the loop.

    python scripts/build_codec_bank.py --data_dir PREPARED [--variants 3] [--workers 10]

The test clips are MP3 64 kbps @ 48 kHz mono, i.e. WhatsApp voice notes (Opus) re-encoded as MP3. Each variant
mimics that chain: Opus (voip, 12-24 kbps) -> MP3 64k @ 48k -> back to 16 kHz. Writes clips_codec/*.flac and
train_bank.jsonl (train.jsonl + a `variants` list per clip). Resumable: existing files are skipped.
"""
import argparse
import json
import subprocess
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import soundfile as sf

CHAINS = [("16k",), ("24k",), (None,), ("12k",)]  # opus bitrate before the final mp3 (None = mp3 only)


def _ff(args, data):
    return subprocess.run(["ffmpeg", "-nostdin", "-v", "error", *args], input=data, capture_output=True, check=True, timeout=120).stdout


def render(x: np.ndarray, opus_br):
    cur = x.astype(np.float32).tobytes()
    fmt_in = ["-f", "f32le", "-ar", "16000", "-ac", "1", "-i", "-"]
    if opus_br:
        cur = _ff(fmt_in + ["-c:a", "libopus", "-b:a", opus_br, "-application", "voip", "-f", "ogg", "-"], cur)
        cur = _ff(["-i", "-", "-ar", "48000", "-ac", "1", "-c:a", "libmp3lame", "-b:a", "64k", "-f", "mp3", "-"], cur)
    else:
        cur = _ff(fmt_in + ["-ar", "48000", "-c:a", "libmp3lame", "-b:a", "64k", "-f", "mp3", "-"], cur)
    y = np.frombuffer(_ff(["-i", "-", "-ar", "16000", "-ac", "1", "-f", "f32le", "-"], cur), np.float32)
    return y[: len(x)] if len(y) >= len(x) else np.pad(y, (0, len(x) - len(y)))


def work(job):
    root, rel, cid, variants = job
    x, sr = sf.read(str(Path(root) / rel), dtype="float32")
    outs = []
    for k, (br,) in enumerate(variants):
        out_rel = f"clips_codec/{cid}_v{k}.flac"
        p = Path(root) / out_rel
        if not p.exists():
            try:
                y = render(x, br)
            except Exception as e:  # keep going; a missing variant is simply not used
                print(f"[bank] {cid} v{k} failed: {type(e).__name__}", file=sys.stderr)
                continue
            p.parent.mkdir(exist_ok=True)
            sf.write(p, y, sr, subtype="PCM_16")
        outs.append(out_rel)
    return cid, outs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--variants", type=int, default=3)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    root = Path(a.data_dir)
    rows = [json.loads(l) for l in open(root / "train.jsonl", encoding="utf-8")]
    todo = [r for r in rows if not r.get("nonspeech")]
    if a.limit:
        todo = todo[: a.limit]
    variants = CHAINS[: a.variants]
    with Pool(a.workers) as pool:
        res = dict(pool.imap_unordered(work, [(str(root), r["audio"], r["id"], variants) for r in todo], chunksize=8))
    with open(root / "train_bank.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            if res.get(r["id"]):
                r = dict(r, variants=res[r["id"]])
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[bank] {sum(1 for v in res.values() if v)} clips x up to {len(variants)} variants -> {root/'train_bank.jsonl'}")


if __name__ == "__main__":
    main()
