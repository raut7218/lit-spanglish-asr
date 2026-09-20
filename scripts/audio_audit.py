"""Compare acoustic statistics of two sets of prepared FLAC clips (e.g. dev voice notes vs Miami train).

    python scripts/audio_audit.py --a data/dev.jsonl --b data/train.jsonl [--n 150] [--root DIR]

Prints median/percentile loudness, noise floor, band-energy fractions, 95% spectral roll-off and clipping
so `augment.py` can be calibrated to make Miami look like the test domain (WhatsApp voice notes).
"""
import argparse
import json
import random
from pathlib import Path

import numpy as np
import soundfile as sf

BANDS = [(0, 500), (500, 1000), (1000, 2000), (2000, 4000), (4000, 6000), (6000, 8000)]


def stats(x, sr=16000):
    x = x.astype(np.float64)
    fr = 400
    n = len(x) // fr
    if n < 5:
        return None
    rms = np.sqrt((x[: n * fr].reshape(n, fr) ** 2).mean(1) + 1e-12)
    db = 20 * np.log10(rms)
    spec = np.abs(np.fft.rfft(x[: (len(x) // 512) * 512].reshape(-1, 512) * np.hanning(512), axis=1)) ** 2
    ps = spec.mean(0)
    f = np.fft.rfftfreq(512, 1 / sr)
    tot = ps.sum() + 1e-12
    bands = [ps[(f >= lo) & (f < hi)].sum() / tot for lo, hi in BANDS]
    cs = np.cumsum(ps) / tot
    return dict(
        loud_db=20 * np.log10(np.sqrt((x ** 2).mean()) + 1e-12),
        peak_db=20 * np.log10(np.abs(x).max() + 1e-12),
        floor_db=np.percentile(db, 10),
        dyn_db=np.percentile(db, 90) - np.percentile(db, 10),
        rolloff95=f[np.searchsorted(cs, 0.95)],
        clip_frac=float((np.abs(x) > 0.98).mean()),
        **{f"band{lo}-{hi}": b for (lo, hi), b in zip(BANDS, bands)},
    )


def load(manifest, root, n, seed=0):
    rows = [json.loads(l) for l in open(manifest, encoding="utf-8")]
    random.Random(seed).shuffle(rows)
    out = []
    for r in rows[:n]:
        x, sr = sf.read(Path(root) / r["audio"], dtype="float32")
        s = stats(x, sr)
        if s:
            out.append(s)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--root_a")
    ap.add_argument("--root_b")
    ap.add_argument("--n", type=int, default=150)
    a = ap.parse_args()
    A = load(a.a, a.root_a or Path(a.a).parent, a.n)
    B = load(a.b, a.root_b or Path(a.b).parent, a.n)
    print(f"{'stat':16s} {'A p10/p50/p90':>26s} {'B p10/p50/p90':>26s}")
    for k in A[0]:
        pa = np.percentile([s[k] for s in A], [10, 50, 90])
        pb = np.percentile([s[k] for s in B], [10, 50, 90])
        print(f"{k:16s} {pa[0]:8.3f} {pa[1]:8.3f} {pa[2]:8.3f}   {pb[0]:8.3f} {pb[1]:8.3f} {pb[2]:8.3f}")


if __name__ == "__main__":
    main()
