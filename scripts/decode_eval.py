"""Score runtime decoding configs (the exact faster-whisper path that ships) on dev / hold-out / synthetic long clips.

    python scripts/decode_eval.py --ct2 EXPORT/ct2 --data_dir PREP --sets dev long \
        --grid '[{"beam_size":1},{"beam_size":5},{"beam_size":5,"language":"en"}]' [--dump_dir DIR]

`long` = dev clips concatenated (0.4 s gaps) into 60-240 s clips: the test has clips up to ~4 min, dev none over 60 s,
so this is the only check of the VAD-chunked long-form path.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from lit.audio import SR, load_audio  # noqa: E402
from lit.infer import Transcriber  # noqa: E402
from lit.normalize import wer  # noqa: E402
from lit.postprocess import postprocess  # noqa: E402


def load_set(root: Path, name: str, max_clips: int):
    if name == "long":
        rows = [json.loads(l) for l in open(root / "dev.jsonl", encoding="utf-8")]
        rng, out, i = np.random.default_rng(0), [], 0
        while i < len(rows):
            target, a, refs = rng.uniform(60, 240), [], []
            while i < len(rows) and sum(len(x) for x in a) / SR < target:
                a += [load_audio(root / rows[i]["audio"]), np.zeros(int(0.4 * SR), np.float32)]
                refs.append(rows[i]["ref"])
                i += 1
            out.append((np.concatenate(a), " ".join(refs)))
        return out
    rows = [json.loads(l) for l in open(root / f"{name}.jsonl", encoding="utf-8")]
    rows = [r for r in rows if r.get("text", "").strip() and not r.get("nonspeech")]
    if name == "miami_holdout":
        rows = [r for r in rows if r.get("kind", "turn") == "turn"]
    rows = rows[:max_clips] if max_clips else rows
    return [(load_audio(root / r["audio"]), r.get("ref", r["text"])) for r in rows]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ct2", required=True)
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--sets", nargs="+", default=["dev"])
    ap.add_argument("--grid", default='[{}]', help="JSON list of infer-config overrides")
    ap.add_argument("--max_clips", type=int, default=300)
    ap.add_argument("--dump_dir")
    a = ap.parse_args()
    root = Path(a.data_dir)
    data = {s: load_set(root, s, a.max_clips) for s in a.sets}
    grid = json.loads(a.grid)
    t = Transcriber(a.ct2, grid[0])
    for g in grid:
        t.cfg.update(g)
        t.beam = t.cfg["beam_size"]
        res = {}
        for s, items in data.items():
            t0 = time.time()
            hyps = [postprocess(t.transcribe_array(x), None, t.cfg.get("rules")) for x, _ in items]
            res[s] = f"{wer([r for _, r in items], hyps):.4f} ({(time.time()-t0)/len(items):.2f}s/clip)"
            if a.dump_dir:
                import pandas as pd

                from lit.normalize import norm

                tag = "_".join(f"{k}{v}" for k, v in g.items()) or "default"
                pd.DataFrame(dict(id=range(len(items)), ref=[norm(r) for _, r in items], hyp=hyps)).to_csv(
                    Path(a.dump_dir) / f"{s}_{tag}.csv", index=False)
        print(f"[decode] {json.dumps(g)} -> {res}", flush=True)


if __name__ == "__main__":
    main()
