"""Per-clip quality audit: Silero-VAD speech ratio / silences, VAD-based SNR, and a zero-shot Whisper check of the
label (catches CHAT timing misalignments: the words of the label are not in the audio).

    python -m lit.quality --data_dir PREP --splits train miami_holdout dev --model openai/whisper-large-v3-turbo

Writes PREP/quality.jsonl (one row per clip id) and prints percentile tables per split/kind. lit.train reads the
file and drops clips failing `quality.*` thresholds.
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from .audio import SR, load_audio

FR = 400  # 25 ms frames


def vad_stats(audio: np.ndarray) -> dict:
    from faster_whisper.vad import VadOptions, get_speech_timestamps

    n = len(audio) // FR
    if n < 8:
        return dict(speech_ratio=0.0, lead_s=0.0, trail_s=0.0, speech_db=-99.0, snr_db=0.0)
    segs = get_speech_timestamps(audio, VadOptions(threshold=0.5, min_silence_duration_ms=300, speech_pad_ms=60))
    db = 10 * np.log10((audio[: n * FR].reshape(n, FR).astype(np.float64) ** 2).mean(1) + 1e-10)
    sp = np.zeros(n, bool)
    for s in segs:
        sp[s["start"] // FR : s["end"] // FR + 1] = True
    sp = sp[:n]
    speech_db = float(np.median(db[sp])) if sp.sum() >= 4 else float(np.percentile(db, 90))
    noise_db = float(np.median(db[~sp])) if (~sp).sum() >= 8 else float(np.percentile(db, 5))  # no pauses: quietest frames
    dur = len(audio) / SR
    return dict(speech_ratio=round(float(sp.mean()), 3),
                lead_s=round(segs[0]["start"] / SR if segs else dur, 2),
                trail_s=round(dur - segs[-1]["end"] / SR if segs else dur, 2),
                speech_db=round(speech_db, 1), snr_db=round(speech_db - noise_db, 1))


def _job(args):
    root, r = args
    return r["id"], vad_stats(load_audio(Path(root) / r["audio"]))


def summarize(rows, keys=("duration", "speech_ratio", "lead_s", "trail_s", "speech_db", "snr_db", "zs_wer", "len_ratio")):
    groups = {}
    for r in rows:
        groups.setdefault(f"{r['split']}/{r.get('kind', '-')}", []).append(r)
    for g, rs in sorted(groups.items()):
        print(f"\n== {g}: {len(rs)} clips")
        for k in keys:
            v = [r[k] for r in rs if r.get(k) is not None]
            if v:
                p = np.percentile(v, [5, 25, 50, 75, 95])
                print(f"  {k:13s} p5 {p[0]:7.2f}  p25 {p[1]:7.2f}  p50 {p[2]:7.2f}  p75 {p[3]:7.2f}  p95 {p[4]:7.2f}")
        if any("zs_wer" in r for r in rs):  # does label quality track the acoustics?
            for lo, hi in ((-99, 5), (5, 10), (10, 15), (15, 25), (25, 99)):
                b = [r["zs_wer"] for r in rs if lo <= r["snr_db"] < hi and "zs_wer" in r]
                if b:
                    print(f"  snr [{lo:3d},{hi:3d}) n={len(b):5d} zero-shot WER median {np.median(b):.3f}  >0.6: {np.mean(np.array(b) > 0.6):.1%}")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--splits", nargs="+", default=["train", "miami_holdout", "dev"])
    ap.add_argument("--model", default="openai/whisper-large-v3-turbo", help="zero-shot label check; 'none' to skip")
    ap.add_argument("--language", default="en")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--max_clips", type=int, default=0, help="per split, for a quick look")
    a = ap.parse_args(argv)
    root = Path(a.data_dir)
    from .model_utils import read_manifest

    rows = []
    for s in a.splits:
        rs = read_manifest(root / f"{s}.jsonl")
        rs = rs[: a.max_clips] if a.max_clips else rs
        rows += [dict(r, split=s) for r in rs]
    with ProcessPoolExecutor(a.workers) as ex:
        stats = dict(ex.map(_job, [(root, r) for r in rows], chunksize=32))
    out = []
    for r in rows:
        out.append(dict(id=r["id"], split=r["split"], kind=r.get("kind", r["split"]), duration=r["duration"],
                        text=r.get("ref", r.get("text", "")), **stats[r["id"]]))

    if a.model != "none":
        import torch
        from transformers import WhisperFeatureExtractor

        from .model_utils import load_base, load_clip_arrays, load_tokenizer, transcribe_hf
        from .normalize import norm, wer
        from .postprocess import postprocess

        dev = torch.device("cuda")
        fe, tok = WhisperFeatureExtractor.from_pretrained(a.model), load_tokenizer(a.model, a.language)
        model = load_base(a.model, torch.bfloat16, dev)
        speech = [i for i, r in enumerate(rows) if r.get("text", "").strip()]
        order = sorted(speech, key=lambda i: -rows[i]["duration"])  # length-sorted batches: little padding waste
        for s in range(0, len(order), 2048):
            idx = order[s : s + 2048]
            hyps = transcribe_hf(model, tok, fe, load_clip_arrays([rows[i] for i in idx], root), dev, a.language,
                                 a.batch_size, 1, amp_dtype=torch.bfloat16)
            for i, h in zip(idx, hyps):
                ref, hyp = norm(out[i]["text"]), postprocess(h)
                out[i].update(zs_hyp=hyp, zs_wer=round(min(2.0, wer([ref], [hyp])), 3),
                              len_ratio=round(len(hyp.split()) / max(1, len(ref.split())), 2))
            print(f"[quality] zero-shot {min(s + 2048, len(order))}/{len(order)}", flush=True)

    with open(root / "quality.jsonl", "w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    summarize(out)
    print(f"\n[quality] wrote {root/'quality.jsonl'} ({len(out)} clips)")


if __name__ == "__main__":
    main()
