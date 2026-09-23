"""Evaluate a Canary model dir (converted base or lit.train's final_model) with scorer-exact WER.

    python -m lit.evaluate --model DIR --data_dir DATA [--split dev] [--languages es en] [--beams 1] [--no_verbatim]

--no_verbatim drops the <|verbatim|> style token from the prompt: use it for the zero-shot base model (gate G1),
which has never seen that token.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from . import canary
from .audio import load_clip_arrays, read_manifest
from .casing import build_lexicon
from .normalize import norm, wer
from .postprocess import postprocess


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--split", default="dev", choices=["dev", "dev_spk1", "dev_spk2", "miami_holdout"])
    ap.add_argument("--kind", default="all", choices=["all", "turn", "window"], help="hold-out clip kind")
    ap.add_argument("--languages", nargs="+", default=["es", "en"])
    ap.add_argument("--no_verbatim", action="store_true")
    ap.add_argument("--beams", type=int, default=1)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--max_clips", type=int, default=0)
    ap.add_argument("--dtype", default="auto")
    ap.add_argument("--dump", help="write predictions csv here")
    a = ap.parse_args(argv)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = canary.pick_dtype(a.dtype) if device.type == "cuda" else None
    root = Path(a.data_dir)
    rows = read_manifest(root / f"{a.split}.jsonl")
    rows = [r for r in rows if r.get("text", "").strip() and not r.get("nonspeech")]  # WER needs a non-empty reference
    if a.kind != "all":
        rows = [r for r in rows if r.get("kind", "turn") == a.kind]
    if a.max_clips:
        rows = rows[: a.max_clips]
    model, tok = canary.load(a.model, device, torch.float32)
    if a.no_verbatim:
        tok.verbatim = None
    hyps = canary.transcribe(model, tok, load_clip_arrays(rows, root), a.languages, a.beams, a.batch_size, amp_dtype=amp)
    refs = [r.get("ref", r["text"]) for r in rows]
    res = dict(split=a.split, kind=a.kind, n=len(rows), languages=a.languages, beams=a.beams, verbatim=tok.verbatim,
               wer_raw=wer(refs, hyps), wer_post=wer(refs, [postprocess(h) for h in hyps]))
    # casing lexicon learned from TRAIN transcripts only (honest for dev)
    lex = build_lexicon([r["text"] for r in read_manifest(root / "train.jsonl")])
    res["wer_post_casing"] = wer(refs, [postprocess(h, lex) for h in hyps])
    print(json.dumps(res, indent=2))
    if a.dump:
        import pandas as pd

        pd.DataFrame(dict(id=[r["id"] for r in rows], ref=[norm(r) for r in refs], hyp=[postprocess(h) for h in hyps])).to_csv(a.dump, index=False)
    return res


if __name__ == "__main__":
    main()
