"""Evaluate a Whisper model (base or base+LoRA adapter) on dev / Miami-holdout with scorer-exact WER.

    python -m lit.evaluate --model openai/whisper-small --data_dir DATA [--adapter DIR] [--split dev] [--language es]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from peft import PeftModel
from transformers import WhisperFeatureExtractor

from .casing import apply_casing, build_lexicon
from .model_utils import load_base, load_clip_arrays, load_tokenizer, pick_dtype, read_manifest, transcribe_hf
from .normalize import norm, wer
from .postprocess import postprocess


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--adapter")
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--split", default="dev", choices=["dev", "miami_holdout"])
    ap.add_argument("--language", default="es")
    ap.add_argument("--beams", type=int, default=1)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--max_clips", type=int, default=0)
    ap.add_argument("--dtype", default="auto")
    ap.add_argument("--dump", help="write predictions csv here")
    a = ap.parse_args(argv)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = pick_dtype(a.dtype) if device.type == "cuda" else torch.float32
    root = Path(a.data_dir)
    rows = read_manifest(root / f"{a.split}.jsonl")
    if a.max_clips:
        rows = rows[: a.max_clips]
    fe = WhisperFeatureExtractor.from_pretrained(a.model)
    tok = load_tokenizer(a.model, a.language)
    model = load_base(a.model, dtype, device)
    if a.adapter:
        model = PeftModel.from_pretrained(model, a.adapter).merge_and_unload()
    hyps = transcribe_hf(model, tok, fe, load_clip_arrays(rows, root), device, a.language, a.batch_size, a.beams,
                         amp_dtype=dtype if device.type == "cuda" else None)
    refs = [r.get("ref", r["text"]) for r in rows]
    res = dict(split=a.split, n=len(rows), wer_raw=wer(refs, hyps), wer_post=wer(refs, [postprocess(h) for h in hyps]))
    # casing lexicon learned from TRAIN transcripts only (honest for dev)
    train_texts = [r["text"] for r in read_manifest(root / "train.jsonl")]
    lex = build_lexicon(train_texts)
    res["wer_post_casing"] = wer(refs, [postprocess(h, lex) for h in hyps])
    print(json.dumps(res, indent=2))
    if a.dump:
        import pandas as pd

        pd.DataFrame(dict(id=[r["id"] for r in rows], ref=[norm(r) for r in refs], hyp=[norm(h) for h in hyps])).to_csv(a.dump, index=False)
    return res


if __name__ == "__main__":
    main()
