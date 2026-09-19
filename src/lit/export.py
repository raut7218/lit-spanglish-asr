"""Merge LoRA into the base model, save HF weights, convert to CTranslate2 (faster-whisper) and
build the casing lexicon. Output dir is what goes into submission.zip.

    python -m lit.export --model openai/whisper-large-v3-turbo --adapter RUN/final_adapter \
        --data_dir DATA --out EXPORT [--quantization float16] [--include_dev_in_lexicon]
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch
from peft import PeftModel
from transformers import WhisperFeatureExtractor, WhisperForConditionalGeneration, WhisperTokenizerFast

from .casing import build_lexicon, save_lexicon
from .model_utils import read_manifest


def merge_and_save(model_name, adapter, hf_dir):
    model = WhisperForConditionalGeneration.from_pretrained(model_name, torch_dtype=torch.float32)
    if adapter:
        model = PeftModel.from_pretrained(model, adapter).merge_and_unload()
    model.config.use_cache = True
    model.save_pretrained(hf_dir, safe_serialization=True)
    WhisperFeatureExtractor.from_pretrained(model_name).save_pretrained(hf_dir)
    WhisperTokenizerFast.from_pretrained(model_name).save_pretrained(hf_dir)  # writes tokenizer.json


def convert_ct2(hf_dir, ct2_dir, quantization="float16"):
    from ctranslate2.converters import TransformersConverter

    TransformersConverter(str(hf_dir), copy_files=["tokenizer.json", "preprocessor_config.json"]).convert(
        str(ct2_dir), quantization=quantization, force=True
    )


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--adapter")
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--quantization", default="float16")
    ap.add_argument("--include_dev_in_lexicon", action="store_true")
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    hf_dir = out / "hf_merged"
    merge_and_save(a.model, a.adapter, hf_dir)
    convert_ct2(hf_dir, out / "ct2", a.quantization)
    root = Path(a.data_dir)
    texts = [r["text"] for r in read_manifest(root / "train.jsonl")]
    if a.include_dev_in_lexicon:
        texts += [r["text"] for r in read_manifest(root / "dev.jsonl")]
    lex = build_lexicon(texts)
    save_lexicon(lex, out / "casing_lexicon.json")
    meta = dict(base_model=a.model, adapter=str(a.adapter), quantization=a.quantization, lexicon_words=len(lex))
    (out / "export.json").write_text(json.dumps(meta, indent=2))
    print(f"[export] ct2 model -> {out/'ct2'}  lexicon words={len(lex)}")


if __name__ == "__main__":
    main()
