"""Shared model helpers: load Whisper (+LoRA), tokenisation, HF-generate transcription."""

from __future__ import annotations

import numpy as np
import torch
from transformers import WhisperFeatureExtractor, WhisperForConditionalGeneration, WhisperTokenizer

from .audio import load_audio, split_long
from .features import LogMel, pad_batch

LANG_NAME = {"es": "spanish", "en": "english"}


def pick_dtype(name: str):
    if name == "auto":
        if torch.cuda.is_available():
            return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        return torch.float32
    return dict(fp16=torch.float16, bf16=torch.bfloat16, fp32=torch.float32)[name]


def load_tokenizer(name_or_dir, language="es"):
    tok = WhisperTokenizer.from_pretrained(name_or_dir)
    tok.set_prefix_tokens(language=LANG_NAME[language], task="transcribe", predict_timestamps=False)
    return tok


def load_base(name_or_dir, dtype, device):
    model = WhisperForConditionalGeneration.from_pretrained(name_or_dir, torch_dtype=dtype)
    model.config.use_cache = False
    return model.to(device)


def encode_targets(tok, text: str, max_len: int = 440):
    """Full decoder sequence: <sot><lang><task><notimestamps> text <eot>. Returns (dec_in, labels).

    labels for the 3 forced prefix positions (lang/task/notimestamps) are masked with -100.
    """
    ids = tok(text.strip()).input_ids  # includes prefix tokens + eot with set_prefix_tokens
    ids = ids[:max_len]
    if ids[-1] != tok.eos_token_id:
        ids[-1] = tok.eos_token_id
    dec_in, labels = ids[:-1], ids[1:]
    labels = list(labels)
    labels[:3] = [-100, -100, -100]
    return dec_in, labels


@torch.no_grad()
def transcribe_hf(model, tok, fe, audios: list[np.ndarray], device, language="es", batch_size=8,
                  num_beams=1, max_new_tokens=220, amp_dtype=None):
    """Greedy/beam transcription of long-or-short arrays with HF generate (chunks >29 s, then re-joins)."""
    mel = LogMel(fe, device)
    pieces, owner = [], []
    for i, a in enumerate(audios):
        for c in split_long(a):
            pieces.append(c)
            owner.append(i)
    outs = [""] * len(pieces)
    order = np.argsort([-len(p) for p in pieces])
    model.eval()
    use_cache_prev = model.config.use_cache
    model.config.use_cache = True
    for s in range(0, len(order), batch_size):
        idx = order[s : s + batch_size]
        feats = mel(pad_batch([pieces[i] for i in idx]))
        if amp_dtype is not None:
            feats = feats.to(amp_dtype)
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_dtype is not None and device.type == "cuda"):
            gen = model.generate(
                input_features=feats, language=LANG_NAME[language], task="transcribe", return_timestamps=False,
                num_beams=num_beams, max_new_tokens=max_new_tokens, no_repeat_ngram_size=0,
            )
        for i, text in zip(idx, tok.batch_decode(gen, skip_special_tokens=True)):
            outs[i] = text.strip()
    model.config.use_cache = use_cache_prev
    merged = [""] * len(audios)
    for o, t in zip(owner, outs):
        merged[o] = (merged[o] + " " + t).strip()
    return merged


def read_manifest(path):
    import json

    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def load_clip_arrays(rows, root):
    from pathlib import Path

    return [load_audio(Path(root) / r["audio"]) for r in rows]
