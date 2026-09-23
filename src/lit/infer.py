"""Runtime inference with the fine-tuned Canary (lit.canary: torch + transformers' ParakeetEncoder, NO NeMo).
This module is copied into submission.zip and must run in the offline competition image.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from .postprocess import postprocess

SR = 16000

DEFAULT_CFG = dict(
    languages=["es", "en"],  # decode with each prompt, keep the best length-normalised log-prob
    beam_size=4,
    batch_size=16,
    group_size=64,  # clips decoded per transcribe() call (length-sorted inside)
    compute_type="auto",  # bf16 if supported, else fp16 (autocast over fp32 weights)
    max_s=40.0,  # longer clips are split at quiet points
    tokens_per_s=8.0,
    rules=[],  # names from lit.rules.RULES applied after normalisation
    time_budget_s=6000,  # degrade (beam 1, first language only) if the projected total exceeds this
)


def load_cfg(model_dir) -> dict:
    cfg = dict(DEFAULT_CFG)
    p = Path(model_dir) / "infer_config.json"
    if p.exists():
        cfg.update(json.loads(p.read_text()))
    return cfg


def _safe_err(e: Exception, path) -> str:
    """Exception summary for logs with the clip path/name scrubbed (no test-data info in logs)."""
    msg = str(e).replace(str(path), "<clip>").replace(Path(str(path)).name, "<clip>")
    return f"{type(e).__name__}: {msg[:160]}"


def decode_audio(path) -> np.ndarray:
    try:  # PyAV via faster-whisper: what the competition image ships and the previous submission used
        from faster_whisper.audio import decode_audio as fw_decode

        return fw_decode(str(path), sampling_rate=SR).astype(np.float32)
    except ImportError:
        import librosa

        return librosa.load(str(path), sr=SR, mono=True)[0].astype(np.float32)


def _probe_audio():
    t = np.arange(SR) / SR
    return (0.05 * np.sin(2 * np.pi * 200 * t) + 0.01 * np.random.default_rng(0).standard_normal(SR)).astype(np.float32)


class Transcriber:
    def __init__(self, model_dir, cfg: dict | None = None, lexicon: dict | None = None, device: str = "auto"):
        import torch

        self.cfg = {**DEFAULT_CFG, **(cfg or {})}
        self.lex = lexicon
        self.model_dir = Path(model_dir)
        self.beam, self.languages = self.cfg["beam_size"], list(self.cfg["languages"])
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        try:
            self._build(device)
            self.transcribe_arrays([_probe_audio()])  # a real decode: surfaces CUDA problems now, not mid-run
        except Exception as e:
            if device == "cpu":
                raise
            print(f"[infer] !!! GPU path failed ({type(e).__name__}); FALLING BACK TO CPU (slow but correct)", flush=True)
            self._build("cpu")
            self.transcribe_arrays([_probe_audio()])
        print(f"[infer] ready on {self.device} (autocast {self.amp}), beam {self.beam}, languages {self.languages}", flush=True)

    def _build(self, device):
        import torch

        from . import canary

        self.torch = torch
        self.model, self.tok = canary.load(self.model_dir, device, torch.float32)
        ct = self.cfg["compute_type"]
        self.amp = None if device == "cpu" else (canary.pick_dtype("auto") if ct == "auto" else canary.pick_dtype(ct))
        self.device = device

    def transcribe_arrays(self, audios: list[np.ndarray]) -> list[str]:
        from . import canary

        keep = [i for i, a in enumerate(audios) if len(a) >= 0.2 * SR]
        out = [""] * len(audios)
        if keep:
            texts = canary.transcribe(self.model, self.tok, [audios[i] for i in keep], self.languages, self.beam,
                                      self.cfg["batch_size"], self.cfg["max_s"], self.cfg["tokens_per_s"], self.amp)
            for i, t in zip(keep, texts):
                out[i] = t
        return [postprocess(t, self.lex, self.cfg.get("rules")) for t in out]


def transcribe_many(t: Transcriber, paths, log_every=100):
    n, g = len(paths), t.cfg["group_size"]
    out, t0, failed = [""] * n, time.time(), 0
    budget = t.cfg["time_budget_s"]
    for s in range(0, n, g):
        idx = list(range(s, min(n, s + g)))
        audios = []
        for i in idx:
            try:
                audios.append(decode_audio(paths[i]))
            except Exception as e:  # unreadable clip: empty prediction, counted
                failed += 1
                if failed <= 5:
                    print(f"[infer] clip {i} unreadable ({_safe_err(e, paths[i])})", flush=True)
                audios.append(np.zeros(0, np.float32))
        try:
            texts = t.transcribe_arrays(audios)
        except Exception as e:  # never lose a whole group: retry clip by clip
            print(f"[infer] group decode failed ({type(e).__name__}); retrying clip by clip", flush=True)
            texts = []
            for i, a in zip(idx, audios):
                try:
                    texts.append(t.transcribe_arrays([a])[0])
                except Exception as e2:
                    failed += 1
                    if failed <= 5:
                        print(f"[infer] clip {i} FAILED ({_safe_err(e2, paths[i])})", flush=True)
                    texts.append("")
        for i, x in zip(idx, texts):
            out[i] = x
        done, el = idx[-1] + 1, time.time() - t0
        if (t.beam > 1 or len(t.languages) > 1) and el / done * n > budget:
            print(f"[infer] projected {el/done*n:.0f}s > budget {budget}s: switching to beam 1, {t.languages[0]} only", flush=True)
            t.beam, t.languages = 1, t.languages[:1]
        if done // log_every != (done - len(idx)) // log_every or done == n:
            print(f"[infer] {done}/{n} clips, {el:.0f}s elapsed, {failed} failed", flush=True)
    if failed > max(2, 0.05 * n):  # systemic problem: a mostly-empty CSV would only score ~1.0 WER, so fail loudly
        raise RuntimeError(f"{failed}/{n} clips failed to transcribe")
    return out
