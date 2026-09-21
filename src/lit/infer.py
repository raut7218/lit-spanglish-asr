"""Runtime inference with faster-whisper (CTranslate2). NO torch / transformers / NeMo imports:
this module is copied into submission.zip and must run in the offline competition image.
"""

from __future__ import annotations

import ctypes
import glob
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

from .postprocess import postprocess

SR = 16000
MAX_DIRECT_S = 29.5

DEFAULT_CFG = dict(
    language="es",
    beam_size=5,
    batch_size=16,
    compute_type="float16",
    vad_threshold=0.4,
    vad_min_silence_ms=1500,
    vad_speech_pad_ms=300,
    no_speech_threshold=0.6,
    log_prob_threshold=-1.0,
    compression_ratio_threshold=2.4,
    repetition_penalty=1.0,
    no_repeat_ngram_size=0,
    max_new_tokens_per_s=8.0,
    extra={},  # extra faster-whisper transcribe() kwargs, e.g. {"length_penalty": 1.2, "patience": 2.0, "initial_prompt": "..."}
    rules=[],  # names from lit.rules.RULES applied after normalisation
    time_budget_s=6000,  # degrade to greedy if the projected total exceeds this
)


def preload_cuda12_libs():
    """ctranslate2 4.x needs CUDA-12 cuBLAS/cuDNN. In the competition image they arrive as pip wheels
    (nvidia-cublas-cu12, nvidia-cudnn-cu12 via tensorflow[and-cuda]) that are NOT on the loader path,
    so dlopen them by absolute path (RTLD_GLOBAL) before ctranslate2 first touches the GPU."""
    loaded = []
    roots = [p for p in sys.path if p.endswith("site-packages") or p.endswith("dist-packages")]
    for root in roots:
        for pat in ("nvidia/cuda_runtime/lib/libcudart.so.12", "nvidia/cublas/lib/libcublasLt.so.12",
                    "nvidia/cublas/lib/libcublas.so.12", "nvidia/cudnn/lib/libcudnn.so.9",
                    "nvidia/cuda_nvrtc/lib/libnvrtc.so.12"):
            for f in glob.glob(os.path.join(root, pat)):
                try:
                    ctypes.CDLL(f, mode=ctypes.RTLD_GLOBAL)
                    loaded.append(os.path.basename(f))
                except OSError:
                    pass
    return loaded


def load_cfg(model_dir) -> dict:
    cfg = dict(DEFAULT_CFG)
    p = Path(model_dir) / "infer_config.json"
    if p.exists():
        cfg.update(json.loads(p.read_text()))
    return cfg


_WARNED = [0]


def _safe_err(e: Exception, path) -> str:
    """Exception summary for logs with the clip path/name scrubbed (no test-data info in logs)."""
    msg = str(e).replace(str(path), "<clip>").replace(Path(str(path)).name, "<clip>")
    return f"{type(e).__name__}: {msg[:160]}"


def _probe_audio():
    t = np.arange(SR) / SR
    return (0.05 * np.sin(2 * np.pi * 200 * t) + 0.01 * np.random.default_rng(0).standard_normal(SR)).astype(np.float32)


class Transcriber:
    def __init__(self, ct2_dir, cfg: dict | None = None, lexicon: dict | None = None, device: str = "auto"):
        print(f"[infer] preloaded CUDA libs: {preload_cuda12_libs()}", flush=True)
        from faster_whisper import BatchedInferencePipeline, WhisperModel

        self.cfg = {**DEFAULT_CFG, **(cfg or {})}
        self.lex = lexicon
        if device == "auto":
            try:
                import ctranslate2

                device = "cuda" if ctranslate2.get_cuda_device_count() > 0 else "cpu"
            except Exception:
                device = "cpu"
        self.beam = self.cfg["beam_size"]
        self.ct2_dir = ct2_dir
        self.device = device
        try:
            self._build(device)
            self.transcribe_array(_probe_audio())  # a real GPU decode: surfaces missing/incompatible CUDA libs now
        except Exception as e:
            if device == "cpu":
                raise
            print(f"[infer] !!! GPU path failed ({e!r}); FALLING BACK TO CPU int8 (slow but correct)", flush=True)
            self._build("cpu")
            self.transcribe_array(_probe_audio())
        print(f"[infer] ready on {self.device}", flush=True)

    def _build(self, device):
        from faster_whisper import BatchedInferencePipeline, WhisperModel

        compute = self.cfg["compute_type"] if device == "cuda" else "int8"
        self.model = WhisperModel(str(self.ct2_dir), device=device, compute_type=compute, local_files_only=True)
        self.pipe = BatchedInferencePipeline(self.model)
        self.device = device

    def _common(self, dur_s):
        c = self.cfg
        return dict(
            language=c["language"], task="transcribe", beam_size=self.beam,
            repetition_penalty=c["repetition_penalty"], no_repeat_ngram_size=c["no_repeat_ngram_size"],
            no_speech_threshold=c["no_speech_threshold"], log_prob_threshold=c["log_prob_threshold"],
            compression_ratio_threshold=c["compression_ratio_threshold"],
            max_new_tokens=int(min(440, max(48, c["max_new_tokens_per_s"] * min(dur_s, 30) + 24))),
            **c.get("extra", {}),
        )

    def transcribe_array(self, audio: np.ndarray) -> str:
        dur = len(audio) / SR
        if dur < 0.2:
            return ""
        c = self.cfg
        if dur <= MAX_DIRECT_S:  # one chunk, no VAD: cannot delete quiet speech
            segs, _ = self.pipe.transcribe(audio, vad_filter=False, batch_size=1, **self._common(dur))
        else:
            vad = dict(threshold=c["vad_threshold"], min_silence_duration_ms=c["vad_min_silence_ms"],
                       speech_pad_ms=c["vad_speech_pad_ms"])
            segs, _ = self.pipe.transcribe(audio, vad_filter=True, vad_parameters=vad,
                                           batch_size=c["batch_size"], **self._common(dur))
        return " ".join(s.text.strip() for s in segs).strip()

    def transcribe_file(self, path) -> str:
        from faster_whisper.audio import decode_audio

        audio = decode_audio(str(path), sampling_rate=SR)
        try:
            text = self.transcribe_array(audio)
        except Exception as e:  # never lose a row: retry with the sequential decoder, then give up quietly
            _WARNED[0] += 1
            if _WARNED[0] <= 5:
                print(f"[infer] batched decode failed ({_safe_err(e, path)}); retrying sequentially", flush=True)
            try:
                segs, _ = self.model.transcribe(audio, language=self.cfg["language"], beam_size=self.beam,
                                                without_timestamps=False, condition_on_previous_text=False)
                text = " ".join(s.text.strip() for s in segs)
            except Exception as e2:
                raise RuntimeError(f"both decoders failed ({_safe_err(e2, path)})") from None
        return postprocess(text, self.lex, self.cfg.get("rules"))


def transcribe_many(t: Transcriber, paths, log_every=100):
    out, t0, failed = [], time.time(), 0
    budget = t.cfg["time_budget_s"]
    n = len(paths)
    for i, p in enumerate(paths):
        try:
            out.append(t.transcribe_file(p))
        except Exception as e:  # one bad clip must not kill the run: emit "" for it, but count it
            failed += 1
            if failed <= 5:  # keep the log budget (500 lines) free for the real stack trace
                print(f"[infer] clip {i} FAILED ({_safe_err(e, p)})", flush=True)
            out.append("")
        el = time.time() - t0
        if t.beam > 1 and el / (i + 1) * n > budget:
            print(f"[infer] projected {el/(i+1)*n:.0f}s > budget {budget}s: switching to greedy", flush=True)
            t.beam = 1
        if (i + 1) % log_every == 0 or i + 1 == n:
            print(f"[infer] {i+1}/{n} clips, {el:.0f}s elapsed, {failed} failed", flush=True)
    if failed > max(2, 0.05 * n):  # systemic problem: a mostly-empty CSV would only score ~1.0 WER, so fail loudly
        raise RuntimeError(f"{failed}/{n} clips failed to transcribe")
    return out
