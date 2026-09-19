"""Runtime inference with faster-whisper (CTranslate2). NO torch / transformers / NeMo imports:
this module is copied into submission.zip and must run in the offline competition image.
"""

from __future__ import annotations

import json
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
    time_budget_s=6000,  # degrade to greedy if the projected total exceeds this
)


def load_cfg(model_dir) -> dict:
    cfg = dict(DEFAULT_CFG)
    p = Path(model_dir) / "infer_config.json"
    if p.exists():
        cfg.update(json.loads(p.read_text()))
    return cfg


class Transcriber:
    def __init__(self, ct2_dir, cfg: dict | None = None, lexicon: dict | None = None, device: str = "auto"):
        from faster_whisper import BatchedInferencePipeline, WhisperModel

        self.cfg = {**DEFAULT_CFG, **(cfg or {})}
        self.lex = lexicon
        if device == "auto":
            try:
                import ctranslate2

                device = "cuda" if ctranslate2.get_cuda_device_count() > 0 else "cpu"
            except Exception:
                device = "cpu"
        compute = self.cfg["compute_type"] if device == "cuda" else "int8"
        self.model = WhisperModel(str(ct2_dir), device=device, compute_type=compute, local_files_only=True)
        self.pipe = BatchedInferencePipeline(self.model)
        self.beam = self.cfg["beam_size"]
        self.device = device

    def _common(self, dur_s):
        c = self.cfg
        return dict(
            language=c["language"], task="transcribe", beam_size=self.beam,
            repetition_penalty=c["repetition_penalty"], no_repeat_ngram_size=c["no_repeat_ngram_size"],
            no_speech_threshold=c["no_speech_threshold"], log_prob_threshold=c["log_prob_threshold"],
            compression_ratio_threshold=c["compression_ratio_threshold"],
            max_new_tokens=int(min(440, max(48, c["max_new_tokens_per_s"] * min(dur_s, 30) + 24))),
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
            print(f"[infer] batched decode failed for {path}: {e!r}; retrying sequentially", flush=True)
            try:
                segs, _ = self.model.transcribe(audio, language=self.cfg["language"], beam_size=self.beam,
                                                without_timestamps=False, condition_on_previous_text=False)
                text = " ".join(s.text.strip() for s in segs)
            except Exception as e2:
                print(f"[infer] sequential decode failed too: {e2!r}", flush=True)
                text = ""
        return postprocess(text, self.lex)


def transcribe_many(t: Transcriber, paths, log_every=25):
    out, t0 = [], time.time()
    budget = t.cfg["time_budget_s"]
    n = len(paths)
    for i, p in enumerate(paths):
        try:
            out.append(t.transcribe_file(p))
        except Exception as e:
            print(f"[infer] FAILED {p}: {e!r}", flush=True)
            out.append("")
        el = time.time() - t0
        if t.beam > 1 and el / (i + 1) * n > budget:
            print(f"[infer] projected {el/(i+1)*n:.0f}s > budget {budget}s: switching to greedy", flush=True)
            t.beam = 1
        if (i + 1) % log_every == 0 or i + 1 == n:
            print(f"[infer] {i+1}/{n} clips, {el:.0f}s elapsed", flush=True)
    return out
