"""Qwen3-ASR (qwen-asr 0.0.6, runtime-pinned) helpers shared by training, evaluation and the runtime.

Targets use Qwen's native format "language X<asr_text>text" (X = the clip's majority language; empty clips
"language None<asr_text>"), so decoding with language=None lets the model pick the prefix per clip and the
package strips it. Long clips need no VAD: qwen-asr decodes up to 1200 s in one pass.

Runtime (vLLM, its own process so the GPU is released before faster-whisper loads):
    python -m lit.qwen --model DIR --paths_file clips.txt --out hyps.json [--cfg '{"rules": [...]}']
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from .postprocess import postprocess

SR = 16000
ASR_TAG = "<asr_text>"
DEFAULT_CFG = dict(language=None, rules=[], max_new_tokens=1024, gpu_memory_utilization=0.85, max_model_len=8192,
                   batch=128, context="")


def target_text(row: dict) -> str:
    text = " ".join(str(row.get("text") or "").split())
    if row.get("nonspeech") or not text:
        return f"language None{ASR_TAG}"
    lang = "Spanish" if (row.get("spa_frac") or 0.0) >= 0.5 else "English"
    return f"language {lang}{ASR_TAG}{text}"


def load_wavs(paths):
    """Runtime decoder: PyAV via faster-whisper (ships in the image, handles every container); lit.audio elsewhere."""
    try:
        from faster_whisper.audio import decode_audio

        return [decode_audio(str(p), sampling_rate=SR) for p in paths]
    except ImportError:
        from .audio import load_audio

        return [load_audio(p) for p in paths]


def load_vllm(model_dir, cfg: dict):
    from qwen_asr import Qwen3ASRModel

    return Qwen3ASRModel.LLM(model=str(model_dir), gpu_memory_utilization=cfg["gpu_memory_utilization"],
                             max_model_len=cfg["max_model_len"], max_inference_batch_size=cfg["batch"],
                             max_new_tokens=cfg["max_new_tokens"])


def transcribe(model, wavs, cfg: dict) -> list[str]:
    """wavs: 16 kHz float32 arrays -> postprocessed (scorer-normalised + rules) hypotheses, order kept."""
    out = []
    for i in range(0, len(wavs), cfg["batch"]):
        chunk = [(w, SR) for w in wavs[i : i + cfg["batch"]]]
        res = model.transcribe(audio=chunk, context=cfg["context"], language=cfg["language"])
        out += [r.text for r in res]
    return [postprocess(t, None, cfg.get("rules")) for t in out]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--paths_file", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cfg", default="{}")
    a = ap.parse_args(argv)
    cfg = {**DEFAULT_CFG, **json.loads(a.cfg)}
    paths = Path(a.paths_file).read_text(encoding="utf-8").splitlines()
    t0 = time.time()
    model = load_vllm(a.model, cfg)
    print(f"[qwen] loaded in {time.time()-t0:.0f}s", flush=True)
    wavs, hyps = [], []
    for i, p in enumerate(paths):  # a clip that fails to decode must not kill the run: it gets ""
        try:
            wavs.append(load_wavs([p])[0])
        except Exception as e:
            print(f"[qwen] clip {i} unreadable ({type(e).__name__})", flush=True)
            wavs.append(None)
    ok = [i for i, w in enumerate(wavs) if w is not None and len(w) >= int(0.2 * SR)]
    got = dict(zip(ok, transcribe(model, [wavs[i] for i in ok], cfg)))
    hyps = [got.get(i, "") for i in range(len(paths))]
    Path(a.out).write_text(json.dumps(hyps, ensure_ascii=False), encoding="utf-8")
    print(f"[qwen] {len(paths)} clips in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    sys.exit(main())
