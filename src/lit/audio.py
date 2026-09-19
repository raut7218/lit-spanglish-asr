"""Audio IO helpers (ffmpeg-based, so mp3/ogg/opus all work the same on any machine)."""

from __future__ import annotations

import subprocess

import numpy as np

SR = 16000


def load_audio(path, sr: int = SR, start: float | None = None, dur: float | None = None) -> np.ndarray:
    """Decode any audio file to mono float32 at `sr` using ffmpeg."""
    cmd = ["ffmpeg", "-nostdin", "-v", "error"]
    if start is not None:
        cmd += ["-ss", f"{start:.3f}"]
    if dur is not None:
        cmd += ["-t", f"{dur:.3f}"]
    cmd += ["-i", str(path), "-ac", "1", "-ar", str(sr), "-f", "f32le", "-"]
    out = subprocess.run(cmd, capture_output=True, check=True).stdout
    return np.frombuffer(out, dtype=np.float32).copy()


def split_long(audio: np.ndarray, max_s: float = 29.0, sr: int = SR, search_s: float = 8.0):
    """Split a long clip into <=max_s chunks, cutting at the quietest 0.3 s window near the end.

    Used only for HF-generate evaluation of >30 s clips; the runtime path uses VAD chunking.
    """
    max_n = int(max_s * sr)
    if len(audio) <= max_n:
        return [audio]
    chunks, pos = [], 0
    win = int(0.3 * sr)
    while len(audio) - pos > max_n:
        lo, hi = pos + max_n - int(search_s * sr), pos + max_n
        seg = np.abs(audio[lo:hi])
        if len(seg) < 2 * win:
            cut = hi
        else:
            env = np.convolve(seg, np.ones(win) / win, mode="valid")
            cut = lo + int(np.argmin(env)) + win // 2
        chunks.append(audio[pos:cut])
        pos = cut
    chunks.append(audio[pos:])
    return chunks
