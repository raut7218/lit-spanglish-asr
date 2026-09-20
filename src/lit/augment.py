"""On-the-fly waveform augmentation that pushes clean face-to-face audio (Bangor Miami, belt mics)
towards the test domain: WhatsApp voice notes (phone mic, Opus/AMR/MP3 codecs, noise, room reverb).

All ops take/return float32 mono numpy at 16 kHz. Everything is seeded by the passed `rng`.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass

import numpy as np

SR = 16000


@dataclass
class AugConfig:
    p_speed: float = 0.5
    speeds: tuple = (0.9, 1.0, 1.1)
    p_gain: float = 0.8
    p_noise: float = 0.6
    snr_db: tuple = (5.0, 30.0)
    p_babble: float = 0.15
    p_reverb: float = 0.3
    p_bandlimit: float = 0.2
    p_codec: float = 0.6
    p_clip: float = 0.1
    enabled: bool = True


# ------------------------------------------------------------------ basic ops
def _rms(x):
    return float(np.sqrt(np.mean(np.square(x)) + 1e-12))


def speed_perturb(x, factor):
    """Resample by `factor` (tempo + pitch change), like sox speed."""
    if abs(factor - 1.0) < 1e-3:
        return x
    n = int(round(len(x) / factor))
    return np.interp(np.linspace(0, len(x) - 1, n), np.arange(len(x)), x).astype(np.float32)


def colored_noise(n, rng, beta):
    """1/f^beta noise (beta 0 white, 1 pink, 2 brown)."""
    white = rng.standard_normal(n).astype(np.float32)
    if beta == 0:
        return white
    spec = np.fft.rfft(white)
    f = np.fft.rfftfreq(n, 1 / SR)
    f[0] = f[1]
    spec = spec / (f ** (beta / 2))
    out = np.fft.irfft(spec, n).astype(np.float32)
    return out / (np.std(out) + 1e-9)


def add_noise(x, rng, snr_db):
    beta = rng.choice([0.0, 1.0, 2.0])
    noise = colored_noise(len(x), rng, beta)
    # random slow amplitude modulation makes the noise less stationary
    if rng.random() < 0.5:
        t = np.arange(len(x)) / SR
        noise = noise * (0.6 + 0.4 * np.sin(2 * np.pi * rng.uniform(0.1, 1.5) * t + rng.uniform(0, 6.28)))
    scale = _rms(x) / (10 ** (snr_db / 20)) / (_rms(noise) + 1e-9)
    return (x + noise * scale).astype(np.float32)


def add_babble(x, rng, pool, snr_db):
    """Mix 2-4 other random clips from `pool` (list of arrays) as background chatter."""
    if not pool:
        return x
    n = len(x)
    mix = np.zeros(n, np.float32)
    for _ in range(int(rng.integers(2, 5))):
        y = pool[int(rng.integers(len(pool)))]
        if len(y) < n:
            y = np.tile(y, n // len(y) + 1)
        s = int(rng.integers(0, len(y) - n + 1))
        mix += y[s : s + n]
    scale = _rms(x) / (10 ** (snr_db / 20)) / (_rms(mix) + 1e-9)
    return (x + mix * scale).astype(np.float32)


def reverb(x, rng):
    """Convolve with a synthetic exponentially-decaying RIR (RT60 0.1-0.6 s)."""
    rt60 = rng.uniform(0.1, 0.6)
    n = int(rt60 * SR)
    t = np.arange(n) / SR
    rir = rng.standard_normal(n) * np.exp(-6.9 * t / rt60)
    rir[0] = 1.0
    rir = (rir / np.linalg.norm(rir)).astype(np.float32)
    dry = rng.uniform(0.3, 0.7)
    wet = np.convolve(x, rir)[: len(x)].astype(np.float32)
    out = dry * x + (1 - dry) * wet
    return (out * (_rms(x) / (_rms(out) + 1e-9))).astype(np.float32)


def bandlimit(x, rng):
    """Telephone-band style low-pass by down/up-sampling through 8k or 6k."""
    low = int(rng.choice([6000, 8000]))
    n = int(len(x) * low / SR)
    down = np.interp(np.linspace(0, len(x) - 1, n), np.arange(len(x)), x)
    # crude anti-alias then back up
    k = max(1, SR // low)
    down = np.convolve(down, np.ones(k) / k, mode="same")
    return np.interp(np.linspace(0, n - 1, len(x)), np.arange(n), down).astype(np.float32)


def soft_clip(x, rng):
    g = rng.uniform(2.0, 6.0)
    return (np.tanh(x * g) / np.tanh(g)).astype(np.float32)


# ------------------------------------------------------------------ codecs (ffmpeg)
CODECS = [
    ("opus", ["-c:a", "libopus", "-application", "voip"], ["12k", "16k", "24k", "32k"], "ogg", 16000),
    ("mp3", ["-c:a", "libmp3lame"], ["32k", "48k", "64k"], "mp3", 16000),
    ("amr", ["-c:a", "libopencore_amrnb"], ["6.7k", "10.2k", "12.2k"], "amr", 8000),
]


def codec_roundtrip(x, rng):
    """Encode+decode through a lossy codec via ffmpeg pipes. Returns x unchanged on any failure."""
    name, args, rates, fmt, rate = CODECS[int(rng.integers(len(CODECS)))]
    br = rates[int(rng.integers(len(rates)))]
    try:
        enc = subprocess.run(
            ["ffmpeg", "-nostdin", "-v", "error", "-f", "f32le", "-ar", str(SR), "-ac", "1", "-i", "-",
             "-ar", str(rate), "-ac", "1", *args, "-b:a", br, "-f", fmt, "-"],
            input=x.astype(np.float32).tobytes(), capture_output=True, check=True, timeout=30,
        ).stdout
        dec = subprocess.run(
            ["ffmpeg", "-nostdin", "-v", "error", "-i", "-", "-ar", str(SR), "-ac", "1", "-f", "f32le", "-"],
            input=enc, capture_output=True, check=True, timeout=30,
        ).stdout
        y = np.frombuffer(dec, np.float32).copy()
        if len(y) < 100:
            return x
        # codecs add delay/padding; trim/pad to the original length
        y = y[: len(x)] if len(y) >= len(x) else np.pad(y, (0, len(x) - len(y)))
        return y
    except Exception:
        return x


class Augmenter:
    def __init__(self, cfg: AugConfig | None = None, babble_pool=None):
        self.cfg = cfg or AugConfig()
        self.pool = babble_pool or []

    def __call__(self, x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        c = self.cfg
        if not c.enabled:
            return x
        x = x.astype(np.float32)
        if rng.random() < c.p_speed:
            x = speed_perturb(x, float(rng.choice(c.speeds)))
        if rng.random() < c.p_reverb:
            x = reverb(x, rng)
        if rng.random() < c.p_bandlimit:
            x = bandlimit(x, rng)
        if rng.random() < c.p_babble:
            x = add_babble(x, rng, self.pool, rng.uniform(8, 25))
        if rng.random() < c.p_noise:
            x = add_noise(x, rng, rng.uniform(*c.snr_db))
        if rng.random() < c.p_clip:
            x = soft_clip(x, rng)
        if rng.random() < c.p_codec:
            x = codec_roundtrip(x, rng)
        if rng.random() < c.p_gain:
            x = x * float(10 ** (rng.uniform(-12, 6) / 20))
        peak = float(np.max(np.abs(x)) + 1e-9)
        if peak > 1.0:
            x = x / peak
        return x.astype(np.float32)


def _ff(args, data):
    return subprocess.run(["ffmpeg", "-nostdin", "-v", "error", *args], input=data, capture_output=True, check=True, timeout=60).stdout


def codec_chain(x: np.ndarray, rng) -> np.ndarray:
    """WhatsApp-voice-note chain used by the test set: Opus (voip, 12-24 kbps) -> MP3 64 kbps @ 48 kHz -> 16 kHz.
    Any ffmpeg failure returns the input unchanged."""
    br = rng.choice(["12k", "16k", "24k", "none"])
    try:
        cur = x.astype(np.float32).tobytes()
        fmt_in = ["-f", "f32le", "-ar", "16000", "-ac", "1", "-i", "-"]
        if br != "none":
            cur = _ff(fmt_in + ["-c:a", "libopus", "-b:a", br, "-application", "voip", "-f", "ogg", "-"], cur)
            cur = _ff(["-i", "-", "-ar", "48000", "-ac", "1", "-c:a", "libmp3lame", "-b:a", "64k", "-f", "mp3", "-"], cur)
        else:
            cur = _ff(fmt_in + ["-ar", "48000", "-c:a", "libmp3lame", "-b:a", "64k", "-f", "mp3", "-"], cur)
        y = np.frombuffer(_ff(["-i", "-", "-ar", "16000", "-ac", "1", "-f", "f32le", "-"], cur), np.float32)
        if len(y) < 100:
            return x
        return y[: len(x)] if len(y) >= len(x) else np.pad(y, (0, len(x) - len(y)))
    except Exception:
        return x
