"""Batched GPU waveform augmentation aimed at the TEST acoustics (WhatsApp voice notes), not at "more noise".

Audit (scripts/audio_audit.py): dev voice notes are loud (median -20 dBFS, peaks ~ -0.5 dB), fairly clean
(speech-to-floor ~28 dB) and bright; Bangor Miami belt-mic audio is ~14 dB quieter, ~8 dB SNR and muffled
(<0.1% of the energy above 4 kHz). So per clip we (1) optionally denoise (spectral gating), (2) apply a random
brightening EQ tilt + high-pass, (3) add only light noise / babble / reverb, (4) normalise the level to
dev-like loudness with a soft limiter. Everything is vectorised over the batch (FFT based), zero padding is preserved.
Inputs: wav (B, N) float32 on the device, lengths (B,) valid samples. The stateful RNG is torch's (seed via torch.manual_seed).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

SR = 16000
N_FFT, HOP = 512, 128


@dataclass
class GpuAugConfig:
    enabled: bool = True
    level_db_mean: float = -20.0  # dev median loudness
    level_db_std: float = 2.0
    p_denoise: float = 0.6
    denoise_alpha: float = 1.6
    denoise_floor: float = 0.12
    p_eq: float = 0.7
    tilt_db_per_oct: tuple = (0.0, 4.0)  # brightening: + dB per octave above 1 kHz (Miami is muffled)
    highpass_hz: tuple = (60.0, 160.0)
    p_noise: float = 0.3
    snr_db: tuple = (18.0, 40.0)  # dev is clean: never train on much noisier audio than that
    p_babble: float = 0.08
    babble_snr_db: tuple = (10.0, 25.0)
    p_reverb: float = 0.2
    rt60_s: tuple = (0.1, 0.5)
    p_bandlimit: float = 0.1
    bandlimit_hz: tuple = (4000.0, 7000.0)
    p_clip: float = 0.05


def _valid_mask(lengths: torch.Tensor, n: int) -> torch.Tensor:
    return torch.arange(n, device=lengths.device)[None, :] < lengths[:, None]


def _rms(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return ((x ** 2 * mask).sum(1) / mask.sum(1).clamp(min=1)).sqrt().clamp(min=1e-6)


def _pick(p: float, b: int, device) -> torch.Tensor:
    return torch.rand(b, device=device) < p


def _u(lo_hi, b: int, device) -> torch.Tensor:
    lo, hi = lo_hi
    return lo + (hi - lo) * torch.rand(b, device=device)


class GpuAugmenter:
    def __init__(self, cfg: GpuAugConfig | None = None):
        self.cfg = cfg or GpuAugConfig()
        self._win = None

    # ------------------------------------------------------------------ STFT-domain: denoise + EQ
    def _stft_ops(self, x, mask, do_dn, do_eq, tilt, hp):
        b, n = x.shape
        c = self.cfg
        if self._win is None or self._win.device != x.device:
            self._win = torch.hann_window(N_FFT, device=x.device)
        X = torch.stft(x, N_FFT, HOP, window=self._win, return_complex=True)  # (B, F, T)
        P = torch.view_as_real(X).pow(2).sum(-1)  # |X|^2 without the complex-abs JIT kernel (needs NVRTC builtins)
        f = torch.fft.rfftfreq(N_FFT, 1 / SR).to(x.device)  # (F,)
        gain = torch.ones_like(P)
        T = X.shape[-1]
        if do_dn.any():
            frame_e = P.sum(1)  # (B, T)
            n_valid = (mask.sum(1) / HOP).long().clamp(min=8, max=T)  # valid frames per clip
            fr_ok = torch.arange(T, device=x.device)[None, :] < n_valid[:, None]
            e = frame_e.masked_fill(~fr_ok, float("inf"))
            k = (n_valid.float() * 0.1).long().clamp(min=3)
            for i in torch.nonzero(do_dn).flatten().tolist():
                idx = torch.topk(e[i], int(k[i]), largest=False).indices  # quietest 10% of frames = noise estimate
                noise = P[i][:, idx].mean(1, keepdim=True)  # (F, 1)
                m = (1 - c.denoise_alpha * noise / P[i].clamp(min=1e-10)).clamp(min=c.denoise_floor, max=1.0)
                m = torch.nn.functional.avg_pool1d(m[None], 3, 1, 1, count_include_pad=False)[0]  # light smoothing over freq
                gain[i] = m
        if do_eq.any():
            oct_ = torch.log2(f.clamp(min=1.0) / 1000.0)  # (F,)
            tilt_db = tilt[:, None] * oct_[None, :].clamp(min=0.0)  # boost only above 1 kHz
            hp_db = -24.0 * torch.relu(torch.log2(hp[:, None] / f.clamp(min=1.0)[None, :]))  # 4th-order-ish high-pass
            eq_db = (tilt_db + hp_db) * do_eq[:, None].float()
            gain = gain * (10 ** (eq_db / 20))[:, :, None]
        y = torch.istft(torch.view_as_complex(torch.view_as_real(X) * gain[..., None]), N_FFT, HOP, window=self._win, length=n)
        return y

    def _colored_noise(self, b, n, device):
        white = torch.randn(b, n, device=device)
        spec = torch.fft.rfft(white)
        f = torch.fft.rfftfreq(n, 1 / SR).to(device).clamp(min=20.0)
        beta = torch.randint(0, 3, (b, 1), device=device).float()  # 0 white / 1 pink / 2 brown
        out = torch.fft.irfft(spec / f[None, :] ** (beta / 2), n)
        return out / out.std(1, keepdim=True).clamp(min=1e-6)

    @torch.no_grad()
    def __call__(self, wav: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        c = self.cfg
        if not c.enabled:
            return wav
        b, n = wav.shape
        dev = wav.device
        mask = _valid_mask(lengths.to(dev), n).float()
        x = wav * mask

        # reverb (synthetic exponentially decaying RIR, dry/wet mix), FFT convolution
        sel = _pick(c.p_reverb, b, dev)
        if sel.any():
            rt = _u(c.rt60_s, b, dev)
            t = torch.arange(int(0.5 * SR), device=dev)[None, :] / SR
            rir = torch.randn(b, t.shape[1], device=dev) * torch.exp(-6.9 * t / rt[:, None])
            rir[:, 0] = 1.0
            rir = rir / rir.norm(dim=1, keepdim=True)
            L = n + rir.shape[1]
            wet = torch.fft.irfft(torch.fft.rfft(x, L) * torch.fft.rfft(rir, L), L)[:, :n]
            dry = _u((0.35, 0.8), b, dev)[:, None]
            mixed = dry * x + (1 - dry) * wet
            mixed = mixed * (_rms(x, mask) / _rms(mixed, mask))[:, None]
            x = torch.where(sel[:, None], mixed, x) * mask

        # bandlimit (phone-band low-pass)
        sel = _pick(c.p_bandlimit, b, dev)
        if sel.any():
            cutoff = _u(c.bandlimit_hz, b, dev)
            F_ = torch.fft.rfft(x)
            f = torch.fft.rfftfreq(n, 1 / SR).to(dev)
            keep = (f[None, :] <= cutoff[:, None]) | (~sel[:, None])
            x = torch.fft.irfft(F_ * keep, n) * mask

        # denoise + brightening EQ tilt + high-pass (one STFT round trip)
        do_dn, do_eq = _pick(c.p_denoise, b, dev), _pick(c.p_eq, b, dev)
        if do_dn.any() or do_eq.any():
            x = self._stft_ops(x, mask, do_dn, do_eq, _u(c.tilt_db_per_oct, b, dev), _u(c.highpass_hz, b, dev)) * mask

        # light additive noise / in-batch babble at high SNR (only inside the valid region)
        sel = _pick(c.p_noise, b, dev)
        if sel.any():
            noise = self._colored_noise(b, n, dev) * mask
            snr = _u(c.snr_db, b, dev)
            scale = _rms(x, mask) / (10 ** (snr / 20)) / _rms(noise, mask)
            x = x + noise * (scale * sel.float())[:, None]
        sel = _pick(c.p_babble, b, dev) if b > 1 else torch.zeros(b, dtype=torch.bool, device=dev)
        if sel.any():
            other = torch.roll(x, 1, 0)
            snr = _u(c.babble_snr_db, b, dev)
            scale = _rms(x, mask) / (10 ** (snr / 20)) / _rms(other, mask)
            x = x + other * mask * (scale * sel.float())[:, None]

        # level normalisation to dev-like loudness (+ soft limiter): removes the 14 dB Miami/dev level gap
        target = c.level_db_mean + c.level_db_std * torch.randn(b, device=dev)
        x = x * (10 ** (target / 20) / _rms(x, mask))[:, None]
        peak = x.abs().amax(1, keepdim=True)
        x = torch.where(peak > 0.97, torch.tanh(x / peak.clamp(min=1e-6) * 1.6) / 1.6 * 0.97 / 0.5, x)  # rare soft limit
        sel = _pick(c.p_clip, b, dev)
        if sel.any():
            g = _u((2.0, 5.0), b, dev)[:, None]
            x = torch.where(sel[:, None], torch.tanh(x * g) / torch.tanh(g), x)
        return (x * mask).clamp(-1.0, 1.0)


def normalize_level_db(wav: torch.Tensor, lengths: torch.Tensor, target_db=-20.0) -> torch.Tensor:
    """Deterministic level normalisation (used for inference/eval paths when enabled)."""
    mask = _valid_mask(lengths.to(wav.device), wav.shape[1]).float()
    x = wav * mask
    t = torch.as_tensor(target_db, dtype=x.dtype, device=x.device).expand(x.shape[0])
    return (x * (10 ** (t / 20) / _rms(x, mask))[:, None]).clamp(-1.0, 1.0)
