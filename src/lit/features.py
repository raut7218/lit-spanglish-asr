"""Whisper log-mel on GPU/CPU (torch) + SpecAugment. Matches HF WhisperFeatureExtractor."""

from __future__ import annotations

import numpy as np
import torch

N_FFT, HOP, SR, CHUNK_S = 400, 160, 16000, 30
N_SAMPLES = SR * CHUNK_S


class LogMel:
    def __init__(self, feature_extractor, device):
        self.filters = torch.from_numpy(np.asarray(feature_extractor.mel_filters)).float().to(device)  # (201, n_mels)
        self.window = torch.hann_window(N_FFT).to(device)
        self.device = device
        self.n_mels = self.filters.shape[1]

    @torch.no_grad()
    def __call__(self, wav: torch.Tensor) -> torch.Tensor:
        """wav: (B, 480000) float32 (already zero-padded) -> (B, n_mels, 3000)."""
        wav = wav.to(self.device, dtype=torch.float32)
        stft = torch.stft(wav, N_FFT, HOP, window=self.window, return_complex=True)
        mag = stft[..., :-1].abs() ** 2
        mel = self.filters.T @ mag
        log = torch.clamp(mel, min=1e-10).log10()
        log = torch.maximum(log, log.amax(dim=(-2, -1), keepdim=True) - 8.0)
        return (log + 4.0) / 4.0


def pad_batch(arrs: list[np.ndarray]) -> torch.Tensor:
    out = np.zeros((len(arrs), N_SAMPLES), np.float32)
    for i, a in enumerate(arrs):
        a = a[:N_SAMPLES]
        out[i, : len(a)] = a
    return torch.from_numpy(out)


def spec_augment(feats: torch.Tensor, lengths_frames: list[int], n_freq=2, freq_w=27, n_time=4, time_frac=0.05):
    """Vectorised SpecAugment on (B, n_mels, T): frequency masks anywhere, time masks only inside each clip's real
    frames. No Python loop over the batch (the old version issued ~6 tiny kernels per sample)."""
    B, F_, T = feats.shape
    dev = feats.device
    lens = torch.as_tensor(lengths_frames, device=dev).clamp(min=10)
    fa, ta = torch.arange(F_, device=dev)[None, :], torch.arange(T, device=dev)[None, :]
    fmask = torch.zeros(B, F_, dtype=torch.bool, device=dev)
    for _ in range(n_freq):
        w = torch.randint(0, freq_w + 1, (B,), device=dev)
        f0 = (torch.rand(B, device=dev) * (F_ - w).clamp(min=1)).long()
        fmask |= (fa >= f0[:, None]) & (fa < (f0 + w)[:, None])
    tmask = torch.zeros(B, T, dtype=torch.bool, device=dev)
    for _ in range(n_time):
        w = (torch.rand(B, device=dev) * (time_frac * lens).clamp(min=1)).long()
        t0 = (torch.rand(B, device=dev) * (lens - w).clamp(min=1)).long()
        tmask |= (ta >= t0[:, None]) & (ta < (t0 + w)[:, None])
    return feats.masked_fill(fmask[:, :, None] | tmask[:, None, :], 0.0)
