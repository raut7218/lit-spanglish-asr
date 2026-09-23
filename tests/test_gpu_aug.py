import pytest

torch = pytest.importorskip("torch")

from lit.gpu_aug import GpuAugConfig, GpuAugmenter, normalize_level_db  # noqa: E402


def _batch(b=4, n=16000 * 6, valid=16000 * 4, level=0.02, seed=0):
    g = torch.Generator().manual_seed(seed)
    t = torch.arange(n) / 16000
    sig = level * torch.sin(2 * torch.pi * 220 * t) * (1 + torch.sin(2 * torch.pi * 3 * t))
    x = sig[None].repeat(b, 1) + 0.002 * torch.randn(b, n, generator=g)
    lens = torch.full((b,), valid)
    x[:, valid:] = 0
    return x.float(), lens


def _rms_db(x, lens):
    return [20 * torch.log10((x[i, : lens[i]] ** 2).mean().sqrt()).item() for i in range(len(x))]


def test_shapes_padding_and_finite():
    x, lens = _batch()
    torch.manual_seed(1)
    y = GpuAugmenter(GpuAugConfig(p_reverb=1, p_noise=1, p_babble=1, p_bandlimit=1, p_denoise=1, p_eq=1, p_clip=1))(x, lens)
    assert y.shape == x.shape and torch.isfinite(y).all()
    assert y.abs().max() <= 1.0
    assert (y[:, lens[0]:] == 0).all(), "zero padding must be preserved (Whisper pads with zeros)"


def test_level_is_normalised_to_dev_like_loudness():
    quiet, lens = _batch(level=0.005)
    loud, _ = _batch(level=0.2)
    cfg = GpuAugConfig(level_db_std=0.0, p_denoise=0, p_eq=0, p_noise=0, p_babble=0, p_reverb=0, p_bandlimit=0, p_clip=0)
    aug = GpuAugmenter(cfg)
    for x in (quiet, loud):
        y = aug(x, lens)
        assert all(abs(v - (-20.0)) < 0.7 for v in _rms_db(y, lens))  # both land at -20 dBFS: the 14 dB gap is gone


def test_denoise_raises_speech_to_floor():
    x, lens = _batch(level=0.02)
    x = x + 0.01 * torch.randn_like(x) * (torch.arange(x.shape[1])[None, :] < lens[:, None])
    cfg = GpuAugConfig(p_denoise=1.0, p_eq=0, p_noise=0, p_babble=0, p_reverb=0, p_bandlimit=0, p_clip=0, level_db_std=0.0)
    y = GpuAugmenter(cfg)(x, lens)

    def contrast(z):  # loud-frame energy over quiet-frame energy (dB)
        fr = z[:, : lens[0] // 400 * 400].reshape(z.shape[0], -1, 400).pow(2).mean(2)
        return (10 * torch.log10(fr.quantile(0.9, dim=1) / fr.quantile(0.1, dim=1).clamp(min=1e-12))).mean().item()

    assert contrast(y) > contrast(x)


def test_normalize_level_db():
    x, lens = _batch(level=0.001)
    y = normalize_level_db(x, lens, -20.0)
    assert all(abs(v + 20.0) < 0.5 for v in _rms_db(y, lens))


def test_spec_augment_vectorised_masks_stay_in_clip():
    from lit.gpu_aug import spec_augment

    f = torch.ones(3, 80, 3000)
    out = spec_augment(f.clone(), [3000, 1500, 400])
    assert out.shape == f.shape and (out == 0).any() and (out == 1).any()
    # frames beyond a clip's real length are never masked by the TIME masks (only freq masks touch them)
    tail = out[2, :, 500:]
    assert ((tail == 0).all(dim=1) | (tail == 1).all(dim=1)).all()  # whole rows only => frequency masks, not time masks


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_all_ops_run_on_gpu():
    x, lens = _batch()
    x, lens = x.cuda(), lens.cuda()
    torch.manual_seed(2)
    cfg = GpuAugConfig(p_reverb=1, p_noise=1, p_babble=1, p_bandlimit=1, p_denoise=1, p_eq=1, p_clip=1)
    y = GpuAugmenter(cfg)(x, lens)
    torch.cuda.synchronize()
    assert y.is_cuda and torch.isfinite(y).all() and (y[:, int(lens[0]):] == 0).all()
