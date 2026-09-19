import numpy as np

from lit.augment import AugConfig, Augmenter, codec_roundtrip, reverb, speed_perturb


def _sig(n=16000 * 3):
    t = np.arange(n) / 16000
    return (0.3 * np.sin(2 * np.pi * 220 * t) * (1 + np.sin(2 * np.pi * 3 * t))).astype(np.float32)


def test_ops_shapes():
    rng = np.random.default_rng(0)
    x = _sig()
    assert abs(len(speed_perturb(x, 1.1)) - len(x) / 1.1) < 2
    assert len(reverb(x, rng)) == len(x)
    for _ in range(6):
        y = codec_roundtrip(x, rng)
        assert len(y) == len(x) and np.isfinite(y).all()


def test_augmenter_safe_range():
    aug = Augmenter(AugConfig(p_codec=1.0, p_noise=1.0, p_reverb=1.0, p_bandlimit=1.0, p_babble=1.0), [_sig()])
    for s in range(5):
        y = aug(_sig(), np.random.default_rng(s))
        assert y.dtype == np.float32 and np.isfinite(y).all() and np.max(np.abs(y)) <= 1.0 + 1e-6
