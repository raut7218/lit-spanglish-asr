import numpy as np
import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")


def test_logmel_matches_hf():
    from transformers import WhisperFeatureExtractor

    from lit.features import LogMel, pad_batch

    try:
        fe = WhisperFeatureExtractor()  # default 80-mel config, no download needed
    except Exception as e:  # pragma: no cover
        pytest.skip(str(e))
    x = (np.random.default_rng(0).standard_normal(16000 * 5) * 0.1).astype(np.float32)
    ref = fe(x, sampling_rate=16000, return_tensors="np").input_features[0]
    got = LogMel(fe, torch.device("cpu"))(pad_batch([x]))[0].numpy()
    assert got.shape == ref.shape
    assert np.abs(got - ref).max() < 1e-3
