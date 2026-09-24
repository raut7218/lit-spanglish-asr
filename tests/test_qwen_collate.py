"""Qwen collator: loss only on each clip's own target + eos, whatever side the processor pads on."""
import os

import numpy as np
import pytest

qwen_asr = pytest.importorskip("qwen_asr")
MODEL = os.environ.get("LIT_QWEN_PROCESSOR", "Qwen/Qwen3-ASR-0.6B")


def test_labels_are_exactly_the_targets():
    from qwen_asr.core.transformers_backend.processing_qwen3_asr import Qwen3ASRProcessor

    from lit.train_qwen import Collate

    try:
        proc = Qwen3ASRProcessor.from_pretrained(MODEL)
    except Exception as e:  # offline without a cached processor
        pytest.skip(f"processor unavailable: {e}")
    col = Collate(proc)
    tg = ["language Spanish<asr_text>hola que tal", "language English<asr_text>a much longer target sentence here okay"]
    b = col([(np.zeros(16000 * 3, np.float32), tg[0]), (np.zeros(16000 * 9, np.float32), tg[1])])  # different lengths -> padding
    for i in range(2):
        got = proc.tokenizer.decode(b["input_ids"][i][b["labels"][i] != -100])
        assert got == tg[i] + proc.tokenizer.eos_token


def test_gpu_logmel_matches_processor_features():
    import torch
    from qwen_asr.core.transformers_backend.processing_qwen3_asr import Qwen3ASRProcessor

    from lit.features import LogMel
    from lit.train_qwen import Collate

    try:
        proc = Qwen3ASRProcessor.from_pretrained(MODEL)
    except Exception as e:
        pytest.skip(f"processor unavailable: {e}")
    rng = np.random.default_rng(0)
    b = Collate(proc, return_wav=True)([(0.1 * rng.standard_normal(16000 * 3).astype(np.float32), "language None<asr_text>", True),
                                        (0.1 * rng.standard_normal(16037 * 7).astype(np.float32), "language English<asr_text>hi", False)])
    feats = LogMel(proc.feature_extractor, "cpu")(b["wav"])
    assert feats.shape == b["input_features"].shape
    assert torch.allclose(feats, b["input_features"], atol=1e-4)
    assert b["ns"].tolist() == [True, False] and b["wav_len"].tolist() == [48000, 16037 * 7]
