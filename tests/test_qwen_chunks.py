"""lit.qwen.transcribe: long clips are split into <=max_chunk_s pieces and re-joined in order, per clip."""
import numpy as np

from lit import qwen


class FakeASR:
    def __init__(self):
        self.lengths = []

    def transcribe(self, audio, context, language):
        class R:
            def __init__(self, t):
                self.text = t

        self.lengths += [len(w) / qwen.SR for w, _ in audio]
        return [R(f"w{round(len(w) / qwen.SR)}") for w, _ in audio]


def test_long_clip_split_and_rejoined():
    m = FakeASR()
    wavs = [np.ones(10 * qwen.SR, np.float32), np.ones(70 * qwen.SR, np.float32), np.ones(5 * qwen.SR, np.float32)]
    out = qwen.transcribe(m, wavs, {**qwen.DEFAULT_CFG, "batch": 2})
    assert max(m.lengths) <= 29.0 and len(m.lengths) == 5  # 10 s, 70 s -> 3 pieces, 5 s
    assert out[0] == "w10" and out[2] == "w5" and len(out[1].split()) == 3
