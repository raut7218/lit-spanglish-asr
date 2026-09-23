"""Gate G0: our NeMo-free port reproduces NeMo's canary-1b-v2 (features, encoder states, greedy transcripts).

    python scripts/convert_canary.py --out DIR --golden clip1.flac clip2.flac ...   # on Colab, NeMo installed
    LIT_CANARY_DIR=DIR pytest -q tests/test_canary_parity.py
"""
import os
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
DIR = os.environ.get("LIT_CANARY_DIR")
pytestmark = pytest.mark.skipif(not DIR or not (Path(DIR) / "golden.pt").exists(),
                                reason="set LIT_CANARY_DIR to a converted dir with golden.pt")


@pytest.fixture(scope="module")
def setup():
    from lit import canary

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cuda.matmul.allow_tf32 = False  # compare in true fp32
    torch.backends.cudnn.allow_tf32 = False
    model, tok = canary.load(DIR, device, torch.float32)
    tok.verbatim = None  # NeMo's stock prompt
    return model, tok, torch.load(Path(DIR) / "golden.pt", weights_only=False), device


def test_features_and_encoder_match_nemo(setup):
    from lit import canary
    from lit.audio import load_audio

    model, _, g, device = setup
    for clip, f_ref, e_ref in zip(g["clips"], g["features"], g["encoder"]):
        wav, lens = canary.pad_waves([load_audio(clip)])
        f, m = model.features(wav.to(device), lens.to(device))
        n = int(m.sum())
        assert n == f_ref.shape[0], f"frame count {n} vs NeMo {f_ref.shape[0]}"
        assert (f[0, :n].cpu() - f_ref).abs().max() < 1e-3, "log-mel frontend differs from NeMo"
        with torch.no_grad():
            e, em = model.encode(f, m)
        k = int(em.sum())
        assert k == e_ref.shape[0], f"encoder length {k} vs NeMo {e_ref.shape[0]}"
        err = (e[0, :k].cpu() - e_ref).abs().max() / e_ref.abs().max()
        assert err < 1e-3, f"encoder relative error {err:.2e}"


@pytest.mark.parametrize("lang", ["es", "en"])
def test_greedy_transcripts_match_nemo(setup, lang):
    from lit import canary
    from lit.audio import load_audio
    from lit.normalize import norm

    model, tok, g, _ = setup
    ours = canary.transcribe(model, tok, [load_audio(c) for c in g["clips"]], [lang], beam=1, batch_size=1)
    theirs = g["text"][lang]
    same = sum(norm(a) == norm(b) for a, b in zip(ours, theirs))
    for a, b in zip(ours, theirs):
        if norm(a) != norm(b):
            print(f"\n ours : {a}\n nemo : {b}")
    assert same == len(theirs), f"{same}/{len(theirs)} identical greedy transcripts ({lang})"
