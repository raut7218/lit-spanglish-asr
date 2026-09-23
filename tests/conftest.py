import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "tests"))


import pytest  # noqa: E402


@pytest.fixture(scope="session")
def spm_path(tmp_path_factory):
    """Tiny SentencePiece model with Canary's task tokens as user-defined pieces."""
    spm = pytest.importorskip("sentencepiece")
    from test_canary import SPECIALS

    d = tmp_path_factory.mktemp("spm")
    text = d / "t.txt"
    text.write_text("\n".join(["hola que tal como estas", "I think we should go", "pues vamos a la casa okay",
                               "you know what I mean entonces", "no se yo creo que si"] * 40))
    spm.SentencePieceTrainer.train(input=str(text), model_prefix=str(d / "tok"), vocab_size=80, model_type="bpe",
                                   user_defined_symbols=SPECIALS, pad_id=2, eos_id=-1, bos_id=-1, unk_id=0, minloglevel=2)
    return d / "tok.model"
