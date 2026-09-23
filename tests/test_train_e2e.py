"""End-to-end training smoke test: a tiny random Canary (built locally, no download) on a synthetic mini dataset.
Exercises the sampler, turn/window/non-speech kinds, GPU aug, the frozen lower encoder, per-clip language + verbatim
prompts, EMA, multi-set validation, checkpoint selection/averaging, resume and the final model export."""
import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("sentencepiece")

WORDS = "hola que tal maybe I think pues vamos para la casa okay entonces you know".split()


def _clip(root: Path, rel: str, dur: float, seed: int):
    rng = np.random.default_rng(seed)
    t = np.arange(int(dur * 16000)) / 16000
    x = 0.1 * np.sin(2 * np.pi * (150 + 40 * rng.random()) * t) * (1 + np.sin(2 * np.pi * 3 * t)) + 0.01 * rng.standard_normal(len(t))
    (root / rel).parent.mkdir(parents=True, exist_ok=True)
    sf.write(root / rel, x.astype(np.float32), 16000)


def _rows(root, name, n, kind, seed, dev=False):
    rows, rng = [], np.random.default_rng(seed)
    for i in range(n):
        dur = float(rng.uniform(2, 9)) if kind != "window" else float(rng.uniform(10, 20))
        rel = f"{name}/{kind}_{i}.flac"
        _clip(root, rel, dur, seed * 100 + i)
        text = " ".join(rng.choice(WORDS, size=int(dur * 2))) if kind != "nonspeech" else ""
        r = dict(id=f"{name}_{kind}_{i}", audio=rel, duration=round(dur, 2), text=text, kind=kind, speaker=str(1 + i % 2), conv="c",
                 spa_frac=float(rng.random()))
        if kind == "nonspeech":
            r["nonspeech"] = True
        if dev:
            r.pop("kind")
            r["ref"] = text
        rows.append(r)
    return rows


def _write(path, rows):
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")


def _tiny_model(tmp_path, spm_path):
    from test_canary import tiny_cfg

    from lit import canary

    tok = canary.Tokenizer(spm_path)
    cfg = dict(tiny_cfg(tok.sp.get_piece_size()), verbatim_token="<|spltoken0|>")
    m = canary.CanaryModel(cfg)
    from transformers import ParakeetFeatureExtractor

    m.mel_fb.copy_(ParakeetFeatureExtractor(feature_size=cfg["preprocessor"]["features"]).mel_filters)
    canary.save(m, tmp_path / "base", tokenizer_path=spm_path)
    return tmp_path / "base"


def test_train_canary_end_to_end(tmp_path, spm_path):
    from lit import canary
    from lit.train import main

    d = tmp_path / "data"
    train = _rows(d, "clips", 10, "turn", 1) + _rows(d, "clips", 6, "window", 2) + _rows(d, "clips", 2, "nonspeech", 3)
    hold = _rows(d, "hold", 6, "turn", 4) + _rows(d, "hold", 4, "window", 5)
    dev = _rows(d, "dev_clips", 8, "turn", 6, dev=True)
    _write(d / "train.jsonl", train)
    _write(d / "miami_holdout.jsonl", hold)
    _write(d / "dev.jsonl", dev)
    out = tmp_path / "run"
    base = _tiny_model(tmp_path, spm_path)
    args = ["--config", str(Path(__file__).resolve().parents[1] / "configs" / "smoke_canary.yaml"), "--set", f"model={base}",
            f"data_dir={d}", f"out_dir={out}", "dev_in_train=spk1"]
    main(args + ["max_steps=2"])
    assert (out / "last" / "weights.pt").exists() and (out / "train_state.pt").exists()
    main(args)  # resumes at step 2 and finishes at 4
    fj = json.loads((out / "final_model" / "final.json").read_text())
    assert fj["winner"] in ("single", "avg") and "holdout_turn" in fj["scores"][fj["winner"]] and fj["steps"] == 4
    m, tok = canary.load(out / "final_model")
    assert tok.verbatim == "<|spltoken0|>"
    m0, _ = canary.load(base)
    frozen = "encoder.layers.0.feed_forward1.linear1.weight"
    assert torch.equal(m.state_dict()[frozen], m0.state_dict()[frozen].half().float())  # frozen layer untouched
    assert not torch.equal(m.state_dict()["layers.0.fc1.weight"], m0.state_dict()["layers.0.fc1.weight"].half().float())
    assert len((out / "experiments.jsonl").read_text().splitlines()) >= 2

    # the exported dir through the runtime path (decode files -> batched beam search, both prompts -> postprocess)
    from lit.infer import Transcriber, transcribe_many

    t = Transcriber(out / "final_model", {"beam_size": 2, "group_size": 3, "batch_size": 2, "rules": ["capital_i"]}, device="cpu")
    paths = [d / r["audio"] for r in dev[:4]]
    _clip(d, "tiny.flac", 0.1, 99)  # < 0.2 s: skipped, empty prediction, not a failure
    texts = transcribe_many(t, paths + [d / "tiny.flac"])
    assert len(texts) == 5 and all(isinstance(x, str) for x in texts) and texts[-1] == ""
