"""End-to-end training smoke test with a tiny Whisper on a synthetic mini dataset (set LIT_E2E=1; needs the model
files: whisper-tiny is downloaded/cached). Exercises the sampler, turn/window/non-speech kinds, GPU aug, the LoRA
freeze regex, EMA, multi-set validation, checkpoint selection and adapter export."""
import json
import os
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

pytest.importorskip("torch")
pytestmark = pytest.mark.skipif(os.environ.get("LIT_E2E") != "1", reason="set LIT_E2E=1 to run (downloads whisper-tiny)")

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


def test_train_v2_end_to_end(tmp_path):
    from lit.train import main

    d = tmp_path / "data"
    train = _rows(d, "clips", 10, "turn", 1) + _rows(d, "clips", 6, "window", 2) + _rows(d, "clips", 2, "nonspeech", 3)
    hold = _rows(d, "hold", 6, "turn", 4) + _rows(d, "hold", 4, "window", 5)
    dev = _rows(d, "dev_clips", 8, "turn", 6, dev=True)
    _write(d / "train.jsonl", train)
    _write(d / "miami_holdout.jsonl", hold)
    _write(d / "dev.jsonl", dev)
    out = tmp_path / "run"
    main(["--config", str(Path(__file__).resolve().parents[1] / "configs" / "smoke_v2.yaml"), "--set", f"data_dir={d}", f"out_dir={out}",
          "max_steps=4", "eval_steps=2", "save_steps=2", "batch_size=2", "grad_accum=1", "epoch_samples=16", "num_workers=0",
          "val.holdout_turn=4", "val.holdout_window=3", "dev_in_train=spk1"])
    fj = json.loads((out / "final_adapter" / "final.json").read_text())
    assert fj["winner"] in ("single", "avg") and "holdout_turn" in fj["scores"][fj["winner"]]
    assert (out / "final_adapter" / "adapter_config.json").exists()
    assert len((out / "experiments.jsonl").read_text().splitlines()) >= 2
