"""Decoding sweep on the dev clips through the SHIPPED faster-whisper path (lit.infer.Transcriber).

    python scripts/decode_sweep.py --ct2 EXPORT/ct2 --prepared PREP --dev_raw RAW/enspa_dev \
        [--rules capital_i,ah_to_uh,expand_gonna] [--only beam1,beam5] [--out sweep.json]

The model is loaded once; each config mutates the Transcriber cfg, decodes all dev clips (mp3, as the platform
provides them), and reports scorer-exact WER (overall + per speaker) and seconds/clip. A config that faster-whisper
rejects is reported as an error instead of stopping the sweep. Dev is 155 clips (+-0.01): only trust differences
that are consistent across both speakers.
"""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import jiwer  # noqa: E402
from faster_whisper.audio import decode_audio  # noqa: E402

from lit.infer import Transcriber  # noqa: E402
from lit.normalize import norm  # noqa: E402
from lit.postprocess import postprocess  # noqa: E402

PROMPT_ES_EN = "Pues, uh, I think we should go al mall, eh, y luego vamos a comer, okay? Yeah, no sé, like, it's going to be fun."

GRID = {
    "beam1": dict(beam_size=1),
    "beam5": dict(beam_size=5),  # shipped baseline
    "beam8": dict(beam_size=8),
    "beam5_lp0.8": dict(beam_size=5, extra={"length_penalty": 0.8}),
    "beam5_lp1.3": dict(beam_size=5, extra={"length_penalty": 1.3}),
    "beam5_pat2": dict(beam_size=5, extra={"patience": 2.0}),
    "lang_auto": dict(beam_size=5, language=None),
    "lang_en": dict(beam_size=5, language="en"),
    "prompt": dict(beam_size=5, extra={"initial_prompt": PROMPT_ES_EN}),
    "reppen1.03": dict(beam_size=5, repetition_penalty=1.03),
    "prompt_pat2": dict(beam_size=5, extra={"initial_prompt": PROMPT_ES_EN, "patience": 2.0}),
    "prompt_beam8": dict(beam_size=8, extra={"initial_prompt": PROMPT_ES_EN}),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ct2", required=True)
    ap.add_argument("--prepared", required=True)
    ap.add_argument("--dev_raw", required=True)
    ap.add_argument("--rules", default="capital_i,ah_to_uh,expand_gonna")
    ap.add_argument("--only", default="", help="comma-separated GRID names (default: all)")
    ap.add_argument("--speaker", help="score only this dev speaker (use the speaker the model was NOT trained on)")
    ap.add_argument("--out")
    a = ap.parse_args()
    rules = [r for r in a.rules.split(",") if r]
    rows = [json.loads(l) for l in open(Path(a.prepared) / "dev.jsonl", encoding="utf-8")]
    if a.speaker:
        rows = [r for r in rows if str(r["speaker"]) == a.speaker]
    t = Transcriber(a.ct2, {"compute_type": "float16"})
    audio = [decode_audio(str(Path(a.dev_raw) / "clips" / r["orig"]), sampling_rate=16000) for r in rows]
    names = [n for n in (a.only.split(",") if a.only else GRID) if n]
    base_cfg = dict(t.cfg)
    results = {}
    for name in names:
        t.cfg = dict(base_cfg, **GRID[name])
        t.beam = t.cfg["beam_size"]
        t0 = time.time()
        try:
            hyps = [postprocess(t.transcribe_array(x), None, rules) for x in audio]
        except Exception as e:  # noqa: BLE001
            results[name] = {"error": f"{type(e).__name__}: {str(e)[:120]}"}
            print(f"{name:14s} ERROR {results[name]['error']}", flush=True)
            continue
        el = (time.time() - t0) / len(rows)
        refs = [norm(r["ref"]) for r in rows]
        res = {"wer": round(jiwer.wer(refs, hyps), 4), "s_per_clip": round(el, 2)}
        for spk in sorted({str(r["speaker"]) for r in rows}):
            idx = [i for i, r in enumerate(rows) if str(r["speaker"]) == spk]
            res[f"spk{spk}"] = round(jiwer.wer([refs[i] for i in idx], [hyps[i] for i in idx]), 4)
        results[name] = res
        print(f"{name:14s} {res}", flush=True)
    if a.out:
        Path(a.out).write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
