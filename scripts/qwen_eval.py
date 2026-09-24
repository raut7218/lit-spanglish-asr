"""Score Qwen3-ASR (zero-shot hub id or a fine-tuned merged dir) on dev / Miami hold-out / synthetic long clips through
the runtime path (lit.qwen, vLLM). Dumps id/ref/hyp CSVs so `python -m lit.mbr a.csv b.csv` can combine systems.

    python scripts/qwen_eval.py --model Qwen/Qwen3-ASR-1.7B --data_dir PREP --sets dev miami_holdout long \
        --grid '[{"language": null}, {"language": "Spanish"}, {"language": "English"}]' --dump_dir DIR
    --backend transformers   # CPU smoke test
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from decode_eval import load_set  # noqa: E402
from lit import qwen  # noqa: E402
from lit.normalize import norm, wer  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--sets", nargs="+", default=["dev"])
    ap.add_argument("--grid", default="[{}]", help="JSON list of lit.qwen cfg overrides")
    ap.add_argument("--max_clips", type=int, default=300)
    ap.add_argument("--backend", default="vllm", choices=["vllm", "transformers"])
    ap.add_argument("--dump_dir")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    data = {s: load_set(Path(a.data_dir), s, a.max_clips) for s in a.sets}
    grid = json.loads(a.grid)
    cfg0 = {**qwen.DEFAULT_CFG, **grid[0]}
    if a.backend == "vllm":
        model = qwen.load_vllm(a.model, cfg0)
    else:
        import torch
        from qwen_asr import Qwen3ASRModel

        model = Qwen3ASRModel.from_pretrained(a.model, dtype=torch.float32, device_map="cpu",
                                              max_inference_batch_size=4, max_new_tokens=cfg0["max_new_tokens"])
    for g in grid:
        cfg = {**qwen.DEFAULT_CFG, **g}
        res = {}
        for s, items in data.items():
            t0 = time.time()
            hyps = qwen.transcribe(model, [x for x, _ in items], cfg)
            refs = [r for _, r in items]
            res[s] = f"{wer(refs, hyps):.4f} ({(time.time()-t0)/len(items):.2f}s/clip)"
            if a.dump_dir:
                import pandas as pd

                Path(a.dump_dir).mkdir(parents=True, exist_ok=True)
                tag = a.tag + "_".join(f"{k}{v}" for k, v in g.items() if k != "rules")
                pd.DataFrame(dict(id=range(len(items)), ref=[norm(r) for r in refs], hyp=hyps)).to_csv(
                    Path(a.dump_dir) / f"{s}_qwen{tag}.csv", index=False)
        print(f"[qwen_eval] {json.dumps(g)} -> {res}", flush=True)


if __name__ == "__main__":
    main()
