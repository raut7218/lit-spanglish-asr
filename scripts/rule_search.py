"""Score post-processing rules on a dev predictions dump (`lit.evaluate --dump`).

    python scripts/rule_search.py preds.csv

Prints baseline WER, each rule's individual delta, then a greedy combination (keeps a rule only
if it lowers WER), plus ref/hyp counts of the tokens the rules touch.
"""
import sys
from collections import Counter
from pathlib import Path

import jiwer
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from lit.rules import RULES, apply_rules  # noqa: E402

df = pd.read_csv(sys.argv[1]).fillna("")
refs, hyps = df["ref"].astype(str).tolist(), df["hyp"].astype(str).tolist()
W = lambda h: jiwer.wer(refs, h)
base = W(hyps)
print(f"baseline WER {base:.4f}")
rc, hc = Counter(w for r in refs for w in r.split()), Counter(w for h in hyps for w in h.split())
for w in ["gonna", "going", "wanna", "want", "um", "uh", "ah", "eh", "mm", "hmm", "i", "I", "okay", "ok"]:
    print(f"  {w:6s} ref {rc[w]:4d} hyp {hc[w]:4d}")
print("\nindividual rules:")
delta = {}
for n in RULES:
    w = W([apply_rules(h, [n]) for h in hyps])
    delta[n] = w - base
    print(f"  {n:14s} {w:.4f}  ({w-base:+.4f})")
chosen, cur = [], base
for n in sorted(RULES, key=lambda k: delta[k]):
    w = W([apply_rules(h, chosen + [n]) for h in hyps])
    if w < cur - 1e-9:
        chosen.append(n)
        cur = w
print(f"\ngreedy set: {chosen}  -> WER {cur:.4f}  ({cur-base:+.4f})")
