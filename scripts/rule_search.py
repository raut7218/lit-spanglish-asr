"""Score post-processing rules on prediction dumps (`lit.evaluate --dump`), on SEVERAL sets at once.

    python scripts/rule_search.py dev_preds.csv holdout_preds.csv [--out chosen_rules.json]

A rule is kept only if it lowers the pooled WER and does not raise the WER of any single set (so a rule tuned to
one conversation cannot slip in). Prints per-set WER for the baseline, each rule and the greedy combination.
"""
import argparse
import json
import sys
from pathlib import Path

import jiwer
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from lit.rules import RULES, apply_rules  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("csvs", nargs="+")
ap.add_argument("--out")
ap.add_argument("--primary", help="stem of the csv that matches the test domain (default: the first csv)")
ap.add_argument("--tol", type=float, default=0.0,
                help="a NON-primary set may get this much worse (WER) if the primary set improves (Miami conventions differ from dev/test)")
a = ap.parse_args()
sets = {}
for p in a.csvs:
    df = pd.read_csv(p).fillna("")
    sets[Path(p).stem] = (df["ref"].astype(str).tolist(), df["hyp"].astype(str).tolist())


def score(rules):
    per = {k: jiwer.wer(r, [apply_rules(h, rules) for h in hy]) for k, (r, hy) in sets.items()}
    allr = [x for r, _ in sets.values() for x in r]
    allh = [apply_rules(x, rules) for _, hy in sets.values() for x in hy]
    return per, jiwer.wer(allr, allh)


base, base_all = score([])
print("baseline:", {k: round(v, 4) for k, v in base.items()}, "pooled", round(base_all, 4))
delta = {}
for n in RULES:
    per, allw = score([n])
    delta[n] = allw - base_all
    print(f"  {n:14s}", {k: f"{v-base[k]:+.4f}" for k, v in per.items()}, f"pooled {allw-base_all:+.4f}")
primary = a.primary or next(iter(sets))
chosen, cur, cur_per = [], base_all, base
for n in sorted(RULES, key=lambda k: delta[k]):
    per, allw = score(chosen + [n])
    if allw < cur - 1e-9 and all(per[k] <= cur_per[k] + (1e-9 if k == primary else a.tol) for k in per):
        chosen.append(n)
        cur, cur_per = allw, per
print(f"chosen: {chosen}  pooled {cur:.4f} ({cur-base_all:+.4f}); per set", {k: round(v, 4) for k, v in cur_per.items()})
if a.out:
    Path(a.out).write_text(json.dumps(chosen))
