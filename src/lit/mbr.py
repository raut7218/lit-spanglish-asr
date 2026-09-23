"""Minimum-Bayes-risk system combination: per clip, keep the hypothesis with the smallest total word edit distance
to the other systems' hypotheses (the "medoid"). Different models / language tokens make different mistakes; the
medoid drops the outlier. Pure Python (ships in the runtime zip).

    python -m lit.mbr a.csv b.csv c.csv      # decode_eval dumps of the same set: per-system WER and the MBR WER
"""

from __future__ import annotations


def edit_distance(a: list[str], b: list[str]) -> int:
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def mbr_pick(hyps: list[str]) -> str:
    """Medoid of the hypotheses; ties go to the earliest (= primary) system."""
    toks = [h.split() for h in hyps]
    cost = [sum(edit_distance(t, u) for u in toks) for t in toks]
    return hyps[cost.index(min(cost))]


if __name__ == "__main__":
    import sys

    import pandas as pd

    from .normalize import wer

    assert mbr_pick(["a b c", "a b d", "a b c"]) == "a b c" and edit_distance("a b".split(), "b".split()) == 1
    dfs = [pd.read_csv(p).fillna("") for p in sys.argv[1:]]
    refs = dfs[0]["ref"].astype(str).tolist()
    for p, d in zip(sys.argv[1:], dfs):
        print(f"{wer(refs, d['hyp'].astype(str).tolist()):.4f}  {p}")
    picks = [mbr_pick([d["hyp"].astype(str).iloc[i] for d in dfs]) for i in range(len(refs))]
    print(f"{wer(refs, picks):.4f}  MBR of {len(dfs)} systems")
