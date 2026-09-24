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


def align(a: list[str], b: list[str]):
    """Levenshtein alignment of b to a: for each position i of a the aligned word of b ("" = deleted), and for each
    gap g (before a[g], g = len(a) is the end) the words b inserts there."""
    n, m = len(a), len(b)
    D = [list(range(m + 1))] + [[i] + [0] * m for i in range(1, n + 1)]
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            D[i][j] = min(D[i - 1][j] + 1, D[i][j - 1] + 1, D[i - 1][j - 1] + (a[i - 1] != b[j - 1]))
    sub, ins = [""] * n, [[] for _ in range(n + 1)]
    i, j = n, m
    while i or j:
        if i and j and D[i][j] == D[i - 1][j - 1] + (a[i - 1] != b[j - 1]):
            sub[i - 1] = b[j - 1]; i -= 1; j -= 1
        elif i and D[i][j] == D[i - 1][j] + 1:
            i -= 1
        else:
            ins[i].insert(0, b[j - 1]); j -= 1
    return sub, ins


def rover(hyps: list[str], weights=None) -> str:
    """Word-level voting (pivot ROVER): align every hypothesis to the medoid, vote per pivot word (a deletion is a vote
    for dropping it) and per gap (inserted words); ties keep the pivot. Unlike mbr_pick it can fix an error that no
    single system gets right."""
    w = list(weights or [1.0] * len(hyps))
    piv = mbr_pick(hyps).split()
    slot_votes = [dict() for _ in piv]
    gap_votes = [dict() for _ in range(len(piv) + 1)]
    for h, wt in zip(hyps, w):
        sub, ins = align(piv, h.split())
        for k, x in enumerate(sub):
            slot_votes[k][x] = slot_votes[k].get(x, 0.0) + wt
        for g, xs in enumerate(ins):
            key = " ".join(xs)
            gap_votes[g][key] = gap_votes[g].get(key, 0.0) + wt
    eps = 1e-9  # tie-break: the pivot's own choice (its word, no insertion)
    out = []
    for g in range(len(piv) + 1):
        best = max(gap_votes[g].items(), key=lambda kv: kv[1] + (eps if kv[0] == "" else 0.0))[0]
        if best:
            out.append(best)
        if g < len(piv):
            best = max(slot_votes[g].items(), key=lambda kv: kv[1] + (eps if kv[0] == piv[g] else 0.0))[0]
            if best:
                out.append(best)
    return " ".join(out)


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
    rv = [rover([d["hyp"].astype(str).iloc[i] for d in dfs]) for i in range(len(refs))]
    print(f"{wer(refs, rv):.4f}  ROVER of {len(dfs)} systems")
