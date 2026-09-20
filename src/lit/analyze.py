"""Error analysis for a predictions CSV written by `lit.evaluate --dump` (columns: id, ref, hyp).

    python -m lit.analyze preds.csv [--top 25] [--worst 8]

Prints corpus WER split into substitutions / deletions / insertions, the most frequent word-level
confusions, the most-deleted and most-inserted words, and the worst clips. Refs/hyps in the CSV are
already scorer-normalised.
"""

from __future__ import annotations

import argparse
from collections import Counter

import jiwer
import pandas as pd


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--top", type=int, default=25)
    ap.add_argument("--worst", type=int, default=8)
    a = ap.parse_args(argv)
    df = pd.read_csv(a.csv).fillna("")
    refs, hyps = df["ref"].astype(str).tolist(), df["hyp"].astype(str).tolist()
    out = jiwer.process_words(refs, hyps)
    n = sum(len(r.split()) for r in refs)
    print(f"WER {out.wer:.4f}  ({n} ref words)  sub {out.substitutions/n:.4f}  del {out.deletions/n:.4f}  ins {out.insertions/n:.4f}")

    subs, dels, inss = Counter(), Counter(), Counter()
    for r, h, al in zip(refs, hyps, out.alignments):
        rw, hw = r.split(), h.split()
        for c in al:
            if c.type == "substitute":
                for i, j in zip(range(c.ref_start_idx, c.ref_end_idx), range(c.hyp_start_idx, c.hyp_end_idx)):
                    subs[(rw[i], hw[j])] += 1
            elif c.type == "delete":
                for i in range(c.ref_start_idx, c.ref_end_idx):
                    dels[rw[i]] += 1
            elif c.type == "insert":
                for j in range(c.hyp_start_idx, c.hyp_end_idx):
                    inss[hw[j]] += 1
    print(f"\ntop substitutions (ref -> hyp):")
    for (r, h), c in subs.most_common(a.top):
        print(f"  {c:3d}  {r!r} -> {h!r}")
    print("\ntop deleted ref words:", ", ".join(f"{w}({c})" for w, c in dels.most_common(a.top)))
    print("top inserted hyp words:", ", ".join(f"{w}({c})" for w, c in inss.most_common(a.top)))

    per = []
    for i, (r, h) in enumerate(zip(refs, hyps)):
        if r.strip():
            o = jiwer.process_words(r, h)
            per.append((o.substitutions + o.deletions + o.insertions, len(r.split()), i))
    per.sort(reverse=True)
    print(f"\nworst clips by error count:")
    for err, nw, i in per[: a.worst]:
        print(f"  [{df['id'].iloc[i]}] {err}/{nw} errors\n    REF: {refs[i][:200]}\n    HYP: {hyps[i][:200]}")
    # digit / number mismatch and "..." fragment stats (scorer quirks)
    dig_ref = sum(any(ch.isdigit() for ch in w) for r in refs for w in r.split())
    ell_ref = sum("..." in w for r in refs for w in r.split())
    ell_hyp = sum("..." in w for h in hyps for w in h.split())
    print(f"\nref tokens with digits: {dig_ref} | with '...': ref {ell_ref} hyp {ell_hyp}")


if __name__ == "__main__":
    main()
