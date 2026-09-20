"""Rank tokens by how differently the dev references and the Miami training targets use them.

    python scripts/convention_audit.py --dev DEV_METADATA.tsv --miami MIAMI_CHAT_DIR [--top 40]

Prints per-1k-word frequencies of the most divergent tokens plus spelling-variant families
(ok/okay, uh/ah/eh/um, gonna/going to ...). Use it to extend `src/lit/conventions.json`;
add a rule only when the direction is unambiguous and the counts are large.
"""
import argparse
import csv
import glob
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from lit.chat import parse_cha  # noqa: E402
from lit.normalize import norm  # noqa: E402

FAMILIES = [["gonna", "going", "wanna", "want", "gotta", "kinda"], ["ok", "okay"], ["uh", "ah", "eh", "um", "em", "ehm", "mm", "mmm", "hmm", "mhm", "mmhm"],
            ["yeah", "yes", "yep"], ["cuz", "because", "cause"], ["pos", "pues"], ["sólo", "solo"]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev", required=True)
    ap.add_argument("--miami", required=True)
    ap.add_argument("--top", type=int, default=40)
    a = ap.parse_args()
    dev, mia = Counter(), Counter()
    for r in csv.DictReader(open(a.dev, encoding="utf-8"), delimiter="\t"):
        dev.update(norm(r["transcript"]).split())
    for f in glob.glob(str(Path(a.miami) / "chat" / "*.cha")):
        for u in parse_cha(f):
            mia.update(norm(u.text).split())
    nd, nm = sum(dev.values()), sum(mia.values())
    print(f"dev {nd} words, miami {nm} words\n\ntoken       dev/1k  miami/1k")
    div = sorted(((abs(dev[w] / nd - mia[w] / nm) * 1000, w) for w in set(dev) | set(mia) if dev[w] + mia[w] >= 15), reverse=True)
    for _, w in div[: a.top]:
        print(f"{w:12s}{dev[w]/nd*1000:7.2f}{mia[w]/nm*1000:9.2f}")
    print("\nspelling families (dev count | miami count):")
    for fam in FAMILIES:
        print("  " + "  ".join(f"{w} {dev[w]}|{mia[w]}" for w in fam))


if __name__ == "__main__":
    main()
