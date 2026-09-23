"""Convention-matching rewrite rules applied to the (already normalised) hypothesis text.

The dev/test annotators and the Bangor Miami transcribers spell some things differently
("going to" vs "gonna", "uh" vs "ah"), so a model trained on Miami inherits Miami's conventions.
Each rule is a pure text->text function; `scripts/rule_search.py` measures every rule on dev
predictions and `infer_config.json["rules"]` lists the ones that ship.
"""

from __future__ import annotations

import re

_CONTR_I = re.compile(r"^i(?=$|')")


def expand_gonna(t: str) -> str:
    t = re.sub(r"\bgonna\b", "going to", t)
    t = re.sub(r"\bwanna\b", "want to", t)
    t = re.sub(r"\bgotta\b", "got to", t)
    t = re.sub(r"\bkinda\b", "kind of", t)
    return t


def capital_i(t: str) -> str:
    """Capitalise "i" / "i'm" / "i'll" (the scorer only lowercases a sentence-INITIAL I, and our
    normalised text has no sentence marks left, so an "i" in the middle can never match a ref "I")."""
    toks = t.split()
    return " ".join(w if k == 0 or not _CONTR_I.match(w) else "I" + w[1:] for k, w in enumerate(toks))


def ah_to_uh(t: str) -> str:
    return re.sub(r"\bah\b", "uh", t)


def ok_to_okay(t: str) -> str:
    return re.sub(r"\bok\b", "okay", t)


def drop_ah(t: str) -> str:
    return re.sub(r"\bah\b", " ", t)


def drop_um(t: str) -> str:
    return re.sub(r"\b(um|ehm|em|mmm)\b", " ", t)


def drop_hmm(t: str) -> str:
    return re.sub(r"\b(hmm|mhm|mmhm|mm)\b", " ", t)


RULES = {
    "expand_gonna": expand_gonna,
    "capital_i": capital_i,
    "ah_to_uh": ah_to_uh,
    "ok_to_okay": ok_to_okay,
    "drop_ah": drop_ah,
    "drop_um": drop_um,
    "drop_hmm": drop_hmm,
}


def apply_rules(text: str, names) -> str:
    for n in names or []:
        text = RULES[n](text)
    return re.sub(r"\s+", " ", text).strip()
