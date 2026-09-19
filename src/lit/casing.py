"""Data-driven casing fix-up.

The scorer only lowercases *sentence-initial* letters, so "Okay" mid-sentence vs "okay" is a WER
error. We learn, from the provided transcripts only, how each word is usually cased when it is NOT
sentence-initial, and force the prediction to follow that majority.
"""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict

TOKEN_RE = re.compile(r"[^\W\d_]+(?:['’][^\W\d_]+)*", re.UNICODE)


def build_lexicon(texts, min_count: int = 3, purity: float = 0.9) -> dict[str, str]:
    """{lowercase_word: preferred_surface_form} for words with a clearly dominant mid-sentence casing."""
    forms: dict[str, Counter] = defaultdict(Counter)
    for text in texts:
        prev_end = True  # start of text counts as sentence start
        for m in re.finditer(r"\S+", text):
            tok = m.group(0)
            core = TOKEN_RE.search(tok)
            if core and not prev_end:
                w = core.group(0)
                forms[w.lower()][w] += 1
            prev_end = bool(re.search(r"[.!?…—]+[\"')\]]*$", tok))
    lex = {}
    for low, c in forms.items():
        total = sum(c.values())
        surface, n = c.most_common(1)[0]
        if total >= min_count and n / total >= purity:
            lex[low] = surface
    return lex


def apply_casing(text: str, lex: dict[str, str]) -> str:
    """Rewrite non-sentence-initial words to their dominant casing. Sentence-initial words are left
    alone (the scorer lowercases them anyway)."""
    out, prev_end, pos = [], True, 0
    for m in re.finditer(r"\S+", text):
        out.append(text[pos : m.start()])
        tok = m.group(0)
        core = TOKEN_RE.search(tok)
        if core and not prev_end:
            w = core.group(0)
            fixed = lex.get(w.lower())
            if fixed and fixed != w and not (w.isupper() and len(w) > 1):  # keep acronyms
                tok = tok[: core.start()] + fixed + tok[core.end() :]
        out.append(tok)
        prev_end = bool(re.search(r"[.!?…—]+[\"')\]]*$", tok))
        pos = m.end()
    out.append(text[pos:])
    return "".join(out)


def save_lexicon(lex, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(lex, f, ensure_ascii=False, sort_keys=True)


def load_lexicon(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)
