"""Spelling conventions: make Bangor Miami training targets follow the dev/test annotators' conventions.

The Miami transcribers write "gonna" (820x vs 269x "going") and "ah" (603x); the dev references write
"going to" and never "ah" (a hesitation "uh" instead). A model trained on Miami inherits those spellings and
pays for it at test time, so we rewrite the targets (and the hold-out references) with a small evidence-based map.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

DEFAULT_PATH = Path(__file__).with_name("conventions.json")


def load_conventions(path=None) -> dict:
    with open(path or DEFAULT_PATH, encoding="utf-8") as f:
        c = json.load(f)
    return {"expand": c.get("expand", {}), "rename": c.get("rename", {})}


def _match_case(src: str, dst: str) -> str:
    if src.isupper() and len(src) > 1:
        return dst.upper()
    if src[:1].isupper():
        return dst[:1].upper() + dst[1:]
    return dst


def apply_conventions(text: str, conv: dict | None) -> str:
    """Whole-word, case-preserving rewrite. Punctuation and the rest of the text are untouched."""
    if not conv or not text:
        return text
    table = {**conv.get("expand", {}), **conv.get("rename", {})}
    if not table:
        return text
    pat = re.compile(r"(?<![\w'])(" + "|".join(sorted(map(re.escape, table), key=len, reverse=True)) + r")(?![\w'])", re.I)
    return pat.sub(lambda m: _match_case(m.group(1), table[m.group(1).lower()]), text)
