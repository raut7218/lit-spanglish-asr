"""Hallucination / repetition guards + final scorer-style clean-up applied to every prediction."""

from __future__ import annotations

import re

from .casing import apply_casing
from .normalize import norm


def collapse_loops(text: str, max_repeat: int = 4) -> str:
    """Collapse degenerate loops ("y y y y y y ..." or repeated n-grams) to `max_repeat` repeats."""
    words = text.split()
    changed = True
    while changed:
        changed = False
        for n in range(1, 7):
            i = 0
            while i + n * (max_repeat + 1) <= len(words):
                unit = words[i : i + n]
                reps = 1
                while words[i + reps * n : i + (reps + 1) * n] == unit:
                    reps += 1
                if reps > max_repeat:
                    words = words[: i + max_repeat * n] + words[i + reps * n :]
                    changed = True
                i += 1
    return " ".join(words)


def postprocess(text: str, lexicon: dict | None = None) -> str:
    text = re.sub(r"<\|[^|]*\|>", " ", text or "")
    text = collapse_loops(text)
    if lexicon:
        text = apply_casing(text, lexicon)
    return norm(text)
