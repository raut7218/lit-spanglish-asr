"""Hallucination / repetition guards + final scorer-style clean-up applied to every prediction."""

from __future__ import annotations

import re

from .casing import apply_casing
from .normalize import norm
from .rules import apply_rules


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


def postprocess(text: str, lexicon: dict | None = None, rules=None) -> str:
    text = re.sub(r"<\|[^|]*\|>", " ", text or "")
    text = collapse_loops(text)
    if lexicon:
        text = apply_casing(text, lexicon)
    text = norm(text)
    return apply_rules(text, rules) if rules else text


# Strings pandas.read_csv turns into NaN by default. A transcript equal to one of these (or empty) would be
# read back as a missing value and the platform rejects the whole submission as "not valid".
NA_LIKE = {"", "#n/a", "#n/a n/a", "#na", "-1.#ind", "-1.#qnan", "-nan", "1.#ind", "1.#qnan", "<na>", "n/a",
           "na", "null", "nan", "none"}
PLACEHOLDER = "no"  # costs the same WER as an empty hypothesis (1 error vs N deletions) but is never NaN


def finalize_transcript(text) -> str:
    """Last step before the CSV: a plain, non-empty, single-line string that survives pandas/csv round trips."""
    t = "" if text is None else str(text)
    t = re.sub(r"[\x00-\x1f\x7f]", " ", t)  # control chars (incl. newlines / tabs)
    t = " ".join(t.split())  # any unicode whitespace (\u2028, \x85, nbsp ...) -> single space
    if not t:
        return PLACEHOLDER
    if t.lower() in NA_LIKE:
        return t + "."  # the scorer turns "." into a space, so the scored text is unchanged
    return t
