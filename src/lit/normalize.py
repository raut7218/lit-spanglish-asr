"""Text normalisation.

`normalize_text` is a VERBATIM copy of the function in the organisers' `score.py`
(drivendataorg/lost-in-transcription-runtime, 2026-08-19). Training targets, validation
and post-processing all go through it, so we optimise exactly what the leaderboard scores.
`tests/test_normalize.py` checks it against `scripts/official_score.py` on every run.
"""

import re


def _lowercase_sentence_initial(match):
    delimiter, first, second = match.group(1), match.group(2), match.group(3)
    if first.isupper() and not (second and second.isupper()):
        first = first.lower()
    return delimiter + first + second


def normalize_text(text):
    BRACKETED = re.compile(r"\[[^\]]+\]")
    UNINTELLIGIBLE_PAREN = re.compile(r"\(\?+\)")
    WORD_PAREN = re.compile(r"\(([^()]*)\)")
    PUNCTUATION_OTHER = re.compile('[¿¡";:]+')
    COMMA = re.compile(",+")
    SENTENCE_INITIAL = re.compile(r"(^\s*|[.!?—]\s*)([^\W\d_])([^\W\d_]?)")
    SENTENCE_END = re.compile("[!?]+")
    MULTISPACE = re.compile("  +")

    text = text.replace("~", "")
    text = re.sub(BRACKETED, " ", text)
    text = re.sub(UNINTELLIGIBLE_PAREN, " ", text)
    text = re.sub(WORD_PAREN, r"\1", text)
    text = text.replace("#x27;", "'")
    text = re.sub(PUNCTUATION_OTHER, " ", text)
    text = re.sub(SENTENCE_INITIAL, _lowercase_sentence_initial, text)
    text = text.replace("—", ", ")
    text = re.sub(COMMA, " ", text)
    text = re.sub(SENTENCE_END, " ", text)
    text = text.replace("...", "!ELLIPSIS!").replace(".", " ").replace("!ELLIPSIS!", "...")
    while " ... " in text:
        text = text.replace(" ... ", " ")
    text = re.sub(MULTISPACE, " ", text)
    return text


def norm(text: str) -> str:
    """Scorer normalisation plus strip: what we compare and what we train on."""
    return normalize_text(text or "").strip()


def wer(refs, hyps) -> float:
    """Corpus-level WER exactly as `score.py` computes it (normalise both sides, jiwer)."""
    import jiwer

    return jiwer.wer([norm(r) for r in refs], [norm(h) for h in hyps])
