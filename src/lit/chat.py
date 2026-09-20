"""Parse Bangor Miami CHAT (.cha) transcripts into timed, cleaned utterances."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

TIME_RE = re.compile(r"\x15(\d+)_(\d+)\x15")
LANG_TAG_RE = re.compile(r"@s:([a-z&+]+)")
FILLERS = {"um", "uh", "eh", "ah", "mm", "hmm"}


@dataclass
class Utt:
    speaker: str
    raw: str
    text: str  # cleaned, scorer-style-ready text (no CHAT markup)
    start_ms: int
    end_ms: int
    n_spa: int = 0
    n_eng: int = 0
    has_unintelligible: bool = False
    extra: dict = field(default_factory=dict)

    @property
    def dur(self) -> float:
        return (self.end_ms - self.start_ms) / 1000.0


def read_participants(path: Path) -> dict[str, str]:
    """`@Participants: PAI Paige Adult, SAR Sarah Adult` -> {code: role}."""
    for line in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith("@Participants:"):
            out = {}
            for part in line.split("\t", 1)[1].split(","):
                toks = part.split()
                if toks:
                    out[toks[0]] = " ".join(toks[1:])
            return out
    return {}


def style_text(s: str) -> str:
    """Tidy CHAT tokens into the transcript style of the dev/test set: terminators attached to the
    previous word ("well ." -> "well."), sentence-initial letters capitalised. Keeping sentence
    punctuation matters because the scorer lowercases sentence-initial letters (even "I" -> "i"):
    the model must place sentence boundaries like the references do for those tokens to match.
    """
    s = re.sub(r"\s+", " ", s).strip()
    s = re.sub(r"\s+([.?!])", r"\1", s)
    s = re.sub(r"\.{4,}", "...", s)
    s = re.sub(r"(?:^|(?<=[?!])\s+|(?<=[^.]\.)\s+)([^\W\d_])", lambda m: m.group(0)[:-1] + m.group(1).upper(), s)
    return s if s.strip(" .?!") else ""


def clean_chat_text(raw: str, collect=None) -> tuple[str, bool, int, int]:
    """CHAT main-tier text -> plain verbatim words.

    Returns (text, has_unintelligible, n_spanish_tokens, n_english_tokens).
    Retraced / repeated material is KEPT (it was actually spoken); only markup is dropped.
    """
    s = TIME_RE.sub(" ", raw)
    spa_default = "[- spa]" in s  # untagged words are Spanish in these utterances
    s = re.sub(r"\[-\s*[a-z]+\]", " ", s)
    unintelligible = bool(re.search(r"\b(xxx|yyy|www)\b", s))
    s = re.sub(r"\b(xxx|yyy|www)\b", " ", s)

    s = re.sub(r"\[[^\]]*\]", " ", s)  # [/] [//] [?] [=! laughs] [* ...]
    s = re.sub(r"&=\S+", " ", s)  # events: &=laughs
    # phonological fragments (&e, &nes) -> "e...", "nes..." : the organisers' transcripts write cut-off
    # words that way (e.g. "chul...", "tha..."), and the scorer keeps "..." attached to the token.
    s = re.sub(r"(?<![\w@])&(?![=\-+~])([A-Za-z]+):?", lambda m: m.group(1) if m.group(1).lower() in FILLERS else m.group(1) + "...", s)
    s = re.sub(r"&[-+~]\S+", " ", s)  # remaining markup
    s = re.sub(r"\(\.+\)", " ", s)  # pauses (.) (..)
    s = re.sub(r"\+[<\"/,.^+]*[.?!]*", " ", s)  # linkers / terminators: +< +" +... +//.
    s = s.replace("<", " ").replace(">", " ")

    toks, n_spa, n_eng = [], 0, 0
    for tok in s.split():
        m = LANG_TAG_RE.search(tok)
        lang = None
        if m:
            tag = m.group(1)
            n_spa += tag.startswith("spa")
            n_eng += tag.startswith("eng") and "&" not in tag
            lang = "spa" if tag.startswith("spa") else ("eng" if tag.startswith("eng") and "&" not in tag else None)
            tok = LANG_TAG_RE.sub("", tok)
        elif re.search(r"[A-Za-zÀ-ÿ]", tok):
            n_spa += spa_default
            n_eng += not spa_default
            lang = "spa" if spa_default else "eng"
        tok = re.sub(r"\(([^)]*)\)", r"\1", tok)  # thi(nk) -> think, (be)cause -> because
        tok = tok.replace("_", " ") if tok.lower() != "o_k" else "okay"
        if collect is not None and lang:
            for part in tok.split():
                w = re.sub(r"[^a-zà-ÿ']", "", part.lower())
                if w:
                    collect[(w, lang)] += 1
        toks.append(tok)
    s = " ".join(toks)
    s = re.sub(r"[+^~≈↑↓⌈⌉⌊⌋‹›]", " ", s)
    s = style_text(s)
    return s, unintelligible, n_spa, n_eng


def parse_cha(path: Path, collect=None) -> list[Utt]:
    """Return utterances that carry a time mark, in file order."""
    raw_lines = Path(path).read_text(encoding="utf-8", errors="replace").splitlines()
    tier_lines: list[str] = []
    for line in raw_lines:  # join continuation lines (start with a tab)
        if line.startswith("*"):
            tier_lines.append(line)
        elif line.startswith("\t") and tier_lines and not line.startswith("\t%"):
            if tier_lines[-1] is not None and not tier_lines[-1].startswith("%"):
                tier_lines[-1] += " " + line.strip()
        elif line.startswith("%") or line.startswith("@"):
            tier_lines.append(None)  # break continuation
    utts = []
    for line in tier_lines:
        if not line or not line.startswith("*"):
            continue
        head, _, body = line.partition(":\t")
        speaker = head[1:].strip()
        marks = TIME_RE.findall(body)
        if not marks:
            continue
        start, end = int(marks[-1][0]), int(marks[-1][1])
        if end <= start:
            continue
        text, unint, n_spa, n_eng = clean_chat_text(body, collect)
        utts.append(Utt(speaker, body, text, start, end, n_spa, n_eng, unint))
    return utts


def spa_frac_of(text: str, table: dict) -> float | None:
    """Estimated share of Spanish among the tokens of `text` (token -> P(spanish) table built from Miami tags)."""
    ps = [table[w] for w in re.findall(r"[a-zà-ÿ']+", text.lower()) if w in table]
    return round(sum(ps) / len(ps), 3) if ps else None


def token_language_table(counts, min_count: int = 3) -> dict:
    """{(word, lang): n} -> {word: P(spa)} for words seen >= min_count times with an unambiguous tag."""
    tot: dict = {}
    for (w, lang), n in counts.items():
        a = tot.setdefault(w, [0, 0])
        a[0 if lang == "spa" else 1] += n
    return {w: a[0] / (a[0] + a[1]) for w, a in tot.items() if a[0] + a[1] >= min_count}
