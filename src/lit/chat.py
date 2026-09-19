"""Parse Bangor Miami CHAT (.cha) transcripts into timed, cleaned utterances."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

TIME_RE = re.compile(r"\x15(\d+)_(\d+)\x15")
LANG_TAG_RE = re.compile(r"@s:([a-z&+]+)")


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


def clean_chat_text(raw: str) -> tuple[str, bool, int, int]:
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
    s = re.sub(r"&[-+~]?\S+", " ", s)  # fragments / fillers: &e &nes &-uh
    s = re.sub(r"\(\.+\)", " ", s)  # pauses (.) (..)
    s = re.sub(r"\+[<\"/,.^+]*[.?!]*", " ", s)  # linkers / terminators: +< +" +... +//.
    s = s.replace("<", " ").replace(">", " ")

    toks, n_spa, n_eng = [], 0, 0
    for tok in s.split():
        m = LANG_TAG_RE.search(tok)
        if m:
            tag = m.group(1)
            n_spa += tag.startswith("spa")
            n_eng += tag.startswith("eng") and "&" not in tag
            tok = LANG_TAG_RE.sub("", tok)
        elif re.search(r"[A-Za-zÀ-ÿ]", tok):
            n_spa += spa_default
            n_eng += not spa_default
        tok = re.sub(r"\(([^)]*)\)", r"\1", tok)  # thi(nk) -> think, (be)cause -> because
        tok = tok.replace("_", " ") if tok.lower() != "o_k" else "okay"
        toks.append(tok)
    s = " ".join(toks)
    s = re.sub(r"[+^~≈↑↓⌈⌉⌊⌋‹›]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s, unintelligible, n_spa, n_eng


def parse_cha(path: Path) -> list[Utt]:
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
        text, unint, n_spa, n_eng = clean_chat_text(body)
        utts.append(Utt(speaker, body, text, start, end, n_spa, n_eng, unint))
    return utts
