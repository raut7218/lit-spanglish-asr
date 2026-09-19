import importlib.util
from pathlib import Path

from lit.normalize import norm, normalize_text

SAMPLES = [
    "Pues sí iría si iría but I have to know some of the details like la fecha y el horario you know because like I have to work.",
    "Okay? I will see you in en unos minutos.",
    "Oye pero I'm talking about books for me [laugh] son libros... ne... yo ya lo hubiera comprado.",
    "¿Qué andan haciendo ustedes y los kiddos bueno tú y los kiddos? I just heard like um",
    "Oh yeah we were starving y abrí el refri dije \"bueno, vamos a comer los leftovers.\"",
    "La verdad no quisiera but este I have to do it (?) so we'll see (um) how it goes — I'm not sure what [?]",
    "Áhora Ñoño ÉL is NASA. USA! Ok",
    "",
]


def _official():
    p = Path(__file__).resolve().parents[1] / "scripts" / "official_score.py"
    spec = importlib.util.spec_from_file_location("official_score", p)
    m = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(m)
    except ModuleNotFoundError:  # jiwer/typer/pandas missing: fall back to source-level exec of the fn only
        import re

        src = p.read_text()
        start = src.index("def _lowercase_sentence_initial")
        end = src.index("def _word_error_rate")
        ns = {"re": re}
        exec(src[start:end], ns)
        return ns["normalize_text"]
    return m.normalize_text


def test_matches_official_scorer():
    off = _official()
    for s in SAMPLES:
        assert normalize_text(s) == off(s), s


def test_idempotent():
    for s in SAMPLES:
        assert norm(norm(s)) == norm(s), s


def test_sentence_initial_lowercased_but_midsentence_case_kept():
    assert norm("Okay? I will see Target") == "okay I will see Target"
