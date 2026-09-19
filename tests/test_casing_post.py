from lit.casing import apply_casing, build_lexicon
from lit.normalize import norm
from lit.postprocess import collapse_loops, postprocess


def test_lexicon_and_apply():
    texts = ["yo digo Okay pero okay okay okay no", "you know okay Miami is nice", "we went to Miami then Miami again Miami"] * 2
    lex = build_lexicon(texts, min_count=3, purity=0.8)
    assert lex.get("miami") == "Miami"
    assert lex.get("okay") == "okay"
    assert apply_casing("so Okay we saw miami", lex) == "so okay we saw Miami"
    # sentence-initial untouched
    assert apply_casing("Okay so we go", lex).startswith("Okay")


def test_collapse_loops():
    assert collapse_loops("y " * 20).split() == ["y"] * 4
    assert collapse_loops("no no no no no no i said").split()[:4] == ["no"] * 4
    assert collapse_loops("a b c a b c a b c a b c a b c a b c d") == "a b c a b c a b c a b c d"
    assert collapse_loops("normal text here") == "normal text here"


def test_postprocess_matches_scorer_form():
    assert postprocess("<|es|> Okay? I see, ¿no?") == norm("Okay? I see, ¿no?")
