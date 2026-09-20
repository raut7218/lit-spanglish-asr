from lit.conventions import apply_conventions, load_conventions


def test_expand_and_rename_preserve_case_and_punctuation():
    c = load_conventions()
    assert apply_conventions("I'm gonna go. Gonna? wanna eat", c) == "I'm going to go. Going to? want to eat"
    assert apply_conventions("ah yes, Ah!", c) == "uh yes, Uh!"


def test_whole_word_only():
    c = load_conventions()
    assert apply_conventions("gonnabe wannabe ahora ahh", c) == "gonnabe wannabe ahora ahh"
    assert apply_conventions("", c) == "" and apply_conventions("x", None) == "x"
