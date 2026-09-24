from lit.mbr import align, rover


def test_align():
    sub, ins = align("a b c".split(), "a x c d".split())
    assert sub == ["a", "x", "c"] and ins == [[], [], [], ["d"]]
    sub, ins = align("a b c".split(), "a c".split())
    assert sub == ["a", "", "c"]


def test_rover_fixes_what_no_single_system_gets_right():
    hyps = ["the cat sat on mat", "a cat sat on the mat", "the cat sit on the mat"]
    # medoid picks one whole string; each is wrong somewhere; the vote gets every word right
    assert rover(hyps) == "the cat sat on the mat"


def test_rover_ties_keep_pivot_and_handles_empty():
    assert rover(["x y", "x z"]) in ("x y", "x z")
    assert rover(["", "", "a"]) == ""
    assert rover(["same words", "same words"]) == "same words"
