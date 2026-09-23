from lit.chat import clean_chat_text, parse_cha

CHA = """@UTF8
@Begin
@Languages:\teng, spa
@Participants:\tPAI Paige Adult, SAR Sarah Adult, OSE non_participant Adult
*PAI:\toh <now that I> [/] now that I remember antes@s:spa de@s:spa que@s:spa se@s:spa me@s:spa olvide@s:spa . \x156_2949\x15
%aut:\toh.IM now.ADV
*SAR:\tsubway passes to where to New_York ? \x159384_11073\x15
*PAI:\t[- spa] este thi(nk) o_k@s:eng&spa xxx
\tmore words &=laughs +... \x1512000_13000\x15
*PAI:\tno time mark here .
@End
"""


def test_clean_basic():
    t, unint, spa, eng = clean_chat_text("<now that I> [/] now that I remember antes@s:spa de@s:spa .")
    assert t == "Now that I now that I remember antes de."
    assert not unint and spa == 2 and eng == 7


def test_clean_markup():
    t, unint, *_ = clean_chat_text("[- spa] este thi(nk) o_k@s:eng&spa xxx New_York &=laughs (.) +... &e")
    assert t == "Este think okay New York e..."
    assert unint


def test_parse(tmp_path):
    p = tmp_path / "x.cha"
    p.write_text(CHA, encoding="utf-8")
    u = parse_cha(p)
    assert len(u) == 3  # the utterance without a time mark is dropped
    assert (u[0].speaker, u[0].start_ms, u[0].end_ms) == ("PAI", 6, 2949)
    assert u[2].text == "Este think okay more words" and u[2].has_unintelligible


def test_styled_targets_and_fragments():
    t, *_ = clean_chat_text("I think &e +... es@s:spa &nes que &um it's &=laughs fine ? yeah . ")
    assert t == "I think e... es nes... que um it's fine? Yeah."
    assert clean_chat_text("xxx .")[0] == ""


def test_spanish_matrix_file(tmp_path):
    p = tmp_path / "h.cha"
    p.write_text("@Languages:\tspa, eng\n*ASH:\tme dice que trabaja en furniture@s:eng . \x151_5369\x15\n"
                 "*JAC:\t[- eng] yeah I know . \x156000_7000\x15\n", encoding="utf-8")
    u = parse_cha(p)
    assert (u[0].n_spa, u[0].n_eng) == (5, 1) and (u[1].n_spa, u[1].n_eng) == (0, 3)
