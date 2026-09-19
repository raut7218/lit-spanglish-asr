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
    assert t == "now that I now that I remember antes de ."
    assert not unint and spa == 2 and eng == 6


def test_clean_markup():
    t, unint, *_ = clean_chat_text("[- spa] este thi(nk) o_k@s:eng&spa xxx New_York &=laughs (.) +... &e")
    assert t == "este think okay New York"
    assert unint


def test_parse(tmp_path):
    p = tmp_path / "x.cha"
    p.write_text(CHA, encoding="utf-8")
    u = parse_cha(p)
    assert len(u) == 3  # the utterance without a time mark is dropped
    assert (u[0].speaker, u[0].start_ms, u[0].end_ms) == ("PAI", 6, 2949)
    assert u[2].text == "este think okay more words" and u[2].has_unintelligible
